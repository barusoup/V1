// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Artist and album pages: InnerTube's `browse` endpoint, read into the parts the app shows.
//!
//! An artist's channel is a list of shelves: its songs first, then carousels of albums and singles.
//! An album is a header and one shelf of tracks. Each shelf is read on its own, so a shelf this
//! parser does not know about is skipped rather than failing the whole page.

use serde_json::{json, Value};

use crate::clients::ClientKind;
use crate::endpoints::{collect_key, row_hit, song_from_row};
use crate::error::{Error, Result};
use crate::models::SongHit;
use crate::transport::{Auth, InnerTube};

/// An artist's channel: the name and picture, the songs, and the album shelves.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArtistPage {
    pub browse_id: String,
    pub name: String,
    pub thumbnail: Option<String>,
    /// What the songs shelf is called on the page (usually "Songs").
    pub songs_title: String,
    pub songs: Vec<SongHit>,
    /// The album shelves, in the order the page gives them ("Albums", "Singles & EPs", …).
    pub shelves: Vec<Shelf>,
}

/// One carousel of albums or singles on an artist's page.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Shelf {
    pub title: String,
    pub albums: Vec<AlbumCard>,
}

/// An album or single, as a tile on an artist's page.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AlbumCard {
    pub browse_id: String,
    pub title: String,
    /// Kind and year as the page words them, e.g. `Album  ·  2001`. May be empty.
    pub meta: String,
    pub thumbnail: Option<String>,
}

/// An artist, as a result of a search.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArtistCard {
    /// The `UC…` browse id of the artist's channel.
    pub browse_id: String,
    pub name: String,
    /// What the row says beside the name, e.g. `9.06M monthly audience`. May be empty.
    pub meta: String,
    pub thumbnail: Option<String>,
}

/// An album: its header, and its tracks in order.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AlbumPage {
    pub browse_id: String,
    pub title: String,
    pub artist: String,
    pub artist_id: Option<String>,
    /// Kind, year and length as the page words them, e.g. `Album  ·  2001  ·  10 songs`. May be empty.
    pub meta: String,
    pub thumbnail: Option<String>,
    pub tracks: Vec<SongHit>,
}

impl InnerTube {
    /// An artist's channel page, by its `UC…` browse id.
    pub async fn artist_page(&self, browse_id: &str) -> Result<ArtistPage> {
        let value = self.browse(browse_id).await?;
        let page = parse_artist(browse_id, &value);
        if page.name.is_empty() && page.songs.is_empty() && page.shelves.is_empty() {
            return Err(Error::NoContent { endpoint: "browse".to_owned() });
        }
        Ok(page)
    }

    /// An album page, by its `MPREb_…` browse id.
    pub async fn album_page(&self, browse_id: &str) -> Result<AlbumPage> {
        let value = self.browse(browse_id).await?;
        let page = parse_album(browse_id, &value);
        if page.title.is_empty() && page.tracks.is_empty() {
            return Err(Error::NoContent { endpoint: "browse".to_owned() });
        }
        Ok(page)
    }

    /// `browse` as the web client, anonymous: viewing a page does not write to the account.
    async fn browse(&self, browse_id: &str) -> Result<Value> {
        self.post(
            "browse",
            ClientKind::WebRemix,
            Auth::Anonymous,
            json!({ "browseId": browse_id }),
        )
        .await
    }
}

/// Read an artist's channel response. Missing parts come back empty, never as an error.
pub fn parse_artist(browse_id: &str, root: &Value) -> ArtistPage {
    let header = header(root);
    let name = header.and_then(|h| text_field(h, "title")).unwrap_or_default();
    let mut page = ArtistPage {
        browse_id: browse_id.to_owned(),
        thumbnail: header.and_then(thumbnail_of),
        songs_title: "Songs".to_owned(),
        name,
        ..Default::default()
    };

    for section in sections(root) {
        if let Some(shelf) = section.get("musicShelfRenderer") {
            // The first list of songs is the top songs; later song lists are not shown.
            if page.songs.is_empty() {
                if let Some(title) = text_field(shelf, "title") {
                    page.songs_title = title;
                }
                page.songs = shelf_rows(shelf).filter_map(song_from_row).collect();
                for song in &mut page.songs {
                    credit(song, &page.name, Some(browse_id));
                }
            }
        } else if let Some(carousel) = section.get("musicCarouselShelfRenderer") {
            let albums: Vec<AlbumCard> = carousel_items(carousel).filter_map(album_card).collect();
            if !albums.is_empty() {
                let title = carousel
                    .get("header")
                    .and_then(|h| h.get("musicCarouselShelfBasicHeaderRenderer"))
                    .and_then(|h| text_field(h, "title"))
                    .unwrap_or_default();
                page.shelves.push(Shelf { title, albums });
            }
        }
    }
    page
}

/// Read an album's response. The tracks are the first shelf of rows on the page.
pub fn parse_album(browse_id: &str, root: &Value) -> AlbumPage {
    let mut page = AlbumPage {
        browse_id: browse_id.to_owned(),
        ..Default::default()
    };
    if let Some(header) = header(root) {
        page.title = text_field(header, "title").unwrap_or_default();
        page.thumbnail = thumbnail_of(header);
        // The artist is the link in the header's subtitle; the rest of the subtitle, and the
        // second subtitle (songs and length), make up the meta line.
        let artist = ["straplineTextOne", "subtitle"]
            .iter()
            .flat_map(|key| runs_of(header.get(*key)))
            .find_map(|run| match run_page(run) {
                Some((kind, id)) if is_artist_kind(kind) => Some((text_of_run(run), id)),
                _ => None,
            });
        if let Some((name, id)) = artist {
            page.artist = name;
            page.artist_id = id.map(str::to_owned);
        }
        let mut parts = Vec::new();
        meta_parts(header.get("subtitle"), &mut parts);
        meta_parts(header.get("secondSubtitle"), &mut parts);
        page.meta = parts.join(META_SEP);
    }

    let tracks = sections(root)
        .into_iter()
        .find_map(|section| section.get("musicShelfRenderer"));
    if let Some(shelf) = tracks {
        // Every row of an album's shelf is one of its tracks, so no row is filtered by kind.
        page.tracks = shelf_rows(shelf).filter_map(row_hit).collect();
        for track in &mut page.tracks {
            credit(track, &page.artist, page.artist_id.as_deref());
            track.album_id.get_or_insert_with(|| browse_id.to_owned());
            if track.album.is_none() && !page.title.is_empty() {
                track.album = Some(page.title.clone());
            }
            if track.thumbnail.is_none() {
                track.thumbnail = page.thumbnail.clone();
            }
        }
    }
    page
}

/// Between the parts of a meta line.
pub(crate) const META_SEP: &str = "  ·  ";

/// Fill in what a row may leave out: the artist's name and link, which pages usually supply.
fn credit(song: &mut SongHit, name: &str, artist_id: Option<&str>) {
    if song.artists.is_empty() && !name.is_empty() {
        song.artists.push(name.to_owned());
    }
    if song.artist_id.is_none() {
        song.artist_id = artist_id.map(str::to_owned);
    }
}

/// The page header. An artist's channel keeps it under `header`, as one renderer
/// (`musicImmersiveHeaderRenderer`). An album puts it first in its section list, as
/// `musicResponsiveHeaderRenderer`.
fn header(root: &Value) -> Option<&Value> {
    root.get("header")
        .and_then(|h| h.as_object()?.values().next())
        .or_else(|| {
            sections(root).into_iter().find_map(|section| {
                ["musicResponsiveHeaderRenderer", "musicDetailHeaderRenderer"]
                    .iter()
                    .find_map(|key| section.get(*key))
            })
        })
}

/// Every section of the page, in order, from every section list it has.
fn sections(root: &Value) -> Vec<&Value> {
    let mut lists = Vec::new();
    collect_key(root, "sectionListRenderer", &mut lists);
    lists
        .into_iter()
        .filter_map(|list| list.get("contents")?.as_array())
        .flatten()
        .collect()
}

/// The rows of a shelf: each `musicResponsiveListItemRenderer` in its contents.
fn shelf_rows(shelf: &Value) -> impl Iterator<Item = &Value> {
    shelf
        .get("contents")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| item.get("musicResponsiveListItemRenderer"))
}

/// The items of a carousel: each `musicTwoRowItemRenderer` in its contents.
fn carousel_items(carousel: &Value) -> impl Iterator<Item = &Value> {
    carousel
        .get("contents")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| item.get("musicTwoRowItemRenderer"))
}

/// An album or single from a carousel. Videos and playlists share carousels, so anything whose
/// browse id is not an album is skipped.
fn album_card(item: &Value) -> Option<AlbumCard> {
    let browse_id = item
        .get("navigationEndpoint")?
        .get("browseEndpoint")?
        .get("browseId")?
        .as_str()?;
    if !browse_id.starts_with("MPRE") {
        return None;
    }
    Some(AlbumCard {
        browse_id: browse_id.to_owned(),
        title: text_field(item, "title")?,
        meta: {
            let mut parts = Vec::new();
            meta_parts(item.get("subtitle"), &mut parts);
            parts.join(META_SEP)
        },
        thumbnail: item.get("thumbnailRenderer").and_then(url_in),
    })
}

/// The page a text run links to: its page type, and its browse id when it has one.
fn run_page(run: &Value) -> Option<(&str, Option<&str>)> {
    let endpoint = run.get("navigationEndpoint")?.get("browseEndpoint")?;
    let kind = endpoint
        .get("browseEndpointContextSupportedConfigs")?
        .get("browseEndpointContextMusicConfig")?
        .get("pageType")?
        .as_str()?;
    Some((kind, endpoint.get("browseId").and_then(Value::as_str)))
}

fn is_artist_kind(kind: &str) -> bool {
    matches!(kind, "MUSIC_PAGE_TYPE_ARTIST" | "MUSIC_PAGE_TYPE_USER_CHANNEL")
}

/// The text of one run, trimmed.
fn text_of_run(run: &Value) -> String {
    run.get("text").and_then(Value::as_str).unwrap_or_default().trim().to_owned()
}

/// The runs of a text object (`{ "runs": [...] }`), or none.
fn runs_of(value: Option<&Value>) -> impl Iterator<Item = &Value> {
    value
        .and_then(|v| v.get("runs"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}

/// The meta parts of a subtitle: its runs, without the separators and without the artist, who
/// is already named elsewhere.
pub(crate) fn meta_parts(value: Option<&Value>, out: &mut Vec<String>) {
    for run in runs_of(value) {
        let text = text_of_run(run);
        let is_artist = matches!(run_page(run), Some((kind, _)) if is_artist_kind(kind));
        if text.is_empty() || matches!(text.as_str(), "•" | "·") || is_artist {
            continue;
        }
        out.push(text);
    }
}

/// The text of the field `key` of `value`. See [`text_of`].
pub(crate) fn text_field(value: &Value, key: &str) -> Option<String> {
    text_of(value.get(key)?)
}

/// The text of a text object: its runs joined, or its `simpleText`.
pub(crate) fn text_of(field: &Value) -> Option<String> {
    if let Some(runs) = field.get("runs").and_then(Value::as_array) {
        let text: String = runs.iter().filter_map(|r| r.get("text")?.as_str()).collect();
        return (!text.is_empty()).then_some(text);
    }
    field.get("simpleText")?.as_str().map(str::to_owned)
}

/// The picture of a header or item. Prefers the foreground image, then the plain thumbnail, and
/// takes the largest size on offer.
fn thumbnail_of(value: &Value) -> Option<String> {
    ["foregroundThumbnail", "thumbnail"]
        .iter()
        .find_map(|key| value.get(*key).and_then(url_in))
        .or_else(|| url_in(value))
}

/// The largest URL under the first `thumbnails` list in `value`.
pub(crate) fn url_in(value: &Value) -> Option<String> {
    let mut found = Vec::new();
    collect_key(value, "thumbnails", &mut found);
    found
        .first()?
        .as_array()?
        .last()?
        .get("url")?
        .as_str()
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trimmed artist channel, shaped like the live response: a header, a songs shelf, and an
    /// albums carousel with a video item that must be skipped.
    fn artist_fixture() -> Value {
        json!({
          "header": { "musicImmersiveHeaderRenderer": {
            "title": { "runs": [ { "text": "Daft Punk" } ] },
            "foregroundThumbnail": { "musicThumbnailRenderer": { "thumbnail": { "thumbnails": [
              { "url": "https://x/small", "width": 60 }, { "url": "https://x/large", "width": 544 } ] } } }
          } },
          "contents": { "singleColumnBrowseResultsRenderer": { "tabs": [ { "tabRenderer": { "content": {
            "sectionListRenderer": { "contents": [
              { "musicShelfRenderer": {
                  "title": { "runs": [ { "text": "Songs" } ] },
                  "contents": [ { "musicResponsiveListItemRenderer": {
                      "playlistItemData": { "videoId": "abc123" },
                      "overlay": crate::endpoints::play_overlay("MUSIC_VIDEO_TYPE_ATV"),
                      "flexColumns": [
                        { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                          { "text": "Around the World" } ] } } },
                        { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                          { "text": "Homework", "navigationEndpoint": { "browseEndpoint": {
                            "browseId": "MPREb_homework",
                            "browseEndpointContextSupportedConfigs": { "browseEndpointContextMusicConfig": {
                              "pageType": "MUSIC_PAGE_TYPE_ALBUM" } } } } } ] } } } ] } } ] } },
              { "musicCarouselShelfRenderer": {
                  "header": { "musicCarouselShelfBasicHeaderRenderer": {
                      "title": { "runs": [ { "text": "Albums" } ] } } },
                  "contents": [
                    { "musicTwoRowItemRenderer": {
                        "title": { "runs": [ { "text": "Homework" } ] },
                        "subtitle": { "runs": [ { "text": "Album" }, { "text": " • " }, { "text": "1997" } ] },
                        "navigationEndpoint": { "browseEndpoint": { "browseId": "MPREb_homework" } },
                        "thumbnailRenderer": { "musicThumbnailRenderer": { "thumbnail": { "thumbnails": [
                          { "url": "https://x/album" } ] } } } } },
                    { "musicTwoRowItemRenderer": {
                        "title": { "runs": [ { "text": "A video" } ] },
                        "navigationEndpoint": { "watchEndpoint": { "videoId": "vid9" } } } }
                  ] } }
            ] }
          } } } ] } }
        })
    }

    #[test]
    fn an_artist_page_has_its_name_picture_songs_and_albums() {
        let page = parse_artist("UCdaftpunk", &artist_fixture());
        assert_eq!(page.name, "Daft Punk");
        assert_eq!(page.thumbnail.as_deref(), Some("https://x/large"));
        assert_eq!(page.songs_title, "Songs");
        assert_eq!(page.songs.len(), 1);
        // The row names no artist, so the page's artist is credited, and linked to the page.
        assert_eq!(page.songs[0].artists, vec!["Daft Punk".to_owned()]);
        assert_eq!(page.songs[0].artist_id.as_deref(), Some("UCdaftpunk"));
        assert_eq!(page.songs[0].album_id.as_deref(), Some("MPREb_homework"));

        assert_eq!(page.shelves.len(), 1, "the video item is not an album");
        let albums = &page.shelves[0].albums;
        assert_eq!(page.shelves[0].title, "Albums");
        assert_eq!(albums.len(), 1);
        assert_eq!(albums[0].browse_id, "MPREb_homework");
        assert_eq!(albums[0].meta, "Album  ·  1997");
        assert_eq!(albums[0].thumbnail.as_deref(), Some("https://x/album"));
    }

    #[test]
    fn an_album_page_has_its_header_and_tracks() {
        let root = json!({
          "header": { "musicDetailHeaderRenderer": {
            "title": { "runs": [ { "text": "Homework" } ] },
            "subtitle": { "runs": [
              { "text": "Album" }, { "text": " • " },
              { "text": "Daft Punk", "navigationEndpoint": { "browseEndpoint": {
                  "browseId": "UCdaftpunk",
                  "browseEndpointContextSupportedConfigs": { "browseEndpointContextMusicConfig": {
                      "pageType": "MUSIC_PAGE_TYPE_ARTIST" } } } } },
              { "text": " • " }, { "text": "1997" } ] },
            "secondSubtitle": { "runs": [ { "text": "3 songs" }, { "text": " • " }, { "text": "12 minutes" } ] },
            "thumbnail": { "croppedSquareThumbnailRenderer": { "thumbnail": { "thumbnails": [
              { "url": "https://x/cover" } ] } } }
          } },
          "contents": { "singleColumnBrowseResultsRenderer": { "tabs": [ { "tabRenderer": { "content": {
            "sectionListRenderer": { "contents": [
              { "musicShelfRenderer": { "contents": [
                  { "musicResponsiveListItemRenderer": {
                      "playlistItemData": { "videoId": "t1" },
                      "overlay": crate::endpoints::play_overlay("MUSIC_VIDEO_TYPE_ATV"),
                      "fixedColumns": [ { "musicResponsiveListItemFixedColumnRenderer": { "text": {
                        "runs": [ { "text": "3:45" } ] } } } ],
                      "flexColumns": [ { "musicResponsiveListItemFlexColumnRenderer": { "text": {
                        "runs": [ { "text": "Teachers" } ] } } } ] } },
                  { "musicResponsiveListItemRenderer": {
                      "playlistItemData": { "videoId": "t2" },
                      "overlay": crate::endpoints::play_overlay("MUSIC_VIDEO_TYPE_ATV"),
                      "flexColumns": [ { "musicResponsiveListItemFlexColumnRenderer": { "text": {
                        "runs": [ { "text": "Around the World" } ] } } } ] } }
              ] } }
            ] }
          } } } ] } }
        });
        check_album_header(parse_album("MPREb_homework", &root));
    }

    #[test]
    fn an_album_header_in_the_section_list_names_the_artist_from_its_strapline() {
        // The live shape: the header is the first item of the primary section list, and the
        // artist is the link in `straplineTextOne`, not in the subtitle.
        let root = json!({
          "contents": { "twoColumnBrowseResultsRenderer": { "tabs": [ { "tabRenderer": { "content": {
            "sectionListRenderer": { "contents": [
              { "musicResponsiveHeaderRenderer": {
                  "title": { "runs": [ { "text": "Homework" } ] },
                  "straplineTextOne": { "runs": [ { "text": "Daft Punk", "navigationEndpoint": {
                      "browseEndpoint": { "browseId": "UCdaftpunk", "browseEndpointContextSupportedConfigs": {
                        "browseEndpointContextMusicConfig": { "pageType": "MUSIC_PAGE_TYPE_ARTIST" } } } } } ] },
                  "subtitle": { "runs": [ { "text": "Album" }, { "text": " • " }, { "text": "1996" } ] },
                  "secondSubtitle": { "runs": [ { "text": "16 songs" }, { "text": " • " }, { "text": "1 hour, 14 minutes" } ] },
                  "thumbnail": { "musicThumbnailRenderer": { "thumbnail": { "thumbnails": [
                    { "url": "https://x/cover" } ] } } }
              } },
              { "musicShelfRenderer": { "contents": [
                  { "musicResponsiveListItemRenderer": {
                      "playlistItemData": { "videoId": "t1" },
                      "overlay": crate::endpoints::play_overlay("MUSIC_VIDEO_TYPE_ATV"),
                      "flexColumns": [ { "musicResponsiveListItemFlexColumnRenderer": { "text": {
                        "runs": [ { "text": "Teachers" } ] } } } ] } }
              ] } }
            ] }
          } } } ] } }
        });
        let page = parse_album("MPREb_homework", &root);
        assert_eq!(page.title, "Homework");
        assert_eq!(page.artist, "Daft Punk");
        assert_eq!(page.artist_id.as_deref(), Some("UCdaftpunk"));
        assert_eq!(page.meta, "Album  ·  1996  ·  16 songs  ·  1 hour, 14 minutes");
        assert_eq!(page.thumbnail.as_deref(), Some("https://x/cover"));
        assert_eq!(page.tracks.len(), 1);
        assert_eq!(page.tracks[0].artists, vec!["Daft Punk".to_owned()]);
    }

    #[test]
    fn music_videos_are_left_off_artist_pages_but_kept_as_album_tracks() {
        let mut root = artist_fixture();
        let shelf = &mut root["contents"]["singleColumnBrowseResultsRenderer"]["tabs"][0]
            ["tabRenderer"]["content"]["sectionListRenderer"]["contents"][0]["musicShelfRenderer"];
        let mut video = shelf["contents"][0].clone();
        video["musicResponsiveListItemRenderer"]["playlistItemData"]["videoId"] = json!("omv1");
        video["musicResponsiveListItemRenderer"]["overlay"] =
            crate::endpoints::play_overlay("MUSIC_VIDEO_TYPE_OMV");
        shelf["contents"].as_array_mut().unwrap().insert(0, video);

        let artist = parse_artist("UCdaftpunk", &root);
        let ids: Vec<&str> = artist.songs.iter().map(|s| s.video_id.as_str()).collect();
        assert_eq!(ids, vec!["abc123"]);

        // An album whose tracks link to official uploads (Daft Punk's "Discovery") is still its
        // tracks: dropping them left the page empty.
        let album = parse_album("MPREb_x", &root);
        let ids: Vec<&str> = album.tracks.iter().map(|s| s.video_id.as_str()).collect();
        assert_eq!(ids, vec!["omv1", "abc123"]);
    }

    /// The header and tracks of [`an_album_page_has_its_header_and_tracks`]'s fixture.
    fn check_album_header(page: AlbumPage) {
        assert_eq!(page.title, "Homework");
        assert_eq!(page.artist, "Daft Punk");
        assert_eq!(page.artist_id.as_deref(), Some("UCdaftpunk"));
        assert_eq!(page.meta, "Album  ·  1997  ·  3 songs  ·  12 minutes");
        assert_eq!(page.thumbnail.as_deref(), Some("https://x/cover"));
        assert_eq!(page.tracks.len(), 2);
        let first = &page.tracks[0];
        assert_eq!(first.duration.as_deref(), Some("3:45"));
        assert_eq!(first.artists, vec!["Daft Punk".to_owned()]);
        assert_eq!(first.album.as_deref(), Some("Homework"));
        assert_eq!(first.album_id.as_deref(), Some("MPREb_homework"));
        assert_eq!(first.thumbnail.as_deref(), Some("https://x/cover"));
    }
}
