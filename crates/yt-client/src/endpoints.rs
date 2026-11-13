// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! The InnerTube endpoints this client uses: `/player` and `/search`.

use std::collections::HashSet;

use serde_json::{json, Value};

use crate::clients::ClientKind;
use crate::error::Result;
use crate::models::{PlayerResponse, SongHit};
use crate::pages::{meta_parts, text_field, text_of, url_in, AlbumCard, ArtistCard, META_SEP};
use crate::transport::{Auth, InnerTube};

/// The "Songs" search filter (`params`), as the web client sends it.
pub const SONGS_FILTER: &str = "EgWKAQIIAWoKEAMQBBAJEAoQBQ==";

/// What a `/player` request is for.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlayerRequest<'a> {
    pub video_id: &'a str,
    /// `MLPT` for uploads. Without it YouTube gives uploads URLs that expire in about 30 seconds.
    pub playlist_id: Option<&'a str>,
    /// Signature timestamp from the player script. Only for clients that ask for one.
    pub signature_timestamp: Option<u32>,
    /// PoToken. Only for clients that ask for one.
    pub po_token: Option<&'a str>,
}

impl InnerTube {
    /// `/player` as one identity. Returns the raw response; playability is checked by the caller,
    /// because a non-OK status is an answer, not an error.
    pub async fn player(
        &self,
        kind: ClientKind,
        auth: Auth,
        req: PlayerRequest<'_>,
    ) -> Result<PlayerResponse> {
        let mut body = json!({
            "videoId": req.video_id,
            "contentCheckOk": true,
            "racyCheckOk": true,
        });
        if let Some(p) = req.playlist_id {
            body["playlistId"] = json!(p);
        }
        if let Some(sts) = req.signature_timestamp {
            body["playbackContext"] = json!({
                "contentPlaybackContext": {
                    "html5Preference": "HTML5_PREF_WANTS",
                    "signatureTimestamp": sts
                }
            });
        }
        if let Some(token) = req.po_token {
            body["serviceIntegrityDimensions"] = json!({ "poToken": token });
        }
        let value = self.post("player", kind, auth, body).await?;
        Ok(serde_json::from_value(value)?)
    }

    /// A track's length in whole seconds, from the details of its `/player` answer. No stream is
    /// read, so this is a single small request.
    pub async fn track_length(&self, video_id: &str) -> Result<Option<u64>> {
        let req = PlayerRequest {
            video_id,
            ..Default::default()
        };
        let resp = self
            .player(ClientKind::VisionOs, Auth::Anonymous, req)
            .await?;
        Ok(resp
            .video_details
            .and_then(|details| details.length_seconds))
    }

    /// Songs matching `query`. Always anonymous: a search carries no account, so typing does not
    /// write to the user's search history. Pass `record_history` to send the query signed in.
    pub async fn search_songs(&self, query: &str, record_history: bool) -> Result<Vec<SongHit>> {
        let auth = if record_history {
            Auth::Account
        } else {
            Auth::Anonymous
        };
        let value = self
            .post(
                "search",
                ClientKind::WebRemix,
                auth,
                json!({ "query": query, "params": SONGS_FILTER }),
            )
            .await?;
        Ok(parse_song_hits(&value))
    }

    /// What a search shows: the top result, the songs, and the artists and albums it matched.
    ///
    /// Two requests go out together. The songs filter gives the full song list. The unfiltered
    /// search is the only one that carries the top result, the artists and the albums. If the
    /// unfiltered one fails, the songs still show, with no top result of its own.
    pub async fn search(&self, query: &str, record_history: bool) -> Result<SearchResults> {
        let auth = if record_history {
            Auth::Account
        } else {
            Auth::Anonymous
        };
        let songs = self.post(
            "search",
            ClientKind::WebRemix,
            auth,
            json!({ "query": query, "params": SONGS_FILTER }),
        );
        let mixed = self.post(
            "search",
            ClientKind::WebRemix,
            auth,
            json!({ "query": query }),
        );
        let (songs, mixed) = tokio::join!(songs, mixed);
        let songs = parse_song_hits(&songs?);
        let mixed = mixed.map(|value| parse_mixed(&value)).unwrap_or_default();
        Ok(assemble(songs, mixed))
    }

    /// The album a name and artist point at, when a search finds it. Used to give a song's album
    /// (which some rows name but do not link) its browse id, so the album can be opened.
    pub async fn find_album(&self, title: &str, artist: &str) -> Result<Option<AlbumCard>> {
        let mut query = title.trim().to_owned();
        if !artist.trim().is_empty() {
            query.push(' ');
            query.push_str(artist.trim());
        }
        if query.is_empty() {
            return Ok(None);
        }
        let value = self
            .post(
                "search",
                ClientKind::WebRemix,
                Auth::Anonymous,
                json!({ "query": query }),
            )
            .await?;
        let mut mixed = parse_mixed(&value);
        // The best match can be the top card rather than one of the result rows.
        if let Some(TopResult::Album(card)) = mixed.top.take() {
            mixed.albums.insert(0, card);
        }
        let wanted = normalize_name(title);
        if let Some(found) = mixed
            .albums
            .into_iter()
            .find(|album| normalize_name(&album.title) == wanted)
        {
            return Ok(Some(found));
        }
        // Albums below the top result can be drawn as carousel cards rather than list rows.
        let mut cards = Vec::new();
        collect_key(&value, "musicTwoRowItemRenderer", &mut cards);
        for card in cards {
            let Some(browse_id) = card
                .get("navigationEndpoint")
                .and_then(|endpoint| endpoint.get("browseEndpoint"))
                .and_then(|endpoint| endpoint.get("browseId"))
                .and_then(Value::as_str)
                .filter(|id| id.starts_with("MPRE"))
            else {
                continue;
            };
            let Some(card_title) = text_field(card, "title") else {
                continue;
            };
            if normalize_name(&card_title) != wanted {
                continue;
            }
            let mut parts = Vec::new();
            meta_parts(card.get("subtitle"), &mut parts);
            return Ok(Some(AlbumCard {
                browse_id: browse_id.to_owned(),
                title: card_title,
                meta: parts.join(META_SEP),
                thumbnail: card.get("thumbnail").and_then(url_in),
            }));
        }
        Ok(None)
    }
}

/// A name reduced to its letters and digits, for comparing an album's title across pages.
fn normalize_name(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// What a search shows. See [`InnerTube::search`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchResults {
    /// The best match, shown on its own above the rest. `None` when the search had nothing.
    pub top: Option<TopResult>,
    /// Songs, without the top result when that is a song.
    pub songs: Vec<SongHit>,
    /// Artists the search matched, without the top result when that is an artist.
    pub artists: Vec<ArtistCard>,
    /// Albums the search matched, without the top result when that is an album.
    pub albums: Vec<AlbumCard>,
}

/// The top result: whichever of a song, an artist or an album the search thought best.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopResult {
    Song(SongHit),
    Artist(ArtistCard),
    Album(AlbumCard),
}

/// The parts of an unfiltered search that the songs filter does not give.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MixedResults {
    pub top: Option<TopResult>,
    pub artists: Vec<ArtistCard>,
    pub albums: Vec<AlbumCard>,
}

/// Join the songs search with the unfiltered one. With no top card, the first song is the top
/// result, so there is always one when there is any song.
fn assemble(mut songs: Vec<SongHit>, mixed: MixedResults) -> SearchResults {
    let top = match mixed.top {
        Some(top) => Some(top),
        None if !songs.is_empty() => Some(TopResult::Song(songs.remove(0))),
        None => None,
    };
    let (mut artists, mut albums) = (mixed.artists, mixed.albums);
    match &top {
        Some(TopResult::Song(hit)) => songs.retain(|song| song.video_id != hit.video_id),
        Some(TopResult::Artist(artist)) => artists.retain(|a| a.browse_id != artist.browse_id),
        Some(TopResult::Album(album)) => albums.retain(|a| a.browse_id != album.browse_id),
        None => {}
    }
    SearchResults {
        top,
        songs,
        artists,
        albums,
    }
}

/// Read an unfiltered search. The top result is the first card. Artists and albums are the rows
/// whose name links to an artist or album page rather than to a video; each is kept once.
pub fn parse_mixed(root: &Value) -> MixedResults {
    let mut cards = Vec::new();
    collect_key(root, "musicCardShelfRenderer", &mut cards);
    let mut mixed = MixedResults {
        top: cards.first().and_then(|card| top_result(card)),
        ..Default::default()
    };

    let mut rows = Vec::new();
    collect_key(root, "musicResponsiveListItemRenderer", &mut rows);
    let mut seen = HashSet::new();
    for row in rows {
        let Some(browse_id) = row_browse_id(row) else {
            continue;
        };
        if !seen.insert(browse_id.to_owned()) {
            continue;
        }
        if browse_id.starts_with("UC") {
            mixed.artists.extend(artist_row(row, browse_id));
        } else if browse_id.starts_with("MPRE") {
            mixed.albums.extend(album_row(row, browse_id));
        }
    }
    mixed
}

/// The browse id a row links to: on the row itself, or on its name.
fn row_browse_id(row: &Value) -> Option<&str> {
    row.get("navigationEndpoint")
        .and_then(|endpoint| endpoint.get("browseEndpoint"))
        .or_else(|| {
            flex_text(row, 0)?
                .get("runs")?
                .get(0)?
                .get("navigationEndpoint")?
                .get("browseEndpoint")
        })?
        .get("browseId")?
        .as_str()
}

/// The text object of a row's flex column `i`.
fn flex_text(row: &Value, i: usize) -> Option<&Value> {
    row.get("flexColumns")?
        .get(i)?
        .get("musicResponsiveListItemFlexColumnRenderer")?
        .get("text")
}

/// An artist row: its name, then the meta line without the word "Artist". A row that does not say
/// "Artist" is a person's channel ("Profile"), not an artist, and is skipped.
fn artist_row(row: &Value, browse_id: &str) -> Option<ArtistCard> {
    let name = text_of(flex_text(row, 0)?)?;
    let mut parts = Vec::new();
    meta_parts(flex_text(row, 1), &mut parts);
    if parts.first().is_none_or(|kind| kind != "Artist") {
        return None;
    }
    parts.remove(0);
    Some(ArtistCard {
        browse_id: browse_id.to_owned(),
        name,
        meta: parts.join(META_SEP),
        thumbnail: row.get("thumbnail").and_then(url_in),
    })
}

/// An album row: its title, then the meta line, which keeps the kind ("Album") and the year.
fn album_row(row: &Value, browse_id: &str) -> Option<AlbumCard> {
    let title = text_of(flex_text(row, 0)?)?;
    let mut parts = Vec::new();
    meta_parts(flex_text(row, 1), &mut parts);
    Some(AlbumCard {
        browse_id: browse_id.to_owned(),
        title,
        meta: parts.join(META_SEP),
        thumbnail: row.get("thumbnail").and_then(url_in),
    })
}

/// The top card. Its `onTap` says what it is: a browse endpoint is an artist or an album, a watch
/// endpoint is a song.
fn top_result(card: &Value) -> Option<TopResult> {
    let on_tap = card.get("onTap")?;
    let title = text_field(card, "title")?;
    let subtitle = card.get("subtitle");
    let thumbnail = card.get("thumbnail").and_then(url_in);

    if let Some(endpoint) = on_tap.get("browseEndpoint") {
        let browse_id = endpoint.get("browseId")?.as_str()?.to_owned();
        let mut parts = Vec::new();
        meta_parts(subtitle, &mut parts);
        if browse_id.starts_with("UC") {
            if parts.first().is_some_and(|kind| kind == "Artist") {
                parts.remove(0);
            }
            return Some(TopResult::Artist(ArtistCard {
                browse_id,
                name: title,
                meta: parts.join(META_SEP),
                thumbnail,
            }));
        }
        if browse_id.starts_with("MPRE") {
            return Some(TopResult::Album(AlbumCard {
                browse_id,
                title,
                meta: parts.join(META_SEP),
                thumbnail,
            }));
        }
        return None;
    }

    let endpoint = on_tap.get("watchEndpoint")?;
    if !is_song(endpoint) {
        return None;
    }
    let video_id = endpoint.get("videoId")?.as_str()?.to_owned();
    let runs = subtitle
        .and_then(|s| s.get("runs"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let credits = read_credits(runs);
    Some(TopResult::Song(SongHit {
        video_id,
        title,
        artists: credits.artists,
        album: credits.album,
        duration: credits.duration,
        thumbnail,
        artist_id: credits.artist_id,
        album_id: credits.album_id,
        artist_ids: credits.artist_ids,
    }))
}

/// What YouTube calls a song: the track itself (`ATV`, an "art track"), or a track the user
/// uploaded. Everything else it can play is a video: an official music video or official audio
/// upload (`OMV`), a user video (`UGC`), a podcast episode, and any kind added later.
pub fn is_song_type(kind: &str) -> bool {
    matches!(
        kind,
        "MUSIC_VIDEO_TYPE_ATV" | "MUSIC_VIDEO_TYPE_PRIVATELY_OWNED_TRACK"
    )
}

/// Whether a row or watch endpoint plays a song, by the `musicVideoType` its watch endpoints
/// carry. Strict: anything that does not say it is a song is treated as a video and kept out.
pub(crate) fn is_song(value: &Value) -> bool {
    let mut kinds = Vec::new();
    collect_key(value, "musicVideoType", &mut kinds);
    !kinds.is_empty()
        && kinds
            .iter()
            .all(|kind| kind.as_str().is_some_and(is_song_type))
}

/// Pull song rows out of a search response. Rows without a video id (albums, artists, playlists)
/// are skipped, and so is every video.
pub fn parse_song_hits(root: &Value) -> Vec<SongHit> {
    let mut rows = Vec::new();
    collect_key(root, "musicResponsiveListItemRenderer", &mut rows);
    rows.into_iter().filter_map(song_from_row).collect()
}

pub(crate) fn collect_key<'a>(v: &'a Value, key: &str, out: &mut Vec<&'a Value>) {
    match v {
        Value::Object(map) => {
            for (k, child) in map {
                if k == key {
                    out.push(child);
                } else {
                    collect_key(child, key, out);
                }
            }
        }
        Value::Array(items) => items.iter().for_each(|c| collect_key(c, key, out)),
        _ => {}
    }
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

/// A row's length, when it is in the fixed columns (album tracks put it there, not beside the title).
fn fixed_duration(row: &Value) -> Option<String> {
    row.get("fixedColumns")?
        .as_array()?
        .iter()
        .find_map(|column| {
            let text = column
                .get("musicResponsiveListItemFixedColumnRenderer")?
                .get("text")?
                .get("runs")?
                .as_array()?
                .first()?
                .get("text")?
                .as_str()?;
            is_duration(text).then(|| text.to_owned())
        })
}

/// One song row. `None` for a row that plays nothing, and for a video: videos never leave this
/// crate through search or an artist's songs, so those lists only ever hold songs.
pub(crate) fn song_from_row(row: &Value) -> Option<SongHit> {
    if !is_song(row) {
        return None;
    }
    row_hit(row)
}

/// Any row that plays something, whatever kind YouTube says it is. An album's track list uses
/// this: YouTube links some albums' tracks to the official video or audio upload (`OMV`) rather
/// than an art track, and those rows are still the album's tracks.
pub(crate) fn row_hit(row: &Value) -> Option<SongHit> {
    let flex = |i: usize| -> Option<&Value> {
        row.get("flexColumns")?
            .get(i)?
            .get("musicResponsiveListItemFlexColumnRenderer")?
            .get("text")?
            .get("runs")
    };

    let video_id = row
        .get("playlistItemData")
        .and_then(|p| p.get("videoId"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            flex(0)?
                .get(0)?
                .get("navigationEndpoint")?
                .get("watchEndpoint")?
                .get("videoId")?
                .as_str()
                .map(str::to_owned)
        })?;

    let title = flex(0)?.get(0)?.get("text")?.as_str()?.to_owned();

    // Credits are read from every column after the title. A search row keeps them all in the
    // second column; an artist's top songs spread them out (artists, plays, then the album).
    // Each column is read on its own: in the first, a run with no link is usually the artist (or
    // the album, after a separator); in a later one it is the album.
    let credit_columns: Vec<Vec<Value>> = (1..)
        .map_while(|i| row.get("flexColumns")?.get(i))
        .filter_map(|column| {
            Some(
                column
                    .get("musicResponsiveListItemFlexColumnRenderer")?
                    .get("text")?
                    .get("runs")?
                    .as_array()?
                    .clone(),
            )
        })
        .collect();
    let credits = read_credit_columns(&credit_columns);
    let duration = credits.duration.or_else(|| fixed_duration(row));

    let thumbnail = row
        .get("thumbnail")
        .and_then(|t| t.get("musicThumbnailRenderer"))
        .and_then(|t| t.get("thumbnail"))
        .and_then(|t| t.get("thumbnails"))
        .and_then(Value::as_array)
        .and_then(|a| a.last())
        .and_then(|t| t.get("url"))
        .and_then(Value::as_str)
        .map(str::to_owned);

    Some(SongHit {
        video_id,
        title,
        artists: credits.artists,
        album: credits.album,
        duration,
        thumbnail,
        artist_id: credits.artist_id,
        album_id: credits.album_id,
        artist_ids: credits.artist_ids,
    })
}

/// What a song's second line says about it: its artists, its album, and its length.
#[derive(Default)]
struct Credits {
    artists: Vec<String>,
    artist_id: Option<String>,
    artist_ids: Vec<Option<String>>,
    album: Option<String>,
    album_id: Option<String>,
    duration: Option<String>,
}

impl Credits {
    fn push_artist(&mut self, name: &str, id: Option<&str>) {
        let name = name.trim();
        if name.is_empty() {
            return;
        }
        self.artists.push(name.to_owned());
        self.artist_ids.push(id.map(str::to_owned));
        if self.artist_id.is_none() {
            self.artist_id = id.map(str::to_owned);
        }
    }

    fn set_album(&mut self, name: &str, id: Option<&str>) {
        if self.album.is_some() {
            return;
        }
        let name = name.trim();
        if !name.is_empty() {
            self.album = Some(name.to_owned());
            self.album_id = id.map(str::to_owned);
        }
    }

    /// Add another line's credits to this one, keeping what this one already knows.
    fn add(&mut self, other: Credits) {
        for (name, id) in other.artists.into_iter().zip(other.artist_ids) {
            if self.artist_id.is_none() {
                self.artist_id = id.clone();
            }
            self.artists.push(name);
            self.artist_ids.push(id);
        }
        if self.album.is_none() {
            self.album = other.album;
            self.album_id = other.album_id;
        }
        if self.duration.is_none() {
            self.duration = other.duration;
        }
    }
}

/// Read a song's credits from its flex columns, each on its own. The first is the line beside the
/// title; later ones are extra columns (a play count, then the album on an artist's top songs).
fn read_credit_columns(columns: &[Vec<Value>]) -> Credits {
    let mut credits = Credits::default();
    for (index, runs) in columns.iter().enumerate() {
        credits.add(read_credits_line(runs, index == 0));
    }
    credits
}

/// Read a song's credits from the runs of one line. Each artist, album or length is recognised by
/// the page its run links to, or by its shape. YouTube does not always link a credit (a karaoke
/// upload's artist, say), so an unlinked run is placed by its position on the line: the first part
/// is the artist, a later part is the album, and a later column is the album.
fn read_credits(runs: &[Value]) -> Credits {
    read_credits_line(runs, true)
}

fn read_credits_line(runs: &[Value], first_line: bool) -> Credits {
    let mut credits = Credits::default();
    let mut segment: Vec<&Value> = Vec::new();
    let mut segments: Vec<Vec<&Value>> = Vec::new();
    for run in runs {
        let text = run.get("text").and_then(Value::as_str).unwrap_or_default();
        if is_credit_separator(text) {
            segments.push(std::mem::take(&mut segment));
        } else {
            segment.push(run);
        }
    }
    if !segment.is_empty() {
        segments.push(segment);
    }

    for (index, segment) in segments.iter().enumerate() {
        let mut linked_artist = false;
        let mut linked_album = false;
        let mut plain = Vec::new();
        for run in segment {
            let text = run.get("text").and_then(Value::as_str).unwrap_or_default();
            match run_page(run) {
                Some(("MUSIC_PAGE_TYPE_ARTIST" | "MUSIC_PAGE_TYPE_USER_CHANNEL", id)) => {
                    credits.push_artist(text, id);
                    linked_artist = true;
                }
                Some(("MUSIC_PAGE_TYPE_ALBUM", id)) => {
                    credits.set_album(text, id);
                    linked_album = true;
                }
                _ if is_duration(text) => credits.duration = Some(text.to_owned()),
                _ if is_unlinked_credit(text) => plain.push(text),
                _ => {}
            }
        }
        // A part with no link at all still names something. Join its runs first: a credit split
        // over punctuation runs ("Smith, Jones") is one name, not several.
        let plain = plain.join("").trim().to_owned();
        if plain.is_empty() || is_duration(&plain) || is_stat(&plain) || is_kind(&plain) {
            continue;
        }
        let album_part = index > 0;
        if linked_album {
            // A link already named the album; an unlinked run here is another artist credit.
            credits.push_artist(&plain, None);
        } else if linked_artist {
            // Inside the artist part, so the unlinked run is another credit, not the album.
            if !album_part || first_line {
                credits.push_artist(&plain, None);
            }
        } else if album_part || !first_line {
            credits.set_album(&plain, None);
        } else {
            credits.push_artist(&plain, None);
        }
    }
    credits
}

/// A run that separates the parts of a credit line ("•", "·", "|").
fn is_credit_separator(text: &str) -> bool {
    matches!(text.trim(), "•" | "·" | "|")
}

/// A run that could be a credit, as opposed to punctuation or noise.
fn is_unlinked_credit(text: &str) -> bool {
    let text = text.trim();
    !text.is_empty()
        && text.chars().any(char::is_alphanumeric)
        && !text.chars().all(|c| c.is_ascii_digit() || c == '-')
        && !is_stat(text)
        && !is_kind(text)
}

/// A play count, an audience figure, or a view count: not a credit.
fn is_stat(text: &str) -> bool {
    let text = text.trim().to_ascii_lowercase();
    [" plays", " views", " audience", " subscribers"]
        .iter()
        .any(|suffix| text.ends_with(suffix))
}

/// A row's kind word ("Song", "Album", "Video", …): not a credit.
fn is_kind(text: &str) -> bool {
    matches!(
        text.trim(),
        "Song"
            | "Video"
            | "Album"
            | "Single"
            | "EP"
            | "Playlist"
            | "Profile"
            | "Artist"
            | "Podcast"
            | "Episode"
    )
}

/// `m:ss` or `h:mm:ss`.
fn is_duration(s: &str) -> bool {
    let parts: Vec<&str> = s.split(':').collect();
    (parts.len() == 2 || parts.len() == 3)
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
}

/// A row's play button, as the live rows carry it: the watch endpoint and the kind it plays.
#[cfg(test)]
pub(crate) fn play_overlay(kind: &str) -> Value {
    json!({ "musicItemThumbnailOverlayRenderer": { "content": { "musicPlayButtonRenderer": {
        "playNavigationEndpoint": { "watchEndpoint": {
            "watchEndpointMusicSupportedConfigs": { "watchEndpointMusicConfig": {
                "musicVideoType": kind } } } } } } } })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ATV: &str = "MUSIC_VIDEO_TYPE_ATV";

    #[test]
    fn duration_shapes() {
        assert!(is_duration("5:54"));
        assert!(is_duration("1:02:03"));
        assert!(!is_duration("3.3M plays"));
        assert!(!is_duration("Daft Punk"));
    }

    #[test]
    fn a_song_row_yields_id_title_artists_album_and_duration() {
        let root = json!({
          "contents": [{
            "musicResponsiveListItemRenderer": {
              "playlistItemData": { "videoId": "abc123" },
              "overlay": play_overlay(ATV),
              "flexColumns": [
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "Teachers (Extended Mix)" } ] } } },
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "Daft Punk", "navigationEndpoint": { "browseEndpoint": {
                        "browseId": "UCdaftpunk",
                        "browseEndpointContextSupportedConfigs": { "browseEndpointContextMusicConfig": {
                            "pageType": "MUSIC_PAGE_TYPE_ARTIST" } } } } },
                    { "text": " • " },
                    { "text": "Homework", "navigationEndpoint": { "browseEndpoint": {
                        "browseId": "MPREb_homework",
                        "browseEndpointContextSupportedConfigs": { "browseEndpointContextMusicConfig": {
                            "pageType": "MUSIC_PAGE_TYPE_ALBUM" } } } } },
                    { "text": " • " },
                    { "text": "5:54" } ] } } }
              ],
              "thumbnail": { "musicThumbnailRenderer": { "thumbnail": { "thumbnails": [
                  { "url": "https://x/small" }, { "url": "https://x/large" } ] } } }
            }
          }]
        });
        let hits = parse_song_hits(&root);
        assert_eq!(hits.len(), 1);
        let h = &hits[0];
        assert_eq!(h.video_id, "abc123");
        assert_eq!(h.title, "Teachers (Extended Mix)");
        assert_eq!(h.artists, vec!["Daft Punk".to_owned()]);
        assert_eq!(h.album.as_deref(), Some("Homework"));
        assert_eq!(h.duration.as_deref(), Some("5:54"));
        assert_eq!(h.thumbnail.as_deref(), Some("https://x/large"));
        assert_eq!(h.artist_id.as_deref(), Some("UCdaftpunk"));
        assert_eq!(h.album_id.as_deref(), Some("MPREb_homework"));
    }

    #[test]
    fn every_artist_credit_keeps_its_own_link() {
        let artist = |name: &str, id: &str| {
            json!({ "text": name, "navigationEndpoint": { "browseEndpoint": {
                "browseId": id,
                "browseEndpointContextSupportedConfigs": { "browseEndpointContextMusicConfig": {
                    "pageType": "MUSIC_PAGE_TYPE_ARTIST" } } } } })
        };
        let root = json!({
          "contents": [{
            "musicResponsiveListItemRenderer": {
              "playlistItemData": { "videoId": "duo" },
              "overlay": play_overlay(ATV),
              "flexColumns": [
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "Instant Crush" } ] } } },
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    artist("Daft Punk", "UCdaftpunk"),
                    { "text": " & " },
                    artist("Julian Casablancas", "UCjulian") ] } } }
              ]
            }
          }]
        });
        let hit = &parse_song_hits(&root)[0];
        let links: Vec<_> = hit.artist_links().collect();
        assert_eq!(
            links,
            vec![
                ("Daft Punk", Some("UCdaftpunk")),
                ("Julian Casablancas", Some("UCjulian"))
            ]
        );
        assert_eq!(hit.artist_id.as_deref(), Some("UCdaftpunk"));
    }

    #[test]
    fn an_unlinked_artist_is_still_read_from_its_position() {
        // A live karaoke row: the artist credit is plain text, while the album still links.
        let root = json!({
          "contents": [{
            "musicResponsiveListItemRenderer": {
              "playlistItemData": { "videoId": "karaoke" },
              "overlay": play_overlay(ATV),
              "flexColumns": [
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "Sanctuary (By Joji) (Melody Karaoke Version)" } ] } } },
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "ZZang KARAOKE" },
                    { "text": " • " },
                    { "text": "ZZang KARAOKE 2024 POP Vol.179", "navigationEndpoint": { "browseEndpoint": {
                        "browseId": "MPREb_sVmnqMDhb7A",
                        "browseEndpointContextSupportedConfigs": { "browseEndpointContextMusicConfig": {
                            "pageType": "MUSIC_PAGE_TYPE_ALBUM" } } } } },
                    { "text": " • " },
                    { "text": "2:58" } ] } } },
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "1.7K plays" } ] } } }
              ]
            }
          }]
        });
        let hit = &parse_song_hits(&root)[0];
        assert_eq!(hit.artists, vec!["ZZang KARAOKE".to_owned()]);
        assert_eq!(hit.album.as_deref(), Some("ZZang KARAOKE 2024 POP Vol.179"));
        assert_eq!(hit.album_id.as_deref(), Some("MPREb_sVmnqMDhb7A"));
        assert_eq!(hit.duration.as_deref(), Some("2:58"));
        assert_eq!(hit.artists.len(), 1, "the play count is not a credit");
    }

    #[test]
    fn an_unlinked_album_is_still_read_after_the_artist() {
        let root = json!({
          "contents": [{
            "musicResponsiveListItemRenderer": {
              "playlistItemData": { "videoId": "abc123" },
              "overlay": play_overlay(ATV),
              "flexColumns": [
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "Teachers" } ] } } },
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "Daft Punk", "navigationEndpoint": { "browseEndpoint": {
                        "browseId": "UCdaftpunk",
                        "browseEndpointContextSupportedConfigs": { "browseEndpointContextMusicConfig": {
                            "pageType": "MUSIC_PAGE_TYPE_ARTIST" } } } } },
                    { "text": " • " },
                    { "text": "Homework" },
                    { "text": " • " },
                    { "text": "5:54" } ] } } }
              ]
            }
          }]
        });
        let hit = &parse_song_hits(&root)[0];
        assert_eq!(hit.artists, vec!["Daft Punk".to_owned()]);
        assert_eq!(hit.album.as_deref(), Some("Homework"));
        assert_eq!(hit.album_id, None, "the album had no link to keep");
        assert_eq!(hit.duration.as_deref(), Some("5:54"));
    }

    #[test]
    fn an_artist_top_song_takes_an_unlinked_album_from_a_later_column() {
        // The live shape: the artist links in the first column, the play count sits in the second,
        // and the album is plain text in the third.
        let root = json!({
          "contents": [{
            "musicResponsiveListItemRenderer": {
              "playlistItemData": { "videoId": "abc123" },
              "overlay": play_overlay(ATV),
              "flexColumns": [
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "Instant Crush" } ] } } },
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "Daft Punk", "navigationEndpoint": { "browseEndpoint": {
                        "browseId": "UCdaftpunk",
                        "browseEndpointContextSupportedConfigs": { "browseEndpointContextMusicConfig": {
                            "pageType": "MUSIC_PAGE_TYPE_ARTIST" } } } } } ] } } },
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "1.2B plays" } ] } } },
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "Random Access Memories" } ] } } }
              ]
            }
          }]
        });
        let hit = &parse_song_hits(&root)[0];
        assert_eq!(hit.artists, vec!["Daft Punk".to_owned()]);
        assert_eq!(hit.album.as_deref(), Some("Random Access Memories"));
        assert_eq!(hit.album_id, None);
        assert_eq!(hit.artists.len(), 1, "the play count is not an artist");
    }

    #[test]
    fn an_album_track_takes_its_length_from_the_fixed_column() {
        let row = json!({
            "playlistItemData": { "videoId": "trk1" },
            "overlay": play_overlay(ATV),
            "fixedColumns": [ { "musicResponsiveListItemFixedColumnRenderer": { "text": { "runs": [
                { "text": "3:45" } ] } } } ],
            "flexColumns": [ { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                { "text": "Intro" } ] } } } ]
        });
        let hit = song_from_row(&row).expect("a row with a video id");
        assert_eq!(hit.duration.as_deref(), Some("3:45"));
        assert!(hit.artists.is_empty());
        assert_eq!(hit.artist_id, None);
    }

    #[test]
    fn an_artist_top_song_takes_its_album_from_a_later_column() {
        let row = json!({
            "playlistItemData": { "videoId": "khnokW3Mw24" },
            "overlay": play_overlay(ATV),
            "flexColumns": [
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "Instant Crush" } ] } } },
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "Daft Punk", "navigationEndpoint": { "browseEndpoint": {
                        "browseId": "UC_daft",
                        "browseEndpointContextSupportedConfigs": { "browseEndpointContextMusicConfig": {
                            "pageType": "MUSIC_PAGE_TYPE_ARTIST" } } } } } ] } } },
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "1.2B plays" } ] } } },
                { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                    { "text": "Random Access Memories", "navigationEndpoint": { "browseEndpoint": {
                        "browseId": "MPREb_ram",
                        "browseEndpointContextSupportedConfigs": { "browseEndpointContextMusicConfig": {
                            "pageType": "MUSIC_PAGE_TYPE_ALBUM" } } } } } ] } } }
            ]
        });
        let hit = song_from_row(&row).expect("a song");
        assert_eq!(hit.artists, vec!["Daft Punk".to_owned()]);
        assert_eq!(hit.album.as_deref(), Some("Random Access Memories"));
        assert_eq!(hit.album_id.as_deref(), Some("MPREb_ram"));
        assert_eq!(hit.duration, None);
    }

    #[test]
    fn rows_without_a_video_id_are_skipped() {
        let root = json!({ "x": { "musicResponsiveListItemRenderer": {
            "flexColumns": [ { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                { "text": "An album" } ] } } } ] } } });
        assert!(parse_song_hits(&root).is_empty());
    }

    /// The `browseEndpointContextSupportedConfigs` of a browse endpoint, for a page type.
    fn config(kind: &str) -> Value {
        json!({ "browseEndpointContextMusicConfig": { "pageType": kind } })
    }

    /// An unfiltered search response, trimmed to the shapes that matter: a top card for an artist,
    /// then one row each for an album, an artist and a song.
    fn mixed_fixture() -> Value {
        let thumb = json!({ "musicThumbnailRenderer": { "thumbnail": { "thumbnails": [
            { "url": "https://x/small" }, { "url": "https://x/large" } ] } } });
        json!({ "contents": { "tabbedSearchResultsRenderer": { "tabs": [ { "tabRenderer": { "content": {
          "sectionListRenderer": { "contents": [
            { "musicCardShelfRenderer": {
                "title": { "runs": [ { "text": "Daft Punk",
                    "navigationEndpoint": { "browseEndpoint": { "browseId": "UCdaftpunk",
                        "browseEndpointContextSupportedConfigs": { "browseEndpointContextMusicConfig": {
                            "pageType": "MUSIC_PAGE_TYPE_ARTIST" } } } } } ] },
                "subtitle": { "runs": [ { "text": "Artist" }, { "text": " • " },
                    { "text": "68.2M monthly audience" } ] },
                "thumbnail": thumb.clone(),
                "onTap": { "browseEndpoint": { "browseId": "UCdaftpunk",
                    "browseEndpointContextSupportedConfigs": { "browseEndpointContextMusicConfig": {
                        "pageType": "MUSIC_PAGE_TYPE_ARTIST" } } } } } },
            { "itemSectionRenderer": { "contents": [
                { "musicResponsiveListItemRenderer": {
                    "navigationEndpoint": { "browseEndpoint": { "browseId": "MPREb_homework" } },
                    "flexColumns": [
                      { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                          { "text": "Homework", "navigationEndpoint": { "browseEndpoint": {
                              "browseId": "MPREb_homework" } } } ] } } },
                      { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                          { "text": "Album" }, { "text": " • " },
                          { "text": "Daft Punk", "navigationEndpoint": { "browseEndpoint": {
                              "browseId": "UCdaftpunk", "browseEndpointContextSupportedConfigs": config("MUSIC_PAGE_TYPE_ARTIST") } } },
                          { "text": " • " }, { "text": "1997" } ] } } } ],
                    "thumbnail": thumb.clone() } },
                { "musicResponsiveListItemRenderer": {
                    "flexColumns": [
                      { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                          { "text": "The Weeknd", "navigationEndpoint": { "browseEndpoint": {
                              "browseId": "UClYV6hHlupm" } } } ] } } },
                      { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                          { "text": "Artist" }, { "text": " • " },
                          { "text": "233M monthly audience" } ] } } } ],
                    "thumbnail": thumb.clone() } },
                { "musicResponsiveListItemRenderer": {
                    "playlistItemData": { "videoId": "abc123" },
                    "overlay": play_overlay(ATV),
                    "flexColumns": [
                      { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                          { "text": "Alive" } ] } } },
                      { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                          { "text": "Song" }, { "text": " • " },
                          { "text": "Daft Punk", "navigationEndpoint": { "browseEndpoint": {
                              "browseId": "UCdaftpunk", "browseEndpointContextSupportedConfigs": config("MUSIC_PAGE_TYPE_ARTIST") } } } ] } } } ] } }
            ] } }
          ] } } } } ] } } })
    }

    #[test]
    fn the_top_card_is_read_by_what_it_taps_to() {
        let mixed = parse_mixed(&mixed_fixture());
        assert_eq!(
            mixed.top,
            Some(TopResult::Artist(ArtistCard {
                browse_id: "UCdaftpunk".to_owned(),
                name: "Daft Punk".to_owned(),
                meta: "68.2M monthly audience".to_owned(),
                thumbnail: Some("https://x/large".to_owned()),
            }))
        );
    }

    #[test]
    fn artist_and_album_rows_are_read_and_song_rows_are_left_to_the_songs_search() {
        let mixed = parse_mixed(&mixed_fixture());
        assert_eq!(mixed.albums.len(), 1);
        assert_eq!(mixed.albums[0].browse_id, "MPREb_homework");
        assert_eq!(mixed.albums[0].title, "Homework");
        assert_eq!(mixed.albums[0].meta, "Album  ·  1997");
        assert_eq!(mixed.artists.len(), 1, "the song row is not an artist");
        assert_eq!(mixed.artists[0].browse_id, "UClYV6hHlupm");
        assert_eq!(mixed.artists[0].name, "The Weeknd");
        assert_eq!(mixed.artists[0].meta, "233M monthly audience");
    }

    #[test]
    fn a_person_s_channel_is_not_an_artist() {
        let row = json!({ "flexColumns": [
            { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                { "text": "daft punk fan", "navigationEndpoint": { "browseEndpoint": {
                    "browseId": "UC-AS46mZpdgdc0WZH7b53kg" } } } ] } } },
            { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                { "text": "Profile" }, { "text": " • " }, { "text": "@someone" } ] } } } ] });
        assert_eq!(artist_row(&row, "UC-AS46mZpdgdc0WZH7b53kg"), None);
    }

    #[test]
    fn an_album_top_card_is_read_as_an_album() {
        let card = json!({
            "title": { "runs": [ { "text": "Discovery" } ] },
            "subtitle": { "runs": [ { "text": "Album" }, { "text": " • " }, { "text": "2001" } ] },
            "thumbnail": { "musicThumbnailRenderer": { "thumbnail": { "thumbnails": [ { "url": "https://x/a" } ] } } },
            "onTap": { "browseEndpoint": { "browseId": "MPREb_7ltM34kr0mH" } }
        });
        assert_eq!(
            top_result(&card),
            Some(TopResult::Album(AlbumCard {
                browse_id: "MPREb_7ltM34kr0mH".to_owned(),
                title: "Discovery".to_owned(),
                meta: "Album  ·  2001".to_owned(),
                thumbnail: Some("https://x/a".to_owned()),
            }))
        );
    }

    #[test]
    fn a_song_top_card_keeps_its_artist_album_and_length() {
        let card = json!({
            "title": { "runs": [ { "text": "One More Time" } ] },
            "subtitle": { "runs": [
                { "text": "Song" }, { "text": " • " },
                { "text": "Daft Punk", "navigationEndpoint": { "browseEndpoint": {
                    "browseId": "UCdaftpunk", "browseEndpointContextSupportedConfigs": config("MUSIC_PAGE_TYPE_ARTIST") } } },
                { "text": " • " },
                { "text": "Discovery", "navigationEndpoint": { "browseEndpoint": {
                    "browseId": "MPREb_7ltM34kr0mH", "browseEndpointContextSupportedConfigs": config("MUSIC_PAGE_TYPE_ALBUM") } } },
                { "text": " • " }, { "text": "5:20" } ] },
            "onTap": { "watchEndpoint": { "videoId": "FGBhQbmPwH8",
                "watchEndpointMusicSupportedConfigs": { "watchEndpointMusicConfig": {
                    "musicVideoType": ATV } } } }
        });
        match top_result(&card) {
            Some(TopResult::Song(hit)) => {
                assert_eq!(hit.video_id, "FGBhQbmPwH8");
                assert_eq!(hit.title, "One More Time");
                assert_eq!(hit.artists, vec!["Daft Punk".to_owned()]);
                assert_eq!(hit.artist_id.as_deref(), Some("UCdaftpunk"));
                assert_eq!(hit.album.as_deref(), Some("Discovery"));
                assert_eq!(hit.album_id.as_deref(), Some("MPREb_7ltM34kr0mH"));
                assert_eq!(hit.duration.as_deref(), Some("5:20"));
            }
            other => panic!("expected a song, got {other:?}"),
        }
    }

    #[test]
    fn the_top_result_is_not_repeated_below_itself() {
        let artist = ArtistCard {
            browse_id: "UCdaftpunk".to_owned(),
            name: "Daft Punk".to_owned(),
            ..Default::default()
        };
        let mixed = MixedResults {
            top: Some(TopResult::Artist(artist.clone())),
            artists: vec![
                artist,
                ArtistCard {
                    browse_id: "UClYV6hHlupm".to_owned(),
                    ..Default::default()
                },
            ],
            albums: Vec::new(),
        };
        let songs = vec![hit("abc123"), hit("def456")];
        let results = assemble(songs, mixed);
        assert_eq!(results.artists.len(), 1);
        assert_eq!(results.artists[0].browse_id, "UClYV6hHlupm");
        assert_eq!(
            results.songs.len(),
            2,
            "an artist on top takes no song off the list"
        );
    }

    #[test]
    fn a_song_on_top_is_taken_off_the_song_list() {
        let mixed = MixedResults {
            top: Some(TopResult::Song(hit("abc123"))),
            ..Default::default()
        };
        let results = assemble(vec![hit("abc123"), hit("def456")], mixed);
        let ids: Vec<&str> = results.songs.iter().map(|s| s.video_id.as_str()).collect();
        assert_eq!(ids, vec!["def456"]);
    }

    #[test]
    fn with_no_top_card_the_first_song_is_the_top_result() {
        let results = assemble(vec![hit("abc123"), hit("def456")], MixedResults::default());
        assert_eq!(results.top, Some(TopResult::Song(hit("abc123"))));
        assert_eq!(results.songs, vec![hit("def456")]);
    }

    /// A search row for `kind`, otherwise a complete song row.
    fn row_of_kind(video_id: &str, kind: Option<&str>) -> Value {
        let mut row = json!({
            "playlistItemData": { "videoId": video_id },
            "flexColumns": [ { "musicResponsiveListItemFlexColumnRenderer": { "text": { "runs": [
                { "text": "Get Lucky" } ] } } } ]
        });
        if let Some(kind) = kind {
            row["overlay"] = play_overlay(kind);
        }
        json!({ "musicResponsiveListItemRenderer": row })
    }

    #[test]
    fn only_songs_survive_a_search() {
        let root = json!({ "contents": [
            row_of_kind("song", Some(ATV)),
            row_of_kind("upload", Some("MUSIC_VIDEO_TYPE_PRIVATELY_OWNED_TRACK")),
            row_of_kind("omv", Some("MUSIC_VIDEO_TYPE_OMV")),
            row_of_kind("ugc", Some("MUSIC_VIDEO_TYPE_UGC")),
            row_of_kind("osm", Some("MUSIC_VIDEO_TYPE_OFFICIAL_SOURCE_MUSIC")),
            row_of_kind("podcast", Some("MUSIC_VIDEO_TYPE_PODCAST_EPISODE")),
            row_of_kind("someday", Some("MUSIC_VIDEO_TYPE_SOMETHING_NEW")),
            row_of_kind("untyped", None),
        ] });
        let ids: Vec<String> = parse_song_hits(&root)
            .into_iter()
            .map(|h| h.video_id)
            .collect();
        assert_eq!(ids, vec!["song".to_owned(), "upload".to_owned()]);
    }

    #[test]
    fn a_row_that_says_both_song_and_video_is_a_video() {
        let mut row = row_of_kind("mixed", Some(ATV));
        row["musicResponsiveListItemRenderer"]["menu"] = play_overlay("MUSIC_VIDEO_TYPE_OMV");
        assert!(parse_song_hits(&json!([row])).is_empty());
    }

    #[test]
    fn a_video_top_card_gives_way_to_the_first_song() {
        // The live answer for "daft punk get lucky": the top card is the official video.
        let card = json!({
            "title": { "runs": [ { "text": "Get Lucky (Official Audio)" } ] },
            "onTap": { "watchEndpoint": { "videoId": "5NV6Rdv1a3I",
                "watchEndpointMusicSupportedConfigs": { "watchEndpointMusicConfig": {
                    "musicVideoType": "MUSIC_VIDEO_TYPE_OMV" } } } }
        });
        assert_eq!(top_result(&card), None);
        let untyped = json!({
            "title": { "runs": [ { "text": "Get Lucky" } ] },
            "onTap": { "watchEndpoint": { "videoId": "x" } }
        });
        assert_eq!(
            top_result(&untyped),
            None,
            "a card that does not say song is not one"
        );

        let mixed = parse_mixed(&json!({ "musicCardShelfRenderer": card }));
        assert_eq!(mixed.top, None);
        let results = assemble(vec![hit("song1"), hit("song2")], mixed);
        assert_eq!(results.top, Some(TopResult::Song(hit("song1"))));
    }

    #[test]
    fn the_player_answer_marks_a_video() {
        use crate::models::VideoDetails;
        let details = |kind: Option<&str>| VideoDetails {
            music_video_type: kind.map(str::to_owned),
            ..Default::default()
        };
        assert!(details(Some("MUSIC_VIDEO_TYPE_OMV")).is_video());
        assert!(details(Some("MUSIC_VIDEO_TYPE_UGC")).is_video());
        assert!(!details(Some(ATV)).is_video());
        assert!(!details(None).is_video(), "silence is left to the parsers");
    }

    fn hit(video_id: &str) -> SongHit {
        SongHit {
            video_id: video_id.to_owned(),
            title: "Title".to_owned(),
            artists: Vec::new(),
            album: None,
            duration: None,
            thumbnail: None,
            artist_id: None,
            album_id: None,
            artist_ids: Vec::new(),
        }
    }
}
