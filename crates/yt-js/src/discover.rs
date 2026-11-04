// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

//! Finding this build's entry points in `base.js`, by the shape of the code that calls them.
//!
//! YouTube's player renames its functions and changes their numeric arguments on every build, so
//! nothing here looks up a name. Each entry point is matched by what its call site does:
//!
//! - **Signature pipeline.** Somewhere the player reads a ciphered format's `s` value and runs it
//!   through two VM dispatch calls and a helper: `x=D1(o1,k1,H(oh,kh,obj.s))`. The result is set
//!   on the URL under the `sp` parameter by a call `set(sp, D2(o2,k2,x))`.
//! - **URL class.** Stream URLs are built through a player class: `new g.C(url,!0)`. Its builder
//!   method is found at runtime (see `engine.rs`), because the method name also changes.
//! - **Signature timestamp.** A literal `signatureTimestamp:NNNNN` (or `sts:`) in the config.
//!
//! When a build changes the shape, discovery fails with a clear error. It does not guess.

use regex::Regex;

/// One VM dispatch call: `name(op, key, ...)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    pub name: String,
    pub op: u32,
    pub key: u32,
}

/// The signature pipeline: `D2(o2,k2, D1(o1,k1, H(oh,kh, s)))`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SigPipeline {
    pub helper: Call,
    pub first: Call,
    pub second: Call,
}

/// Everything the engine needs from one player build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entries {
    pub sig: SigPipeline,
    /// The URL class, e.g. `aj`. It is reached as `g.<class>` inside the player's closure.
    pub url_class: String,
    /// The player's own signature timestamp, if the config carries one.
    pub sts: Option<u32>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DiscoverError {
    #[error("no signature pipeline found: the player's cipher layout has changed")]
    NoSignaturePipeline,
    #[error("no URL class found: the player's URL layout has changed")]
    NoUrlClass,
}

pub fn discover(src: &str) -> Result<Entries, DiscoverError> {
    let sig = signature_pipeline(src)?;
    let url_class = url_class(src).ok_or(DiscoverError::NoUrlClass)?;
    Ok(Entries {
        sig,
        url_class,
        sts: signature_timestamp(src),
    })
}

/// `x=D1(o1,k1,H(oh,kh,obj.s))` followed, somewhere later, by `....set-like(sp, D2(o2,k2,x))`.
fn signature_pipeline(src: &str) -> Result<SigPipeline, DiscoverError> {
    let first_re = Regex::new(r"(\w+)=(\w+)\((\d+),(\d+),(\w+)\((\d+),(\d+),\w+\.s\)\)")
        .expect("static regex");
    for caps in first_re.captures_iter(src) {
        let var = &caps[1];
        let first = Call {
            name: caps[2].to_owned(),
            op: caps[3].parse().unwrap_or(0),
            key: caps[4].parse().unwrap_or(0),
        };
        let helper = Call {
            name: caps[5].to_owned(),
            op: caps[6].parse().unwrap_or(0),
            key: caps[7].parse().unwrap_or(0),
        };
        if let Some(second) = second_call(src, var) {
            return Ok(SigPipeline {
                helper,
                first,
                second,
            });
        }
    }
    Err(DiscoverError::NoSignaturePipeline)
}

/// The call that feeds `var` into the setter: `[..](name, D2(o2,k2,var))`.
fn second_call(src: &str, var: &str) -> Option<Call> {
    let re = Regex::new(r"\[\w+\[\d+\]\]\(\w+,(\w+)\((\d+),(\d+),(\w+)\)\)").expect("static regex");
    let found = re.captures_iter(src).find(|caps| &caps[4] == var)?;
    Some(Call {
        name: found[1].to_owned(),
        op: found[2].parse().unwrap_or(0),
        key: found[3].parse().unwrap_or(0),
    })
}

/// `new g.<class>(<something>.url,!0)`. Only the class name matters here.
fn url_class(src: &str) -> Option<String> {
    let re = Regex::new(r"new \w+\.(\w+)\(\w+\.url,!0\)").expect("static regex");
    re.captures(src).map(|c| c[1].to_owned())
}

/// `signatureTimestamp:NNNNN` or `sts:NNNNN`, as the player's config states it.
pub fn signature_timestamp(src: &str) -> Option<u32> {
    let re = Regex::new(r#"(?:signatureTimestamp|sts)"?\s*:\s*(\d{5})\b"#).expect("static regex");
    re.captures(src)?[1].parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same shapes as the player, with invented names and numbers.
    const SNIPPET: &str = r#"
        var Zq=function(a,b,c){var d=b^a;return c};
        var Hx=function(a,b,c){return c};
        Wa=function(a,b,c,d){for(let k of c){if(!k.url)continue;c=new g.Qv(k.url,!0);if(k.s){var f=c,h=k.sp;let l=Zq(7,771,Hx(14,908,k.s));f[w[4]](h,Zq(9,1814,l))}}};
        var cfg={signatureTimestamp:20753,other:1};
    "#;

    #[test]
    fn finds_the_pipeline_by_shape_not_by_name() {
        let e = discover(SNIPPET).expect("entries");
        assert_eq!(
            e.sig.helper,
            Call {
                name: "Hx".into(),
                op: 14,
                key: 908
            }
        );
        assert_eq!(
            e.sig.first,
            Call {
                name: "Zq".into(),
                op: 7,
                key: 771
            }
        );
        assert_eq!(
            e.sig.second,
            Call {
                name: "Zq".into(),
                op: 9,
                key: 1814
            }
        );
        assert_eq!(e.url_class, "Qv");
        assert_eq!(e.sts, Some(20753));
    }

    #[test]
    fn a_changed_layout_fails_loudly() {
        assert_eq!(
            discover("var x=1;"),
            Err(DiscoverError::NoSignaturePipeline)
        );
    }

    #[test]
    fn pipeline_without_its_setter_is_not_accepted() {
        let src = "let l=Zq(7,771,Hx(14,908,k.s));";
        assert_eq!(discover(src), Err(DiscoverError::NoSignaturePipeline));
    }
}
