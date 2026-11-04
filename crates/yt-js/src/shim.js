// Copyright (c) 2026 Hunter Lehr (Barusoup)
// SPDX-License-Identifier: AGPL-3.0-only

// The browser surface that YouTube's player script touches while it loads.
//
// The player is written for a browser. Its functions for deciphering and for URL building do not
// use the page, but the module that contains them reads globals while it initialises. These stubs
// answer those reads with inert values. Nothing here reaches the network or a DOM.
//
// Keep it small. Every stub added here is something the player did not need in a real browser
// context, so each one should be justified by a load error.

var __noop = function () {};

// Anything the player reads from `document` gets an inert object back.
var __inert = function () {
  return __inert;
};
__inert.toString = function () {
  return "";
};
__inert.valueOf = function () {
  return 0;
};

var window = globalThis;
var self = globalThis;

var navigator = {
  userAgent:
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:140.0) Gecko/20100101 Firefox/140.0",
  language: "en-US",
  languages: ["en-US"],
  platform: "Win32",
  plugins: [],
  mimeTypes: [],
};

var location = {
  href: "https://music.youtube.com/",
  origin: "https://music.youtube.com",
  hostname: "music.youtube.com",
  protocol: "https:",
};

var document = new Proxy(function () {}, {
  get: function (target, prop) {
    if (prop === Symbol.toPrimitive) return function () { return ""; };
    if (prop === "length") return 0;
    return __inert;
  },
  apply: function () {
    return __inert;
  },
  construct: function () {
    return __inert;
  },
});

var XMLHttpRequest = function () {
  this.open = __noop;
  this.send = __noop;
  this.setRequestHeader = __noop;
};
var Blob = function () {};
var Image = function () {};
var Worker = function () {};
var MutationObserver = function () {
  this.observe = __noop;
  this.disconnect = __noop;
};
var MessageChannel = function () {
  this.port1 = {};
  this.port2 = {};
};
var addEventListener = __noop;
var removeEventListener = __noop;
var requestAnimationFrame = function () {
  return 0;
};
var cancelAnimationFrame = __noop;
var setTimeout = function () {
  return 0;
};
var clearTimeout = __noop;
var setInterval = function () {
  return 0;
};
var clearInterval = __noop;
var performance = {
  now: function () {
    return Date.now();
  },
};
var console = { log: __noop, warn: __noop, error: __noop, info: __noop, debug: __noop };
