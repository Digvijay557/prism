// Minimal service worker — just enough to make the app installable.
// Caches the shell so it opens instantly even on flaky connections.

const CACHE_NAME = "prism-cache-v1";
const SHELL_FILES = ["/"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_FILES))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  // Network-first: always try live data (important since this app checks
  // live YouTube URLs), fall back to cache only if offline.
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});