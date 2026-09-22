/* MyVoice service worker
   Caches the app itself so it opens with no internet — important for a
   communication aid, which has to work in a car, a clinic corridor, or
   anywhere the signal drops.

   Note: Gesture Mode loads its hand-tracking model from a CDN, so that one
   feature still needs a connection the first time. Everything else — the
   picture board, speech, letters, numbers and body parts — works offline. */

const CACHE = 'myvoice-v1';
const CORE = [
  './',
  './index.html',
  './manifest.json',
  './icon-192.png',
  './icon-512.png'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(CORE)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;

  // Serve the app shell from cache first so it starts instantly and offline.
  if (CORE.some((path) => e.request.url.endsWith(path.replace('./', '')))) {
    e.respondWith(
      caches.match(e.request).then((hit) => hit || fetch(e.request))
    );
    return;
  }

  // Everything else (fonts, the hand-tracking model): try the network, fall
  // back to whatever we cached last time.
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy)).catch(() => {});
        return res;
      })
      .catch(() => caches.match(e.request))
  );
});
