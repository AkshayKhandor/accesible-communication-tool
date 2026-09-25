/* MyVoice service worker
   Network-first strategy: always tries to fetch fresh content from the network.
   Falls back to cache only if offline — so the app still works without internet.

   This means you never need to hard-refresh; you always get the latest version
   automatically. Offline mode (picture board, speech, letters, numbers, body
   parts) still works perfectly. Gesture Mode still needs a connection the first
   time for the hand-tracking model. */

const CACHE = 'myvoice-v2';
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

  // Network-first for everything: always try fresh from network,
  // fall back to cache only when offline.
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
