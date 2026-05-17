/**
 * Service Worker — cache-first strategy for model files.
 *
 * Intercepts fetches for ONNX model files and serves from Cache API
 * on subsequent loads. Transformers.js handles its own caching for
 * HuggingFace Hub downloads, but this catches any local model files.
 */

const CACHE_NAME = 'nextera-doc-triage-v2';
const MODEL_EXTENSIONS = ['.onnx', '.onnx_data', '.bin', '.json'];

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => {
  // Clean up old cache versions
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  const isModel = MODEL_EXTENSIONS.some(ext => url.pathname.endsWith(ext));

  if (!isModel) return; // Let non-model requests pass through

  event.respondWith(cacheFirst(event.request));
});

async function cacheFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);

  if (cached) {
    notifyClients({ type: 'model-cache-hit', url: request.url });
    return cached;
  }

  notifyClients({ type: 'model-download-start', url: request.url });

  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
      notifyClients({ type: 'model-download-complete', url: request.url });
    }
    return response;
  } catch (err) {
    notifyClients({ type: 'model-download-error', url: request.url, error: err.message });
    throw err;
  }
}

async function notifyClients(message) {
  const clients = await self.clients.matchAll({ type: 'window' });
  for (const client of clients) {
    client.postMessage(message);
  }
}
