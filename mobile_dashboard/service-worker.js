/**
 * Service Worker for Trading Bot Mobile Dashboard
 * Provides offline functionality and background sync
 */

const CACHE_NAME = 'trading-bot-v1.0.0';
const DYNAMIC_CACHE = 'trading-bot-dynamic-v1.0.0';

// Static assets to cache
const staticAssets = [
  '/',
  '/static/css/mobile-dashboard.css',
  '/static/js/mobile-dashboard.js',
  '/static/js/chart.min.js',
  '/static/icons/icon-192x192.png',
  '/manifest.json'
];

// API endpoints to cache for offline viewing
const apiEndpoints = [
  '/api/portfolio',
  '/api/positions',
  '/api/alerts',
  '/api/performance'
];

// Install event - cache static assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(staticAssets))
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME && cacheName !== DYNAMIC_CACHE) {
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch event - network first for API, cache first for static
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // API requests - network first with fallback
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(request)
        .then(response => {
          // Cache successful API responses
          if (response.status === 200) {
            const responseClone = response.clone();
            caches.open(DYNAMIC_CACHE)
              .then(cache => cache.put(request, responseClone));
          }
          return response;
        })
        .catch(() => {
          // Fallback to cached version
          return caches.match(request)
            .then(cachedResponse => {
              if (cachedResponse) {
                return cachedResponse;
              }
              // Return offline data structure
              return new Response(JSON.stringify({
                error: 'Offline',
                message: 'No network connection. Showing cached data.',
                timestamp: Date.now()
              }), {
                headers: { 'Content-Type': 'application/json' }
              });
            });
        })
    );
    return;
  }

  // Static assets - cache first
  event.respondWith(
    caches.match(request)
      .then(cachedResponse => {
        if (cachedResponse) {
          return cachedResponse;
        }
        return fetch(request)
          .then(response => {
            // Cache new static assets
            if (response.status === 200) {
              const responseClone = response.clone();
              caches.open(CACHE_NAME)
                .then(cache => cache.put(request, responseClone));
            }
            return response;
          });
      })
  );
});

// Background sync for trade submissions when online
self.addEventListener('sync', event => {
  if (event.tag === 'background-trade-sync') {
    event.waitUntil(syncTrades());
  }
});

// Push notification handler
self.addEventListener('push', event => {
  if (!event.data) return;

  const data = event.data.json();
  const options = {
    body: data.body || 'Trading alert',
    icon: '/static/icons/icon-192x192.png',
    badge: '/static/icons/icon-72x72.png',
    vibrate: [200, 100, 200],
    data: data.data || {},
    actions: [
      {
        action: 'view',
        title: 'View Dashboard'
      },
      {
        action: 'dismiss',
        title: 'Dismiss'
      }
    ]
  };

  event.waitUntil(
    self.registration.showNotification(data.title || 'Trading Bot', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', event => {
  event.notification.close();

  if (event.action === 'view') {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Helper function to sync trades when back online
async function syncTrades() {
  try {
    // Get pending trades from IndexedDB
    const pendingTrades = await getPendingTrades();

    for (const trade of pendingTrades) {
      try {
        await fetch('/api/trade/execute', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(trade)
        });

        // Remove from pending after successful submission
        await removePendingTrade(trade.id);
      } catch (error) {
        console.log('Trade sync failed:', error);
      }
    }
  } catch (error) {
    console.log('Background sync failed:', error);
  }
}

// Placeholder functions for IndexedDB operations
async function getPendingTrades() {
  // Implementation would use IndexedDB to get pending trades
  return [];
}

async function removePendingTrade(tradeId) {
  // Implementation would remove trade from IndexedDB
}