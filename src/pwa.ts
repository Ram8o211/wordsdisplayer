// Offline support for the web/PWA build: caches the app shell and the model
// files so both networks keep working without a connection.
export function registerSW() {
  const isHttp = location.protocol === "https:" || location.hostname === "localhost";
  if (!("serviceWorker" in navigator) || !isHttp || import.meta.env.DEV) return;
  navigator.serviceWorker.register("sw.js").catch(() => {});
}
