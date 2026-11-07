// Desktop shell (Windows / macOS / Linux). Serves the built web app from a
// custom privileged scheme so fetch() of the model files and Web Workers work,
// and grants microphone access.
const { app, BrowserWindow, protocol, net, session, systemPreferences } = require("electron");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

protocol.registerSchemesAsPrivileged([
  { scheme: "app", privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true, stream: true } },
]);

const DIST = path.join(__dirname, "..", "dist");

async function createWindow() {
  protocol.handle("app", (req) => {
    const { pathname } = new URL(req.url);
    const file = path.normalize(path.join(DIST, decodeURIComponent(pathname)));
    if (!file.startsWith(DIST)) return new Response("forbidden", { status: 403 });
    return net.fetch(pathToFileURL(file).toString());
  });
  session.defaultSession.setPermissionRequestHandler((_wc, perm, cb) => cb(perm === "media"));
  session.defaultSession.setPermissionCheckHandler((_wc, perm) => perm === "media");
  if (process.platform === "darwin") await systemPreferences.askForMediaAccess("microphone");
  const win = new BrowserWindow({
    width: 1280, height: 800, backgroundColor: "#05060c", autoHideMenuBar: true,
    webPreferences: { contextIsolation: true, sandbox: true },
  });
  win.loadURL("app://bundle/index.html");
}

app.whenReady().then(createWindow);
app.on("window-all-closed", () => { if (process.platform !== "darwin") app.quit(); });
app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
