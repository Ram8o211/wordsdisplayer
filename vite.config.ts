import { defineConfig } from "vite";

export default defineConfig({
  base: "./", // relative paths: works from file:// (Electron) and Capacitor WebViews
  worker: { format: "es" },
  build: { target: "es2022", chunkSizeWarningLimit: 8000 },
  test: { environment: "node" },
} as any);
