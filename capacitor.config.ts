import type { CapacitorConfig } from "@capacitor/cli";

const config: CapacitorConfig = {
  appId: "com.wordsdisplayer.app",
  appName: "WordsDisplayer",
  webDir: "dist",
  android: { allowMixedContent: false },
};

export default config;
