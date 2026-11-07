/// <reference lib="webworker" />
// Both networks run here, off the UI/render thread.
import { Engine, type Mode } from "./engine";
import type { Tier } from "./device";

export type WorkerIn =
  | { type: "load"; base: string; tier: Tier }
  | { type: "mode"; mode: Mode; eventGated: boolean }
  | { type: "text"; text: string }
  | { type: "reset" };

const engine = new Engine();
let ready = false;
const queue: string[] = [];

function run(text: string) {
  const r = engine.push(text);
  if (r) postMessage({ type: "result", result: r });
}

self.onmessage = async (ev: MessageEvent<WorkerIn>) => {
  const m = ev.data;
  try {
    switch (m.type) {
      case "load":
        ready = false;
        await engine.load(m.base, m.tier);
        ready = true;
        postMessage({ type: "loaded", tier: m.tier, annBytes: engine.annBytes, snnBytes: engine.snnBytes,
          annWindow: engine.ann.window, snnSteps: engine.snn.steps });
        while (queue.length) run(queue.shift()!);
        break;
      case "mode":
        engine.mode = m.mode;
        engine.eventGated = m.eventGated;
        break;
      case "text":
        if (ready) run(m.text);
        else queue.push(m.text);
        break;
      case "reset":
        engine.reset();
        break;
    }
  } catch (e) {
    postMessage({ type: "error", message: String((e as Error)?.message ?? e) });
  }
};
