# WordsDisplayer: live speech → generative video, on-device

WordsDisplayer listens to speech through the microphone and generates, in real
time, a video that matches what is being said. Say *"a storm over the city at
night"* and you get dark clouds, rain, lightning, a lit skyline and a moon. It
runs on **Android, iOS, Windows, macOS, Linux and the web (PWA)**, and **all
inference runs locally** on the device.

The semantic core is two neural networks that you can use **separately or
together**:

| Mode | Network | Kind |
|---|---|---|
| **Traditional** | 2-layer Transformer encoder | conventional ANN |
| **Neuromorphic** | recurrent adaptive-LIF spiking neural network | event-driven SNN |
| **Hybrid** | both, confidence-weighted fusion + optional event gating | — |

Both networks ship in four precision tiers (int4 → fp32). The app benchmarks
the device and picks the best tier the hardware can sustain, so a budget phone
runs both networks in **0.6 MB** (mostly int4) while a workstation runs fp32 with longer context and
finer spiking time steps. Quantisation-aware training keeps the accuracy
loss of the low tiers small (see [measured results](#measured-results)).

```
mic ─► speech-to-text (offline) ─► words ─┬─► Transformer (ANN) ──┐
 │                                        └─► Spiking net (SNN) ──┴─► fusion ─► scene director ─► WebGL video
 └─► DSP features (loudness, bands, onsets) ──────────────────────────────────────────────┘ (reactivity)
```

---

## 1. Quick start

```bash
npm install
npm run dev            # http://localhost:5173 (allow the microphone)
```

* Press **●** and speak, use **▶ Play demo narration** in the ⚙ panel, or type a sentence.
* `?demo` autoplays the narration. `?scene=ocean,sunset&v=0.5&a=-0.2` previews a scene directly.
* `npm test` checks the TypeScript inference against the NumPy reference for all tiers.

### Desktop (Electron)

```bash
npm run electron       # run
npm run dist:desktop   # installers (.exe / .dmg / .AppImage) via electron-builder
```

### Mobile (Capacitor)

```bash
npx cap add android    # once (and/or: npx cap add ios)
npm run cap:android    # build web, sync, open Android Studio
npm run cap:ios        # same for Xcode
```

Add the permissions once the native projects exist:

* **Android** (`android/app/src/main/AndroidManifest.xml`): `<uses-permission android:name="android.permission.RECORD_AUDIO"/>`
* **iOS** (`ios/App/App/Info.plist`): `NSMicrophoneUsageDescription` and `NSSpeechRecognitionUsageDescription`

---

## 2. Network 1: the traditional network (Transformer)

**Task.** Map the stream of recently heard words to:

* 22 visual scene concepts (multi-label): `ocean, beach, forest, mountain, desert, city, space, night, rain, snow, fire, sunset, river, flowers, storm, underwater, technology, music, love, clouds, birds, fields`
* affect: **valence** and **arousal**, which drive colour grading and animation tempo.

**Why this architecture.** Topic tracking in speech needs (a) context, because a
keyword's meaning depends on its neighbours, (b) recency, because the video
should follow the *current* topic, and (c) robustness to ASR noise and
out-of-vocabulary words. It also has to cost a few million MACs per word to run
on a phone. Large pretrained encoders such as BERT or MiniLM need 20–100 MB
even at int8, which rules them out on low-end phones. The chosen design:

| Block | Choice | Reason |
|---|---|---|
| Input | hashed **unigram + bigram** embeddings (8192 buckets × 64) | no vocabulary file and no OOV failures; bigrams capture short phrases ("golden hour", "sott acqua") |
| Positions | learned **recency** positions (0 = most recent word) | lets the model weigh the latest words most |
| Encoder | 2 × pre-LN Transformer, d=64, 4 heads, FFN 192, GELU | attention captures cross-word context; pre-LN is stable under quantisation |
| Pooling | mean + last-token → MLP | global topic plus what was just said |
| Heads | 22 sigmoid + 2 tanh | multi-label concepts and continuous affect |

The model has ~0.62 M parameters, of which 0.52 M are the embedding table. One
word costs 1.4 M MACs at the 16-word window (nano) and 2.9 M at 32 words.
Trained with Adam on the bundled corpus (6000 steps, ~15 min on CPU):
micro-F1 0.94, top-1 0.99.

---

## 3. Network 2: the neuromorphic network (spiking)

**Architecture.**

```
word ─► sparse binary spike code: 10 unigram + 2 bigram afferents out of 4096
      ─► input synapses, factorised: 4096 → 26 populations → 256 neurons
      ─► 256 recurrent adaptive-LIF neurons (ALIF), heterogeneous τ_mem 0.3–3 words, τ_adapt 2–20 words
      ─► 128 LIF neurons, τ_mem 0.5–8 words
      ─► 24 leaky-integrator readout neurons (input from both spiking layers)
```

**Weights are assigned, not trained by BPTT** (`training/design_snn.py`, about
15 s on a laptop CPU). The approach follows the Neural Engineering Framework:

* **Populations.** Layer 1 has 10 neurons per visual concept (220) plus four
  pools of 9 for valence+/−, arousal+/−. Layer 2 mirrors them with slower neurons.
* **Input synapses.** These come from a closed-form minimum-norm ridge solution
  that maps the spike code of each of the ~900 vocabulary words onto the current
  its populations should receive. Keywords drive their concept population,
  mood words drive the affect pools, and function words and small talk drive
  nothing. The matrix has rank 26, so it is stored factorised as
  4096×26 decoders and 26×256 heterogeneous encoders. That makes it 10× smaller
  and needs 10× fewer synaptic operations per input spike, which matters on
  low-end phones.
* **Recurrent synapses.** Each population excites itself, an attractor that
  keeps the topic alive for a few words until threshold adaptation ends it.
  Concept populations inhibit each other (winner-take-most), so a new topic
  displaces the old one.
* **Readout.** The only fitted part is a logistic regression (concepts) and a
  least-squares fit (affect) on the simulated spike traces. Both are convex, and
  they are fitted jointly at T = 2, 4 and 8 so one readout works at every
  time resolution.

Why this beats training it. Surrogate-gradient BPTT is in the repo
(`train.py`, with optional per-word distillation from the Transformer). After an
hour of CPU training it reached micro-F1 ≈ 0.53, while the constructed network
reaches **0.81 in seconds**. The construction is also reproducible and
interpretable: every neuron has a known meaning.

**Why spiking.**

* **Event-driven inference** (`src/nn/snn.ts`). Weights are only read when a
  presynaptic neuron spikes, so the work is proportional to the number of
  synaptic events (accumulates only, no multiplies). Spike rates are 0.7–1.7 %,
  which gives **~4 k synaptic ops per word vs. 1.4–4.5 M MACs** for the
  Transformer.
* **Streaming by construction.** Membrane state carries the context, so a
  window is never re-processed. This is the always-on, low-power network.
* **Hardware-friendly.** Binary spike input, LIF/ALIF neurons, int4/int8
  synapses and thresholds map directly to neuromorphic chips (Loihi 2, Akida,
  SpiNNaker 2).
* **Time in word units.** Time constants are expressed in words and
  discretised with **T steps per word**: T=2 on a phone, up to T=8 on a
  workstation.

## 4. Using one, the other, or both

* **Traditional**: best accuracy, more computation per word.
* **Neuromorphic**: cheapest, streaming, low power.
* **Hybrid**: `p = w·p_ANN + (1−w)·p_SNN`, where *w* is each network's
  confidence (mean |2p−1|) normalised across the two. The same formula is used
  in `export.py` to measure the hybrid.
  * **Event-gated hybrid** (on by default). The SNN runs on every word. The
    Transformer only runs when the SNN's output moves by more than a threshold
    (topic change), and at least every 5th word. Most steady-topic words then
    cost only SNN events. The ⚙ panel shows runs vs. skips.

A **scene director** sits between the networks and the renderer. It keeps the
top-K concepts with hysteresis, applies a fast-attack / slow-release envelope,
and keeps the current scene through small talk, so the video changes with the
topic without flickering.

---

## 5. Quantisation and hardware scalability

`training/export.py` builds every tier from one set of master weights.

1. **Any-precision quantisation-aware training.** In the last 35 % of
   training, each step fake-quantises all weight matrices to a randomly chosen
   precision ∈ {int4, int8, fp32}, with a straight-through estimator. One set of
   weights is therefore robust at every precision.
2. **Formats** (`src/nn/qtensor.ts`). Symmetric weight-only quantisation:
   * `q4`: int4 packed 2 per byte, one fp32 scale per **group of 32** inputs
   * `q8`: int8, one scale **per output channel**
   * `f16`: IEEE half, decoded through a lookup table
   * `f32`

   Weights **stay compressed in memory**. The kernels (`rowDot` for the
   Transformer, `addRow` for embeddings and spike events) dequantise on the fly.
3. **Sensitivity-driven mixed precision.** For each tensor and precision, the
   exporter measures the validation-loss increase when only that tensor is
   quantised. For each tier it then solves a knapsack greedily: start every
   tensor at the cheapest allowed precision, then repeatedly upgrade the tensor
   with the best *loss reduction per extra byte* until the tier's byte budget
   is spent. Sensitive layers (e.g. the embedding, the recurrent SNN matrix)
   get more bits; robust ones stay at int4.
4. **Runtime tiers.** The tier sets precision and also the compute knobs:

| Tier | Precisions allowed | Byte budget | ANN window | SNN steps/word | Render scale | Typical device |
|---|---|---|---|---|---|---|
| nano | int4, int8 | 55 % of int8 | 16 words | 2 | 0.45 | low-end phone |
| mobile | int4, int8, fp16 | int8 size | 24 | 4 | 0.6 | mid/high-end phone |
| desktop | int8, fp16 | fp16 size | 32 | 6 | 0.85 | laptop |
| ultra | fp32 | fp32 size | 48 | 8 | 1.0 | workstation |

5. **Device profiler** (`src/nn/device.ts`). It measures JS matrix-vector
   throughput (60 ms micro-benchmark) and combines it with core count, device
   memory, mobile/desktop and GPU class into a score that selects the tier.
   **Auto** is the default; any tier can be forced in ⚙.

### Measured results

Held-out set of 1000 synthetic speech excerpts. Metrics are computed on the
**packed, quantised weights** of each tier (`public/models/report.json`).

| Tier | Transformer size | Transformer micro-F1 / top-1 | SNN size | SNN micro-F1 / top-1 | Hybrid micro-F1 / top-1 |
|---|---|---|---|---|---|
| nano | 433 KB | 0.943 / 0.993 | 150 KB | 0.811 / 0.921 | 0.955 / 0.989 |
| mobile | 452 KB | 0.943 / 0.993 | 183 KB | 0.815 / 0.917 | 0.957 / 0.988 |
| desktop | 676 KB | 0.944 / 0.990 | 248 KB | 0.812 / 0.913 | 0.955 / 0.988 |
| ultra | 2425 KB | 0.946 / 0.991 | 867 KB | 0.817 / 0.914 | 0.952 / 0.988 |

* The **hybrid beats either network alone** on micro-F1 at every tier. The
  two networks make different mistakes, and confidence weighting fixes many of
  them.
* **Low tiers lose almost nothing**: nano is ~5× smaller than fp32 and within
  0.3 points of its F1, thanks to any-precision QAT and sensitivity-driven
  allocation. Higher tiers mainly add context length (ANN window) and temporal
  resolution (SNN steps).
* Mixed-precision allocation chosen for nano: Transformer `emb int4, b0.qkv int8, b0.o int8, b0.f1 int8, b0.f2 int8, b1.qkv int8, b1.o int8, b1.f1 int8, b1.f2 int8, h1m int8, h1l int8, h2 int8`; SNN `w_in_a int4, w_in_e int8, w_rec int4, w_12 int4, w_out int8, w_out1 int8`.

Per-word latency in Node.js on one core of a 2.1 GHz Xeon (`npx vitest run tests/bench.test.ts`):

| Tier | Transformer | Spiking net | SNN spike rate | SNN synaptic ops/word | Transformer MACs/word |
|---|---|---|---|---|---|
| nano | 2.8 ms | 0.025 ms | 1.7 % | 3.3 k | 1.4 M |
| mobile | 4.6 ms | 0.024 ms | 1.2 % | 4.4 k | 2.1 M |
| desktop | 6.4 ms | 0.026 ms | 0.9 % | 4.7 k | 2.9 M |
| ultra | 8.7 ms | 0.034 ms | 0.7 % | 5.0 k | 4.5 M |

People speak ~2–3 words/s, so even a phone 10× slower than this has 30× headroom.

### Low-end hardware optimisations

* **Weight-only int4/int8 kept compressed in RAM**, dequantised inside the
  kernels. Head layers and other tensors under 32 k elements never go below
  int8: they cost a few KB, and int4 there damages the calibrated outputs.
* **Factorised SNN input synapses** (4096×26 + 26×256 instead of 4096×256):
  154 KB for the whole nano SNN.
* **Event-driven SNN kernels**: only spiking neurons' rows are touched, using
  active-neuron lists. An LRU cache holds per-word input currents.
* **Allocation-free hot loops** in both networks (preallocated buffers), so no
  GC pauses on weak devices.
* **Inference in a Web Worker**, off the render thread.
* **Event-gated hybrid**: the Transformer is skipped while the SNN reports a
  stable topic.
* **Tier-scaled compute**: ANN window 16→48 words, SNN steps 2→8, shader noise
  octaves 3→6, max scene layers 3→6.
* **Dynamic resolution**: the renderer lowers its resolution when frames take
  longer than 1/45 s and raises it again when there is headroom.


---

## 6. Speech recognition (front-end)

The networks consume words, so the speech-to-text front-end is pluggable
(`src/asr/index.ts`). Every option can run offline:

* **Vosk** (default on desktop/web): Kaldi compiled to WebAssembly, fully
  offline everywhere. A ~40 MB model is downloaded once and cached. English and
  Italian models are preconfigured.
* **Native** (default in the mobile apps): Android `SpeechRecognizer` / iOS
  `SFSpeechRecognizer` via Capacitor, which are on-device on recent OS versions.
* **Web Speech API**: uses Chrome's on-device mode (`processLocally`) when
  available. Otherwise the browser may use a cloud service, and the app says so.
* **Typed text / demo narration**: works without a microphone.

Partial hypotheses are turned into a stream of newly-heard words (the last,
unstable word of a partial is held back), so the video reacts while you are
still talking.

---

## 7. Video generation

`src/render/shader.ts` is a single WebGL2 fragment shader. Each concept is a
procedural layer: sky (day/sunset/night), stars, nebula and planet, sun and
moon, clouds, birds, mountains with snow caps, skyline with lit windows,
fields, dunes, a three-depth forest, ocean with sun glitter, beach with a palm,
a winding river, flowers, fire, underwater caustics with fish and bubbles, a
neon tech grid with data rain, an audio-reactive equaliser, hearts, rain, snow
and lightning.

Layer opacities are the director's smoothed concept weights. **Valence** sets
warmth and saturation, **arousal** sets animation tempo and contrast, and the
**microphone** adds loudness and onset pulses. Concepts compose ("snowy forest
by a river at sunset"). Shader detail (noise octaves) and render resolution
follow the tier.

> **Design note.** Pixel-level neural video generators (diffusion / video
> transformers) need multi-GB models and desktop-class GPUs, and are far from
> real time on a phone. To meet "real time" and "runs on a mobile phone"
> together, the neural networks do the **understanding** (what to show, and
> with which mood) and a procedural GPU renderer does the **drawing**, at
> 60 fps on phones. The networks' output is a compact scene description
> (22 weights + valence/arousal), so a heavier neural renderer could replace
> the shader on high-end hardware without changing the networks.

---

## 8. Training and re-exporting the models

The weights are committed in `public/models/`. Rebuilding them needs only
NumPy; there is no deep-learning framework dependency (a small reverse-mode
autodiff lives in `training/autograd.py`):

```bash
cd training
python3 train.py --only ann --steps-ann 6000   # Transformer (~15 min on CPU)
python3 design_snn.py                          # spiking net: assigned weights + fitted readout (~15 s); --search tunes the dynamics
python3 export.py                              # sensitivity analysis, 4 tiers, report, golden vectors
cd .. && npm test                              # TS ↔ NumPy parity
```

`python3 train.py --only snn` is the alternative BPTT path for the SNN
(surrogate gradients + per-word distillation from the Transformer). It is
slower and, on this data, less accurate than `design_snn.py`.

`train.py --extra corpus.jsonl` mixes in your own annotated sentences
(`{"text": ..., "concepts": {"ocean": 1}, "valence": 0.4, "arousal": 0.2}`).
The bundled corpus is synthetic: hand-curated English + Italian lexicons and
templates, built by `training/dataset.py`. Real transcripts would improve
coverage of unusual phrasing.

---

## 9. Project layout

```
src/nn/text.ts       tokeniser + FNV-1a hashing (bit-exact with training/common.py)
src/nn/qtensor.ts    int4 / int8 / fp16 / fp32 tensors and kernels
src/nn/ann.ts        Transformer inference
src/nn/snn.ts        event-driven spiking-network inference
src/nn/fusion.ts     hybrid fusion
src/nn/engine.ts     modes, event gating, tier loading (runs in a Web Worker: worker.ts)
src/nn/device.ts     hardware profiler → tier
src/asr/             speech-to-text front-ends
src/audio/mic.ts     microphone DSP features
src/render/          scene director + WebGL2 renderer
training/            dataset, autodiff, models, ANN training, SNN weight construction, quantisation/export
public/models/       exported tiers (*.json manifest + *.bin weights) + report.json
electron/            desktop shell        capacitor.config.ts  mobile shell
```
