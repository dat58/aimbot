# AimBot

An AI-powered aimbot written in Rust.

## Layout

Two machines:

- **PC1** — the game machine. Sends video to PC2 over NDI or UDP, and keyboard
  and mouse presses to PC2's event listener. The keylogger that does the latter
  lives in [aimbot-keylogger](https://github.com/dat58/aimbot-keylogger).
- **PC2** — **Ubuntu 24.04**, runs this project in Docker. Takes frames in,
  runs detection, and drives the mouse over a serial link (MAKCU) back to PC1.

A capture card attached directly to PC2 replaces the NDI/UDP hop entirely; see
[Capture from a capture card](#capture-from-a-capture-card).

## Frame sources

`SOURCE_STREAM` picks the source by prefix:

| `SOURCE_STREAM` | Source |
| --- | --- |
| `ndi://172.24.36.133` | NDI. Comma-separate several addresses to add more discovery targets. |
| `elgato://` or `v4l2://` | Capture card on `CAPTURE_DEVICE` |
| `elgato:///dev/video1` | Capture card on the named node |
| anything else | Passed to FFmpeg as a URL (UDP/RTSP/...) |

NDI is behind cargo features: `ndi4` (default) or `ndi6` — see
`Dockerfile.ndi6`.

## Prepare a model

Two model families are supported, and the pipeline is chosen from the model
file name, so nothing extra needs configuring.

### Two-class YOLO

Letterboxed preprocess (`INTER_LINEAR`, padded with 114), two classes:

- `0`: entire body
- `1`: head

### `v_light_*` models

`v_light_192_fp16_onnx.onnx` (from the capfkaplus repo) needs a different
preprocess and output decode, both of which live in `impl Model` in
`src/model.rs` next to the existing path. The code is written against
`MODEL_INPUT_SIZE` and the runtime output shape, so the sibling `v_light_256`
works the same way with nothing but a different `MODEL_PATH` and
`MODEL_INPUT_SIZE`:

- **Preprocess** (`preprocess_light`) — nearest-neighbour stretch of the region
  into `MODEL_INPUT_SIZE` square with **no letterbox and no padding**, matching
  capfkaplus: each axis gets its own scale. Then BGR to RGB, NCHW, `1/255`.
  Nearest neighbour rather than bilinear because that is how the graph was
  exported.
- **Postprocess** (`decode_light`) — the graph emits raw candidates as
  `[1, 4 + classes, N]`: `cx, cy, w, h` in model-input pixels followed by one
  score per class, with no objectness channel and no NMS inside the graph.
  Boxes are multiplied by the inverse per-axis factor
  (`sx = REGION_WIDTH / MODEL_INPUT_SIZE`,
  `sy = REGION_HEIGHT / MODEL_INPUT_SIZE`), then the existing per-class
  `non_max_suppression` runs on `MODEL_IOU`, exactly as on the two-class path.

  Nothing is baked in: the input side comes from `MODEL_INPUT_SIZE`, and the
  channel and candidate counts are read from the output shape at run time
  (`[1, 9, 756]` for `v_light_192`, `[1, 9, 1344]` for `v_light_256`).

  The stretch keeps the whole model input carrying image rather than padding,
  at the cost of distorting a non-square region. With a square region — the
  usual configuration, and the only shape capfkaplus ever feeds the model —
  both axes get the same factor and the question does not arise.
- **Classes** — the model emits five (body, head, tm, ability, flash). Only
  `0: body` and `1: head` are targets and they keep the same meaning the
  two-class models use, so nothing downstream changes. The score is an argmax
  over all five, so a teammate or ability box wins its own candidate and is
  skipped rather than being read as a body.

The path is selected by the model file name — a name starting with `v_light`
switches it on, anything else keeps the letterboxed two-class path:

```shell
MODEL_PATH=/app/assets/v_light_192_fp16_onnx.onnx
MODEL_INPUT_SIZE=192
```

`MODEL_INPUT_SIZE` must match what the graph declares — 192 for `v_light_192`,
256 for `v_light_256` — otherwise ONNX Runtime rejects the input tensor shape.
`MODEL_CONF_BODY`, `MODEL_CONF_HEAD` and `MODEL_IOU` apply unchanged. Despite
the `_fp16_` in the file name the graph has **float32** inputs and outputs — the
name refers to how the weights were exported.

## OpenVINO

OpenVINO is the provider these builds target. Use `Dockerfile.openvino`:

```shell
docker build -f Dockerfile.openvino -t aimbot:1.0.0-openvino .
```

All Dockerfiles build with `cargo build --release --locked`. `Cargo.lock` pins
`ort` and `ort-sys` to `2.0.0-rc.9`, and the requirement in `Cargo.toml` is a
caret on a prerelease, which would also accept `rc.10` — where the API changed
(`try_extract_raw_tensor` is gone, `with_arena_allocator()` takes an argument).
`--locked` makes a drifting resolve fail the build instead of silently
compiling against a different ONNX Runtime binding.

No Cargo change is needed — `ort` gates its OpenVINO registration on
`any(feature = "load-dynamic", feature = "openvino")` and `load-dynamic` is
already enabled, so **do not** add the `openvino` feature. What OpenVINO needs
is purely runtime, and that is what the extra Dockerfile provides:

- a `libonnxruntime.so` built **with** the OpenVINO EP — the base image's ONNX
  Runtime is a CUDA/TensorRT build and does not have it, so the image pulls
  Intel's `onnxruntime-openvino` wheel and copies the libraries out of it;
- the OpenVINO runtime, its CPU/GPU/NPU plugins and TBB. They all carry
  `RPATH=$ORIGIN`, so they resolve each other as long as they stay together —
  hence `/app/ort/` plus `ORT_DYLIB_PATH=/app/ort/libonnxruntime.so`;
- an `ubuntu:22.04` runtime layer. The builder image is itself Ubuntu 22.04, so
  every library the binary carries over meets the exact glibc it was linked
  against, and `libstdc++`, `libgcc_s` and `libz` are simply there.

### Runtime libraries

The base image sets `OPENCV_LINKAGE=dynamic`, so the binary needs
`libopencv_*.so` and their codec dependencies at run time, and `ort` is built
with `load-dynamic`, so ONNX Runtime is `dlopen`ed rather than linked. Instead
of hand-listing those paths — which move between base-image revisions — the
`runtime-libs` stage runs `ldd` on the built binary and copies whatever it
resolves into `/usr/local/lib/aimbot`, registered with `ldconfig`. That picks up
OpenCV, NDI and NDI's avahi/dbus/systemd/cap dependencies in one go: 246
libraries, against the 6 a hand-written list carried. glibc is excluded on
purpose — the runtime image already has a matching copy and swapping libc out
from under the loader is a bad idea.

The final stage ends with an `ldd` check, so a missing library fails the build
instead of the deployment.

Note that `Dockerfile`, `Dockerfile.ndi6` and `Dockerfile.mouse_test` still use
the original distroless final stage, which copies only those 6 libraries and
therefore does not carry OpenCV. Only `Dockerfile.openvino` has the `ldd`
harvest.

### Choosing `ORT_OPENVINO_VERSION`

The wheel bundles a matched ONNX Runtime and OpenVINO pair, so this one argument
picks both:

| Wheel | ONNX Runtime | OpenVINO | With `ort` 2.0.0-rc.9 |
| --- | --- | --- | --- |
| `1.20.0` | 1.20.0 | 2024.4 | works |
| `1.21.0` | 1.21.0 | 2025.0 | **registration fails, silent CPU fallback** |
| `1.22.0` | 1.22.0 | 2025.1 | works |
| `1.23.0` | 1.23.0 | 2025.3 | works — current default |
| `1.24.1` | 1.24.1 | 2025.4.1 | works |

`ort` rc.9 registers OpenVINO through the legacy `OrtOpenVINOProviderOptions`
struct. ONNX Runtime **1.21** rejects the `0`/`1` booleans it fills in:

```
ERROR ort::execution_providers: An error occurred when attempting to register
  `OpenVINOExecutionProvider`: [OpenVINO-EP] enable_opencl_throttling should be a boolean.
WARN  ort::execution_providers: No execution providers registered successfully. Falling back to CPU.
```

That failure is **not fatal** — ORT falls back to the CPU provider and the
process runs normally, just several times slower. 1.22 and later accept the
legacy struct again, so 1.21.0 is the one version to avoid.

Separately, rc.9 warns on any runtime that is not 1.20.x:

```
WARN ort: ort 2.0.0-rc.9 may have compatibility issues ...; expected
  GetVersionString to return '1.20.x', but got '1.23.0'
```

That one is cosmetic — rc.9 negotiates C API version 20, which every later ONNX
Runtime still serves. Do not confuse it with the registration failure above.
Whatever version you pick, confirm the provider actually took at startup:

```
INFO ort::execution_providers: Successfully registered `OpenVINOExecutionProvider`
```

The same check is worth doing on the CUDA/TensorRT images: a provider that fails
to register is only a warning in the log.

### Configuration

```shell
MODEL_PROVIDER=openvino
OPENVINO_DEVICE_TYPE=CPU     # or GPU / NPU / GPU.0 ...
OPENVINO_CACHE_DIR=/app/assets
INTRA_THREADS=1
```

`OPENVINO_DEVICE_TYPE=GPU` needs `/dev/dri` passed into the container; the GPU
and NPU plugins are already in the image.

Median of 4 runs on a Xeon 8358P, `INTRA_THREADS=1`, 1920x1080 input, provider
confirmed registered:

| Model | Provider | ONNX Runtime / OpenVINO | inference |
| --- | --- | --- | --- |
| `v_light_256` | OpenVINO CPU | 1.23.0 / 2025.3 | **2.10 ms** |
| `v_light_256` | OpenVINO CPU | 1.20.0 / 2024.4 | 2.19 ms |
| `v_light_256` | ORT CPU | 1.23.0 | 7.49 ms |
| `v_light_192` | OpenVINO CPU | 1.23.0 / 2025.3 | **1.44 ms** |
| `v_light_192` | OpenVINO CPU | 1.20.0 / 2024.4 | 1.50 ms |

Preprocess is 0.26-0.28 ms at 256 and 0.15 ms at 192; postprocess is under
0.015 ms. OpenVINO is roughly 3.5x faster than the default CPU provider, which
is why the silent fallback above matters: with it, the two benchmark identically
because both are running on the CPU one. Moving 2024.4 to 2025.3 is worth about
4% on this server CPU — measure on the box you deploy to, where a newer
microarchitecture or an Intel iGPU may widen the gap.

The CUDA/TensorRT path is still in `Dockerfile` and needs
[nvidia-container-toolkit](https://gist.github.com/atinfinity/f9568aa9564371f573138712070f5bad)
on the host. OpenVINO does not.

## Capture from a capture card

Set `SOURCE_STREAM=elgato://` (or `v4l2://`) to read frames straight from the
card's `/dev/videoN` node instead of over NDI/UDP. Tested against an Elgato
4K X, but nothing in the path is model-specific — any UVC card works.

The card enumerates through `uvcvideo` whether it is plugged into a USB4 /
Thunderbolt port or a plain USB 3.2 one: a USB4 port exposes a USB 3.2 host, so
nothing here is port-specific. What the port *does* decide is available
bandwidth, and therefore which resolution and rate the card will agree to. The
driver adjusts silently rather than failing, so the negotiated mode is logged at
startup — check it against what you asked for.

There is **no library to install**. The capture path talks to V4L2 through
`open`/`ioctl`/`mmap`/`poll` directly: not `libv4l`, not OpenCV's `videoio`. On
Ubuntu 24.04 the `uvcvideo` module is in the stock kernel, so the node appears
as soon as the card is plugged in:

```shell
ls -l /dev/video*
v4l2-ctl --list-formats-ext -d /dev/video0   # optional, from v4l-utils
```

The only requirement is handing the node to the container:

```yaml
    devices:
      - "/dev/video0:/dev/video0"
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `CAPTURE_DEVICE` | `/dev/video0` | V4L2 node |
| `CAPTURE_WIDTH` | `1920` | Requested capture width |
| `CAPTURE_HEIGHT` | `1080` | Requested capture height |
| `CAPTURE_FPS` | `60` | Requested frame rate; `0` leaves the device default |
| `CAPTURE_FOURCC` | (negotiated) | Pin a pixel format: `NV12`, `YU12`, `YUYV`, `UYVY`, `BGR3`, `RGB3`, `AR24`, `MJPG` |
| `CAPTURE_BUFFERS` | `3` | MMAP ring size; 2 is the minimum the kernel accepts |
| `CAPTURE_DROP_STALE` | `true` | Drain the driver queue each grab and keep only the newest frame |
| `CAPTURE_TIMEOUT_MS` | `1000` | Grab timeout before the stream is treated as lost |
| `CAPTURE_OUTPUT` | `bgr` | `bgr` converts for the detection pipeline; `raw` hands back the untouched device bytes |

What keeps the latency down:

- `MMAP` buffers, so a dequeued frame is read out of the DMA buffer with no copy
  on the capture side;
- a small buffer ring, so the driver cannot build a queue of frames ahead of us;
- after the first frame is ready, keep dequeuing until the driver reports
  `EAGAIN` and hand back only the newest one, returning the older buffers
  immediately. A frame that is already stale by the time it is looked at is
  worse than no frame for aiming;
- `O_NONBLOCK` plus a single `poll()` per grab, so a dead signal surfaces as a
  timeout instead of a hung thread;
- format negotiation prefers the 12-bit planar formats (NV12, YU12) over the
  16-bit packed ones and puts MJPEG last, because bytes on the wire are latency
  on a USB card and MJPEG adds a decode pass on top of the transfer.

With `CAPTURE_OUTPUT=bgr` there is exactly one pass over the pixels
(`cvtColor`). `CAPTURE_OUTPUT=raw` skips even that and is the lowest-latency
mode available, but the frames are then in the device's native layout — NV12 for
example arrives as `height * 3 / 2` rows of single-channel data — and the
detection pipeline expects BGR, so `raw` is for measurement and for consumers
that interpret the layout themselves.

## Running on Ubuntu 24.04

Install Docker Engine and the Compose plugin, then pass through the devices the
build needs. `docker-compose.yml` uses `network_mode: host`, so the event
listener is reachable on the host's address directly.

```yaml
    devices:
      - "/dev/ttyACM0:/dev/ttyACM0"   # MAKCU_PORT
      - "/dev/video0:/dev/video0"     # CAPTURE_DEVICE, capture-card source only
      - "/dev/dri:/dev/dri"           # OPENVINO_DEVICE_TYPE=GPU only
```

The container runs as root, so it can open those nodes as-is. To reach them as
your own user outside the container:

```shell
sudo usermod -aG dialout,video "$USER"    # re-login afterwards
```

Open the event listener port if a firewall is active:

```shell
sudo ufw allow 10000/tcp
```

The listener serves `GET /health`, `GET /stream/status`,
`GET /stream/board` (the board controller UI) and
`PUT /stream/event/{id}`, bound to `0.0.0.0:$EVENT_LISTENER_PORT`.

## Mouse movement

A mouse delta is played one of two ways:

| Movement | What it is |
| --- | --- |
| `move_smooth` | The host plans a human flick and plays it as a stream of small `km.move` reports on a fixed cadence. |
| `move_bezier` | The original single `km.move(dx,dy,steps,ref_x,ref_y)`, interpolated by the MAKCU firmware. |

### Which one is used

| Situation | Target chosen by | Movement |
| --- | --- | --- |
| ESP button 2 held | `aim_smooth` | **always** `move_smooth` |
| Otherwise | the current aim mode | `move_smooth` when `MOVE_SMOOTH=true`, else `move_bezier` |

`aim_smooth` chooses its target exactly like the Horizon mode: x goes straight
to the centre of the box, while y is only nudged toward it a random amount each
frame instead of being snapped onto it — up to 21 px down when the crosshair is
above the box, up to 11 px up toward a head box or 21 px toward a body box when
it is below, and −14 to +7 px when it is already level with a body box. Level
with a head box, it aims straight at the head. It
ignores `MOVE_SMOOTH` — the hand-like movement is the point of that path — and
uses `SCALE_MIN_ZONE2`.

`MOVE_SMOOTH` defaults to `false`, so the ordinary modes keep the firmware
bezier unless you opt in. It is read at startup; changing it needs a restart.

### What a flick models

- Movement time grows with the **logarithm** of the distance (Fitts), not in
  proportion to it. The firmware path spends time linear in distance, which
  gives a nearly flat speed however far the shot is.
- Speed follows a lognormal profile peaking at 30–45% of the movement, with a
  deceleration tail about twice as long as the acceleration.
- The ballistic phase stops around a tenth short on purpose. The next frame's
  detection turns the remainder into a visually guided correction.
- The path bows a percent or two off the straight line, biased by the
  direction of travel, and comes back onto it exactly at the end.
- Noise scales with the instantaneous speed and vanishes at both ends, so it
  perturbs the path without ever moving the endpoint.
- Sub-count remainders are carried between reports and between flicks instead
  of being truncated, which is what lets deltas below one count move at all.

### From a detection to a flick

Every number below can be worked out by hand, which is the quickest way to
predict how a setting will feel before trying it.

1. **Engagement.** A flick is planned only when `min_zone < dist ≤ FOV`, both
   in frame pixels. `min_zone` is roughly half the target box, times
   `SCALE_MIN_ZONE1`. With ESP button 2 held it is `aim_smooth`'s zone — half
   the box *width*, or half its larger side when the crosshair is level with
   a head box — times `SCALE_MIN_ZONE2`.

2. **Size, in mouse counts.**

   ```text
   counts = dist × frame_to_screen × 96 / GAME_SENS / MOUSE_DPI
   ```

   `FOV` therefore caps every flick. At `MOUSE_DPI=1000`, `GAME_SENS=1.0`, a
   35 px FOV means no flick is ever larger than 3.4 counts.

3. **Ballistic gain.** The flick commits to `MOVE_SMOOTH_GAIN` of that, ±4.5%.
   A share `MOVE_SMOOTH_OVERSHOOT_P` of flicks overshoot to about 106% instead.
   What is left over is the next frame's job.

4. **Duration (Fitts).** In screen pixels:

   ```text
   W  = 2 × min_zone
   MT = FITTS_A_MS + FITTS_B_MS × log2(2 × dist / W + 1)
   ```

   then ±12% lognormal jitter, then clamped to `[MIN_MS, MAX_MS]`.

5. **Reports.** `MT × POLL_HZ / 1000` of them (at least 2, at most 512), one
   every `1 / POLL_HZ`. Only reports that actually move are written to the
   serial port, so **commands on the wire ≈ counts, not reports**.

6. **Blocking.** The aim thread is busy for the whole of `MT`. That — not the
   serial link — is the delay you feel.

A rule of thumb for how fast each path can follow a moving target:

```text
smooth  ≈ 0.9 × FOV / MT   px/s
bezier  ≈ FOV × fps        px/s
```

**Worked example** — `FOV=35`, `MOUSE_DPI=1000`, `GAME_SENS=1.0`,
`MAKCU_BAUD=4000000`, a target whose `min_zone` is 21 px:

| Step | Value |
| --- | --- |
| counts | 35 × 96 / 1000 = **3.36** |
| W, index of difficulty | 42 px, log2(70 / 42 + 1) = 1.42 bits |
| MT | 35 + 55 × 1.42 ≈ **113 ms** |
| reports at 1000 Hz | 113, of which about **3** are written — 60 bytes, 0.13% of the link |
| tracking | 0.9 × 35 / 0.113 ≈ **280 px/s**, against 2100 px/s for bezier at 60 fps |

That last line is the one that matters at a tight FOV: the default timing is
calibrated for full flicks, and it makes corrections ten times slower than
bezier. The [tight-FOV preset](#presets) fixes most of that.

### Settings

All optional. Every value here is read once at startup.

#### Timing — how fast it feels

| Variable | Default | Effect |
| --- | --- | --- |
| `MOVE_SMOOTH_FITTS_A_MS` | `35` | Fixed cost of every flick. **Dominates at a small FOV**, where the index of difficulty is only 1–2.5 bits. Lower it first when aim feels slow. |
| `MOVE_SMOOTH_FITTS_B_MS` | `55` | Cost per bit of difficulty. Dominates on long flicks. |
| `MOVE_SMOOTH_MIN_MS` | `45` | Floor on `MT`. Lower it together with `FITTS_A_MS`, or it binds and the change does nothing. |
| `MOVE_SMOOTH_MAX_MS` | `320` | Ceiling on `MT`, for the longest flicks. |
| `MOVE_SMOOTH_POLL_HZ` | `0` | Report cadence. `0` picks 1000 Hz at `MAKCU_BAUD` ≥ 2M and 250 Hz below. **Does not change `MT`**; see [Choosing POLL_HZ](#choosing-poll_hz). |

#### Accuracy

| Variable | Default | Effect |
| --- | --- | --- |
| `MOVE_SMOOTH_GAIN` | `0.90` | Share of the distance one flick covers. Toward `1.0`: lands closer, fewer corrections, faster tracking, less human. Much below `0.8`, corrections pile up. |
| `MOVE_SMOOTH_OVERSHOOT_P` | `0.15` | Share of flicks that overshoot and reverse on the correction. `0` never overshoots; people sit around `0.10`–`0.20`. |

#### Shape — how the path looks

| Variable | Default | Effect |
| --- | --- | --- |
| `MOVE_SMOOTH_PEAK_MIN` / `_PEAK_MAX` | `0.30` / `0.45` | Where along the flick speed peaks. Clamped to `0.20`–`0.60`. Near `0.5` it turns into the symmetric bell that no hand produces. |
| `MOVE_SMOOTH_BOW` | `0.03` | Largest sideways deviation, as a fraction of the flick. `0` is a straight line; people sit around `0.01`–`0.05`. |
| `MOVE_SMOOTH_NOISE` | `0.022` | Motor noise proportional to speed. Never moves the endpoint. |
| `MOVE_SMOOTH_TREMOR` | `0.35` | Tremor amplitude, in counts. `0` disables it. Never moves the endpoint. |

At a small FOV the whole flick is a few counts, so `BOW` and `NOISE` mostly
round away below one count. `TREMOR` is an absolute amount rather than a
fraction, so it is the one shape setting that still shows there; set it to
`0` if you see a stray sideways count in the middle of a flick.

#### Reaction latency

| Variable | Default | Effect |
| --- | --- | --- |
| `MOVE_SMOOTH_REACT_MIN_MS` / `_MAX_MS` | `0` / `0` | Wait before the first flick at a fresh target, drawn uniformly from the range. |

Off by default: in trigger mode a command only leaves this process while the
ESP trigger button is held, so the operator's own finger already supplies a
reaction time, and simulating it again would charge for the same latency
twice. Something like `150` / `260` makes sense when running auto aim with no
trigger.

"Fresh" means no flick for 250 ms — and time spent inside `min_zone` counts,
because nothing is flicked there. With the delay on, the hand therefore
re-reacts after every pause, not only on a new target.

### Choosing POLL_HZ

| `MAKCU_BAUD` | Wire time per command | Rate picked by `0` |
| --- | --- | --- |
| `115200` | 1.56 ms | 250 Hz |
| `2000000` | 90 µs | 1000 Hz |
| `4000000` | 45 µs | 1000 Hz |

`POLL_HZ` decides how finely the speed profile is sampled, not how long the
flick lasts. What matters is how many counts a flick carries against how many
slots it gets, so work out the largest flick first:

```text
max counts = FOV × frame_to_screen × 96 / GAME_SENS / MOUSE_DPI
```

- **Up to ~4 counts:** `250` is free — same timing on the wire as 1000 Hz, four
  times fewer thread wake-ups. At 2M and 4M the automatic choice is 1000 Hz, so
  this has to be set by hand.
- **More than that:** leave it at `0`. At 250 Hz a short flick has so few slots
  that counts pile up into a single report. With `GAME_SENS=0.38`,
  `MOUSE_DPI=1000` and `FOV=35` (8.8 counts), a 53 ms flick gets 13 slots at
  250 Hz and its biggest report jumps 8.1 px; at 1000 Hz it gets 53 slots and
  no report exceeds one count (4.2 px).
- **Never above 1000.** No real mouse reports faster, so a higher rate is a
  signature in itself.
- **Keep `POLL_HZ × MAX_MS` ≤ 512 000.** A flick holds at most 512 reports;
  past that the longest flicks get cut short.

### Presets

| Variable | Tight FOV (≤ ~50 px) | Default (wide FOV) |
| --- | --- | --- |
| `MOVE_SMOOTH_POLL_HZ` | `250` if flicks are ≤ ~4 counts, else `0` | `0` |
| `MOVE_SMOOTH_FITTS_A_MS` | `14` | `35` |
| `MOVE_SMOOTH_FITTS_B_MS` | `22` | `55` |
| `MOVE_SMOOTH_MIN_MS` | `22` | `45` |

Everything else stays at its default in both.

> **Mind the capture latency before using the tight preset.** A frame shows the
> game as it was one capture latency ago (NDI or the capture card, then
> decoding). When a flick is shorter than that, the frame read right after it
> does not show the flick yet, the same error is commanded a second time, and
> the aim overshoots and swings back. In simulation, a 30 px shot at a static
> target took 1 flick with either preset at 16 ms of latency, but at 50 ms the
> tight preset needed 9.4 flicks with 3.7 reversals and a 25 px overshoot,
> while the default took 1.5. Until you have
> [measured your latency](#measuring-the-capture-latency), start from the
> default preset.

### Measuring the capture latency

`mouse_test latency` measures how long a mouse move takes to show up in the
frames the aimbot receives — MAKCU, the game, rendering, capture and transport
together. It opens the same `SOURCE_STREAM` the aimbot does.

1. Stop the aimbot; the serial port can only have one owner.
2. In game, stand still facing a detailed, static surface — a textured wall,
   not the sky — with the game focused and not in a menu. Do not touch the
   mouse while it runs.
3. Run it (rebuild the image first if it predates this command):

   ```shell
   docker compose stop aimbot
   docker compose run --rm aimbot mouse_test latency          # 30 counts, 20 samples
   docker compose run --rm aimbot mouse_test latency 60 40    # larger turn, more samples
   ```

It first measures how much the view changes on its own for two seconds, then
for each sample waits for the view to settle, turns it `counts` sideways with a
single `km.move`, and times the first frame whose centre differs by more than
that noise. The view is turned back afterwards. Output ends with:

```text
[latency] 20 samples, 0 missed, a frame every 16.7 ms
[latency] min 41 ms   median 49 ms   p90 56 ms   max 58 ms
```

Samples spread over one frame period: the **min** is close to the pipeline
delay itself, the **median** about half a frame above it. Use the **p90** when
picking a preset — it is how long after a flick ends the frames reliably show
it. If it warns that moves only just cleared the threshold, face a more
textured surface or pass a larger count.

Median `MT` for a body-sized target (`W` = 64 px):

| Distance | Tight FOV | Default |
| --- | --- | --- |
| 35 px | 37 ms | 94 ms |
| 60 px | 48 ms | 119 ms |
| 180 px | 74 ms | 185 ms |
| 600 px | 109 ms | 272 ms |

Measured at `FOV=35`, `MOUSE_DPI=1000`, `GAME_SENS=1.0`, 4M baud, over the
range of `min_zone` a target actually produces:

| | Tight FOV | Default |
| --- | --- | --- |
| `MT` | 39–66 ms | 98–166 ms |
| Reports scheduled | 10–17 | 98–166 |
| Commands written | 1–3 | 1–3.5 |
| Tracking | 240–700 px/s | 95–280 px/s |

Be clear about what the tight-FOV preset gives up. A person takes roughly
80–150 ms over a 10–35 px correction, so 39–66 ms is *faster than a hand*. The
speed profile, the undershoot, the curvature and the sub-count carry are all
still there; the duration just no longer follows human Fitts constants. That is
a reasonable trade at a tight FOV, where the alternative is falling behind
every strafing target.

### Tuning procedure

1. **Get `MOUSE_DPI` and `GAME_SENS` right first.** Every flick is sized in
   counts derived from them, so a wrong value makes each one the wrong size and
   no smooth setting can compensate. Both must be above zero; startup panics
   otherwise.
2. **Pick a preset from your FOV.**
3. **Check the device keeps up.** Run `mouse_test`, type `s` at the `dx`
   prompt to switch paths, and compare the planned duration with the measured
   one it prints. Measured consistently longer than planned means the host or
   the MAKCU cannot hold the cadence — lower `MOVE_SMOOTH_POLL_HZ`.
4. **A/B in game.** Run with `MOVE_SMOOTH=false` and compare a normal shot
   (bezier) with the same shot while holding ESP button 2 (always smooth) —
   bearing in mind that button 2 also changes the target to the Horizon-style
   one. For a like-for-like comparison, restart with `MOVE_SMOOTH=true`.
5. **Adjust one symptom at a time:**

   | Symptom | Change |
   | --- | --- |
   | Slow, falls behind moving targets | Lower `FITTS_A_MS`, with `MIN_MS` alongside; then `FITTS_B_MS` |
   | Lands short, needs several nudges | Raise `GAIN` toward `1.0` |
   | Wobbles, reverses too often | Lower `OVERSHOOT_P` |
   | Stray sideways counts | `TREMOR=0` |
   | Too uniform, looks mechanical | Raise `FITTS_A_MS` / `FITTS_B_MS`, keep `PEAK_*` off 0.5 |

### Fixed in code

These are not exposed as environment variables. Change
`SmoothConfig::default()` in `src/config.rs` if you need to.

| Field | Default | Effect |
| --- | --- | --- |
| `mt_jitter` | `0.12` | Spread of the movement time between flicks |
| `gain_sd` | `0.045` | Spread of the ballistic gain |
| `curve_bias` | `0.30` | How strongly the bow direction follows the direction of travel; `0` is a coin flip |
| `kappa_min` / `kappa_max` | `0.85` / `1.25` | Where along the path the bow peaks |
| `gap_ms` | `250` | Silence that counts as a fresh target |
| `max_counts` | `127` | Ceiling on one report; the excess is deferred, never dropped |

## Environment reference

`Config::new()` reads these at startup. The twelve marked **required** have no
default and the process panics without them — including the TensorRT and
OpenVINO ones, which are read regardless of which provider is selected, so set
`TRT_*` to any valid value even on an OpenVINO build.

### Core

| Variable | Default | Meaning |
| --- | --- | --- |
| `SOURCE_STREAM` | **required** | Frame source; see [Frame sources](#frame-sources) |
| `EVENT_LISTENER_PORT` | `10000` | Port for the event listener |
| `SCREEN_WIDTH` / `SCREEN_HEIGHT` | `1920` / `1080` | Full frame size; the crosshair is the centre of this |
| `REGION_LEFT` / `REGION_TOP` | `0` / `0` | Detection region origin |
| `REGION_WIDTH` / `REGION_HEIGHT` | `0` / `0` | Detection region size. Cropping happens only when this differs from the screen size |
| `SCALE_MIN_ZONE1` | `0.5` | Dead-zone scale for the normal aim path |
| `SCALE_MIN_ZONE2` | `0.8` | Dead-zone scale while ESP button 2 aims through `aim_smooth` |
| `INTRA_THREADS` | `1` | ONNX Runtime intra-op threads |
| `RUST_LOG` | `info` | e.g. `info,aimbot=debug` for per-stage timings |

### Model

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL_PATH` | **required** | Model file. A name starting with `v_light` selects the light pipeline |
| `MODEL_INPUT_SIZE` | **required** | Must match the model input (192 for `v_light_192`) |
| `MODEL_CONF_BODY` | **required** | Confidence threshold for class 0 |
| `MODEL_CONF_HEAD` | **required** | Confidence threshold for class 1 |
| `MODEL_IOU` | **required** | NMS IoU, both pipelines |
| `MODEL_PROVIDER` | `cpu` | `cpu`, `openvino`, `tensorrt`/`trt`, `rocm`, `migraphx`/`mrx` |
| `BUILD_HEAD_IOU` | unset | When set, synthesises a head box from each unmatched body box at this IoU |

### Providers

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENVINO_CACHE_DIR` | **required** | Compiled-blob cache directory |
| `OPENVINO_DEVICE_TYPE` | `CPU` | `CPU`, `GPU`, `NPU`, `GPU.0`, ... |
| `TRT_CACHE_DIR` | **required** | TensorRT engine cache |
| `TRT_MIN_SHAPES` / `TRT_OPT_SHAPES` / `TRT_MAX_SHAPES` | **required** | e.g. `images:1x3x256x256` |
| `TRT_FP16` | unset | Build the engine in FP16 |
| `TRT_MAX_PARTITION_ITERATIONS` | `10` | |
| `TRT_BUILDER_OPTIMIZATION_LEVEL` | `3` | `0`–`5`; below 3 builds faster but runs slower |
| `TRT_DLA_ENABLE` / `TRT_DLA_CORE` | `false` / `0` | |
| `TRT_AUXILIARY_STREAMS` | `-1` | |
| `GPU_ID` | `0` | Device index |
| `GPU_MEM_LIMIT` | `1073741824` | Workspace / memory limit in bytes |

### NDI

| Variable | Default | Meaning |
| --- | --- | --- |
| `NDI_SOURCE_NAME` | unset | Prefer the source whose name matches |
| `NDI_TIMEOUT` | `1000` | Receive timeout in ms |

### Output

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAKCU_PORT` | **required** | Serial port of the MAKCU device |
| `MAKCU_BAUD` | `115200` | |
| `MAKCU_LISTEN` | `false` | Watch mouse buttons to toggle trigger / auto-aim |
| `MOUSE_DPI` | `1000` | |
| `GAME_SENS` | `1.0` | |
| `ESP_PORT` | unset | Serial port of the ESP button board |

### Human-flick movement

What each one does, how to compute its effect and presets by FOV are under
[Mouse movement](#settings).

| Variable | Default | Meaning |
| --- | --- | --- |
| `MOVE_SMOOTH` | `false` | Use `move_smooth` instead of `move_bezier` in the ordinary aim modes. ESP button 2 always uses it |
| `MOVE_SMOOTH_POLL_HZ` | `0` | Report cadence; `0` derives it from `MAKCU_BAUD` |
| `MOVE_SMOOTH_FITTS_A_MS` / `_B_MS` | `35` / `55` | Fitts intercept and slope, ms and ms per bit |
| `MOVE_SMOOTH_MIN_MS` / `_MAX_MS` | `45` / `320` | Movement-time clamp |
| `MOVE_SMOOTH_GAIN` | `0.90` | Fraction of the distance the ballistic phase commits to |
| `MOVE_SMOOTH_OVERSHOOT_P` | `0.15` | Share of flicks that overshoot instead of falling short |
| `MOVE_SMOOTH_PEAK_MIN` / `_PEAK_MAX` | `0.30` / `0.45` | Where peak speed sits, as a fraction of the movement |
| `MOVE_SMOOTH_BOW` | `0.03` | Maximum perpendicular deviation, as a fraction of the amplitude |
| `MOVE_SMOOTH_NOISE` | `0.022` | Signal-dependent motor noise coefficient |
| `MOVE_SMOOTH_TREMOR` | `0.35` | Physiological tremor amplitude in counts; `0` disables it |
| `MOVE_SMOOTH_REACT_MIN_MS` / `_MAX_MS` | `0` / `0` | Reaction latency on a fresh target; off by default |

Capture-card variables are in
[Capture from a capture card](#capture-from-a-capture-card).

## Cargo features

| Feature | Effect |
| --- | --- |
| `ndi4` (default) | NDI 4 via the `ndi` crate |
| `ndi6` | NDI 6 via `grafton-ndi` |
| `disable-mouse` | Build without mouse output; detection only |
| `debug` | Write annotated frames to `assets/debug` |
| `save-bbox` | `debug` plus YOLO-format label files |
