# AimBot

An AI-powered aimbot written in Rust.

## Layout

Two machines:

- **PC1** — the game machine. Sends video to PC2 over NDI or UDP, and keyboard
  and mouse presses to PC2's event listener. The keylogger that does the latter
  lives in [aimbot-keylogger](https://github.com/dat58/aimbot-keylogger).
- **PC2** — **Ubuntu 24.04**, runs this project in Docker. Takes frames in,
  runs detection, and drives the mouse over a serial link back to PC1 — a
  MAKCU box by default, or a Medius one; see [Mouse devices](#mouse-devices).

A capture card attached directly to PC2 replaces the NDI/UDP hop entirely; see
[Capture from a capture card](#capture-from-a-capture-card).

## Configure

Needs **Rust 1.90** — see [Medius](#medius) for why.

```
cp config.example.toml config.toml
cargo run -r --bin aimbot            # reads ./config.toml
cargo run -r --bin aimbot -- --config other.toml
```

Every setting lives in that file; see
[Configuration reference](#configuration-reference). The Docker image reads
`config.docker.toml`, which `docker-compose.yml` mounts at `/app/config.toml`.

## Frame sources

`source_stream` picks the source by prefix:

| `source_stream` | Source |
| --- | --- |
| `ndi://172.24.36.133` | NDI. Comma-separate several addresses to add more discovery targets. |
| `elgato://` or `v4l2://` | Capture card on `capture_device` |
| `elgato:///dev/video1` | Capture card on the named node |
| anything else | Passed to FFmpeg as a URL (UDP/RTSP/...) |

NDI is behind cargo features: `ndi4` (default) or `ndi6`. The shipped
Dockerfiles build `ndi4`; swap the feature in the builder stage for `ndi6`.

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
`model_input_size` and the runtime output shape, so the sibling `v_light_256`
works the same way with nothing but a different `model_path` and
`model_input_size`:

- **Preprocess** (`preprocess_light`) — nearest-neighbour stretch of the region
  into `model_input_size` square with **no letterbox and no padding**, matching
  capfkaplus: each axis gets its own scale. Then BGR to RGB, NCHW, `1/255`.
  Nearest neighbour rather than bilinear because that is how the graph was
  exported.
- **Postprocess** (`decode_light`) — the graph emits raw candidates as
  `[1, 4 + classes, N]`: `cx, cy, w, h` in model-input pixels followed by one
  score per class, with no objectness channel and no NMS inside the graph.
  Boxes are multiplied by the inverse per-axis factor
  (`sx = region_width / model_input_size`,
  `sy = region_height / model_input_size`), then the existing per-class
  `non_max_suppression` runs on `model_iou`, exactly as on the two-class path.

  Nothing is baked in: the input side comes from `model_input_size`, and the
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
model_path=/app/assets/v_light_192_fp16_onnx.onnx
model_input_size=192
```

`model_input_size` must match what the graph declares — 192 for `v_light_192`,
256 for `v_light_256` — otherwise ONNX Runtime rejects the input tensor shape.
`model_conf_body`, `model_conf_head` and `model_iou` apply unchanged. Despite
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

Note that `Dockerfile` still uses the original distroless final stage, which
copies only those 6 libraries and therefore does not carry OpenCV. Only
`Dockerfile.openvino` has the `ldd` harvest.

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
model_provider=openvino
openvino_device_type=CPU     # or GPU / NPU / GPU.0 ...
openvino_cache_dir=/app/assets
intra_threads=1
```

`openvino_device_type = "GPU"` needs `/dev/dri` passed into the container; the GPU
and NPU plugins are already in the image.

Median of 4 runs on a Xeon 8358P, `intra_threads = 1`, 1920x1080 input, provider
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

Set `source_stream = "elgato://"` (or `v4l2://`) to read frames straight from the
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

| Key | Default | Meaning |
| --- | --- | --- |
| `capture_device` | `/dev/video0` | V4L2 node |
| `capture_width` | `1920` | Requested capture width |
| `capture_height` | `1080` | Requested capture height |
| `capture_fps` | `60` | Requested frame rate; `0` leaves the device default |
| `capture_fourcc` | (negotiated) | Pin a pixel format: `NV12`, `YU12`, `YUYV`, `UYVY`, `BGR3`, `RGB3`, `AR24`, `MJPG` |
| `capture_buffers` | `3` | MMAP ring size; 2 is the minimum the kernel accepts |
| `capture_drop_stale` | `true` | Drain the driver queue each grab and keep only the newest frame |
| `capture_timeout_ms` | `1000` | Grab timeout before the stream is treated as lost |
| `capture_output` | `bgr` | `bgr` converts for the detection pipeline; `raw` hands back the untouched device bytes |

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

With `capture_output = "bgr"` there is exactly one pass over the pixels
(`cvtColor`). `capture_output = "raw"` skips even that and is the lowest-latency
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
      - "/dev/ttyACM0:/dev/ttyACM0"   # [mouse] port
      - "/dev/video0:/dev/video0"     # capture_device, capture-card source only
      - "/dev/dri:/dev/dri"           # openvino_device_type=GPU only
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
`PUT /stream/event/{id}`, bound to `0.0.0.0:$event_listener_port`.

## Mouse devices

The device and the movement are two independent choices. `[mouse].device` picks
what holds the mouse; each profile's `mover` picks what shape the movement has.
Adding either is one file and one match arm — `src/mouse/` for a device,
`src/mouse/mover/` for a movement — and neither touches the other.

| `device` | Link | Notes |
| --- | --- | --- |
| `makcu` (default) | Serial, `baud` negotiated at connect | Reads the physical mouse's buttons back, walks a bezier in firmware, and answers `km.move_now` on V4.028+ firmware |
| `medius` | Serial, fixed 6 Mbaud, framed | No button read-back, no firmware bezier; renders injected motion through its own model instead |

```toml
[mouse]
device = "makcu"
port = "/dev/ttyACM0"
baud = 4000000
listen = false
dpi = 1000.0
```

| Key | Default | Meaning |
| --- | --- | --- |
| `device` | `makcu` | `makcu` or `medius` |
| `port` | **required** | Serial node. An empty string means "open the first box found" — Medius only |
| `baud` | `115200` | MAKCU only; one of `115200`, `2000000`, `4000000` |
| `listen` | `false` | Watch the physical mouse's buttons to toggle trigger / auto-aim. Ignored, with a warning, on a device that cannot read them |
| `dpi` | `1000` | Counts per inch the mouse is actually set to. Not a knob, and **not** part of the pixel-to-counts conversion — its only use is the hand-speed envelope |
| `render` | `despiked` | Medius only: `off`, `stock`, `despiked`, `unsmoothed` |
| `render_full` | `false` | Medius only: render the operator's own motion too instead of relaying it untouched |
| `lock_move` | `false` | Medius only: blank the operator's own motion around each injected move, so the two cannot add up |

The `[mouse]` table has to sit **after every top-level key and before the first
`[[profile]]`** — TOML assigns every key following a header to that table, so a
`[mouse]` placed mid-file would swallow `esp_button_1_profile` and friends.

### Medius

An alternative firmware with its own crate, compiled in unconditionally — set
`device = "medius"` and it is there.

The crate is why this project needs **Rust 1.90**: it uses let-chains, which
only stabilised in 1.88, and it advertises an MSRV of 1.85 that it does not
actually honour. `rust-version` in `Cargo.toml` pins 1.90.0, and both
Dockerfiles `rustup install` it over the base image's 1.86.

What it does not do:

- **Read the physical mouse's buttons.** `listen` has no effect, and the MAKCU
  side-4 / side-5 shortcuts are unavailable. The ESP buttons still work, since
  those arrive on their own board.
- **Walk a bezier in firmware.** `mover = "bezier"` falls back to one straight
  move.
- **Use `baud`.** The link is fixed at 6 Mbaud.

`move_smooth_move_now` maps onto the box's immediate path, so a timed flick is
emitted report by report rather than waiting for native motion to ride.

`lock_move = true` wraps every report in `lock_all(Aim)` / `unlock_all(Aim)`,
which is three frames on the wire instead of one. At
`move_smooth_poll_hz = 1000` that is 3000 frames a second — if the velocity
profile smears with it on, this is the first thing to turn off.

## Mouse movement

A mouse delta is played one of three ways, chosen per profile by `mover`:

| `mover` | What it is |
| --- | --- |
| `smooth` (default) | The host plans a human flick and plays it as a stream of small reports on its own fixed cadence. |
| `bezier` | The whole delta in one command, walked by the device (`km.move(dx,dy,steps,ref_x,ref_y)` on MAKCU). A device with no curve command gets a straight move instead. |
| `direct` | The delta exactly as it arrives, one report, no model at all. Auto-clicks like `smooth` does. |

`direct` keeps no state between frames, which means **no sub-count carry**: a
delta under one count truncates to zero and is lost. At a low counts-per-pixel
(high `game_sens`, low `[mouse].dpi`) small corrections will not move at all.
That is the direct consequence of doing no arithmetic, not a bug — use `smooth`
if you want the remainder carried.

### Which one is used

The profile decides. Which profile is in play depends on the ESP button:

| Situation | Profile | Target chosen by | Movement |
| --- | --- | --- | --- |
| ESP button 2 held | `esp_button_2_profile` | the profile's `aim_mode`, else the runtime mode | the profile's `mover` |
| Everything else — ESP button 1, MAKCU side 4, no-trigger auto aim | `esp_button_1_profile` | same | same |

`mover` defaults to `"smooth"`, so a profile that says nothing gets the
human-flick model. `"bezier"` hands the whole delta to the device and never
auto-clicks; `"direct"` sends it untouched and does auto-click.

Setting `aim_mode = "horizon"` reproduces what used to be hard-wired to ESP
button 2. Horizon sends x straight to the centre of the box, while y is only
nudged toward it a random amount each frame instead of being snapped onto it —
up to 21 px down when the crosshair is above the box, up to 11 px up toward a
head box or 21 px toward a body box when it is below, and −14 to +7 px when it
is already level with a body box. Level with a head box, it aims straight at the
head.

Profiles are read at startup; changing one needs a restart.

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

1. **Engagement.** A flick is planned only when `min_zone < dist ≤ fov`, both
   in frame pixels. `min_zone` is roughly half the target box, times
   `scale_min_zone`. Under `aim_mode = "horizon"` it is half the box *width*,
   or half its larger side when the crosshair is level with a head box, times
   the same `scale_min_zone`.

2. **Size, in mouse counts.** Pixels are a perspective projection of the view
   angle, so the relation is an arctangent rather than a scale factor:

   ```text
   focal  = (frame_width / 2) / tan(game_hfov / 2)
   turn   = atan2(dist, focal)                  # degrees of view rotation
   counts = turn / (game_yaw × game_sens) × aim_scale
   ```

   The pitch uses `atan2(dy, hypot(focal, dx))` rather than `atan2(dy, focal)`,
   because it is measured off the already-yawed ray.

   `dist` is in frame pixels and the focal length comes from `frame_width`,
   which is the space it is measured in and which covers the whole field of
   view.

   **The mouse's DPI is not in this and must not be.** The game never sees it:
   it receives counts and converts them itself. DPI only decides how much
   *hand travel* a count is worth, which is why its one use is the speed
   envelope below.

   The bend is small close in and only matters further out. At a 103° field of
   view and a 1920-wide frame:

   | `dist` | compression against a straight line |
   | --- | --- |
   | 35 px | 0.07% |
   | 50 px | 0.14% |
   | 200 px | 2.2% |
   | 800 px | 23% |

   So inside a tight `fov` the projection is very nearly linear — if aim is
   off at 50 px it is the constant, not the curve.

   ⚠️ **`game_yaw` is a constant this repo cannot verify.** The default comes
   out of a reference snippet, and a wrong value scales every turn by a fixed
   factor. `aim_scale` is the measured correction for exactly that — see
   [Calibrating aim_scale](#calibrating-aim_scale).

### Calibrating `aim_scale`

`aim_scale` multiplies whatever the model produces, last. It is the one
number here that can be established in the game actually being played rather
than looked up, so get it right before touching anything else — no smooth
setting compensates for a turn that is three times too far.

Bisect it. Pick a profile with `mover = "direct"` so nothing else is between
the number and the mouse, stand still, and hold the trigger on a target.
A good starting guess is well under `1.0`: the shipped `game_yaw` is almost
certainly wrong for your game.

| What the aim does | What to do |
| --- | --- |
| Sails well past, often out of `fov` | Halve `aim_scale` |
| Stops clearly short | Double it |
| Overshoots a little | Multiply by about 0.8 |
| Lands, then jitters | It is right; go tune the mover |

Four or five rounds is usually enough. Starting from `1.0` and ending at,
say, `0.3` tells you the real constant is about a third of what the model
assumed — worth writing in the config next to the value.

3. **Ballistic gain.** The flick commits to `move_smooth_gain` of that, ±4.5%.
   A share `move_smooth_overshoot_p` of flicks overshoot to about 106% instead.
   What is left over is the next frame's job.

4. **Duration (Fitts).**

   ```text
   W  = 2 × min_zone
   MT = FITTS_A_MS + FITTS_B_MS × log2(2 × dist / W + 1)
   ```

   then ±12% lognormal jitter, then clamped to `[MIN_MS, MAX_MS]`.

   `dist` and `W` are frame pixels, straight from the detection. Only their
   **ratio** is read, so whichever pixel space they are measured in cancels —
   converting both to screen pixels first would multiply both and change
   nothing. All that matters is that they share a space.

5. **Reports.** `MT × poll_hz / 1000` of them (at least 2, at most 512), one
   every `1 / poll_hz`. Only reports that actually move are written to the
   serial port, so **commands on the wire ≈ counts, not reports**.

6. **Blocking.** The aim thread is busy for the whole of `MT`. That — not the
   serial link — is the delay you feel.

A rule of thumb for how fast each path can follow a moving target:

```text
smooth  ≈ 0.9 × fov / MT   px/s
bezier  ≈ fov × fps        px/s
```

**Worked example** — `fov = 35`, `game_sens = 1.0`, `game_hfov = 103`,
`game_yaw = 0.07`, frames arriving at 1920 wide, `[mouse].baud = 4000000`, a
target whose `min_zone` is 21 px:

| Step | Value |
| --- | --- |
| focal | 960 / tan(51.5°) = **763.6 px** |
| turn | atan2(35, 763.6) = 2.625° |
| counts | 2.625 / (0.07 × 1.0) = **37.5** |
| W, index of difficulty | 42 px, log2(70 / 42 + 1) = 1.42 bits |
| Fitts MT | 35 + 55 × 1.42 ≈ 113 ms |
| MT after jitter and the speed envelope | **54 ms** |
| reports at 1000 Hz | **16**, all of them written at ~2.3 counts each — 288 bytes, 1.3% of the link |
| tracking | 0.9 × 35 / 0.054 ≈ **580 px/s**, against 2100 px/s for bezier at 60 fps |

Every number above is produced by the code, not by hand.

### Settings

All optional. Every value here is read once at startup.

#### Timing — how fast it feels

| Key | Default | Effect |
| --- | --- | --- |
| `move_smooth_fitts_a_ms` | `35` | Fixed cost of every flick. **Dominates at a small fov**, where the index of difficulty is only 1–2.5 bits. Lower it first when aim feels slow. |
| `move_smooth_fitts_b_ms` | `55` | Cost per bit of difficulty. Dominates on long flicks. |
| `move_smooth_min_ms` | `45` | Floor on `MT`. Lower it together with `fitts_a_ms`, or it binds and the change does nothing. |
| `move_smooth_max_ms` | `320` | Ceiling on `MT`, for the longest flicks. |
| `move_smooth_poll_hz` | `0` | Report cadence. `0` picks 1000 Hz at `[mouse].baud` ≥ 2M and 250 Hz below. **Does not change `MT`**; see [Choosing poll_hz](#choosing-poll_hz). |

#### Report density — why a flick can look like it is not moving

Counts are integers, so a report carrying less than half of one rounds to zero
and moves nothing. Fitts sizes a flick in *time*, and at 1 kHz a 190 ms flick is
190 reports — right for the hundreds of counts a full-screen flick carries, and
badly wrong for an in-region correction, which is a handful.

| Key | Default | Effect |
| --- | --- | --- |
| `move_smooth_counts_per_report` | `2.0` | Mean counts one report carries, which caps how many reports a flick is worth planning. Below `1` most reports go empty again; raise it for fewer, larger reports. **Does not change the duration** — the reports spread out instead. |
| `move_smooth_min_speed_ips` | `0.6` | Slowest the hand is allowed to travel, inches per second. Below a few hundred counts **this is what sets the duration**, not Fitts. |
| `move_smooth_max_speed_ips` | `40` | Fastest. Rarely binds; it stops a huge flick becoming a teleport. |

Fitts's index of difficulty is a *ratio* of distance to width, so it is the same
for a 5-count flick and a 5000-count one and says nothing about how far the hand
moves. `counts / dpi` does — it is inches. **This is the only place `[mouse].dpi`
is used.** It does not enter the pixel-to-counts conversion, because the game
never sees it; it only answers "how far did the hand actually travel", which is
what makes a speed envelope in inches meaningful.

The envelope is expressed in inches precisely so it holds at any `[mouse].dpi`
and `game_sens` without retuning, since neither is a knob you get to pick.

Fitts describes *aimed* reaches, where landing accurately is the cost. A
correction far too small for that just travels at the hand's comfortable speed,
which makes its duration proportional to its length — Fitts's law is known to
flatten out at very low indices of difficulty for the same reason.

Computed at `game_hfov = 103`, `game_yaw = 0.07`, frames 1920 wide,
`min_zone = 10 px`, all other settings at their defaults:

| Reach | Counts | Reports | Duration | at `game_sens = 1.0` |
| --- | --- | --- | --- | --- |
| 20 px | 56.4 | 24 | 81 ms | 21.4 counts, 9 reports, 31 ms |
| 60 px | 168.9 | 73 | 188 ms | 64.2 counts, 28 reports, 93 ms |
| 135 px | 376.9 | 163 | 246 ms | 143.2 counts, 62 reports, 207 ms |

Fitts asks for 122 / 189 / 247 ms at these reaches. The 60 px and 135 px rows
land on that almost exactly; the 20 px row is still cut short by
`min_speed_ips` (56 counts at 1000 dpi is 0.056 in, which at 0.6 in/s cannot
take longer than 94 ms). So **the envelope still sets the duration on the
shortest corrections and Fitts takes over from roughly 60 px up** — which is
the division of labour it was designed for.

Under the old pixel-proportional conversion the same corrections were worth a
tenth of the counts, so the envelope capped *everything* and flicks came out
about ten times too short.

#### Accuracy

| Key | Default | Effect |
| --- | --- | --- |
| `move_smooth_gain` | `0.90` | Share of the distance one flick covers. Toward `1.0`: lands closer, fewer corrections, faster tracking, less human. Much below `0.8`, corrections pile up. |
| `move_smooth_overshoot_p` | `0.15` | Share of flicks that overshoot and reverse on the correction. `0` never overshoots; people sit around `0.10`–`0.20`. |

#### Shape — how the path looks

| Key | Default | Effect |
| --- | --- | --- |
| `move_smooth_peak_min` / `_PEAK_MAX` | `0.30` / `0.45` | Where along the flick speed peaks. Clamped to `0.20`–`0.60`. Near `0.5` it turns into the symmetric bell that no hand produces. |
| `move_smooth_bow` | `0.03` | Largest sideways deviation, as a fraction of the flick. `0` is a straight line; people sit around `0.01`–`0.05`. |
| `move_smooth_noise` | `0.022` | Motor noise proportional to speed. Never moves the endpoint. |
| `move_smooth_tremor` | `0.35` | Tremor amplitude, in counts. `0` disables it. Never moves the endpoint. |

At a small fov the whole flick is a few counts, so `bow` and `noise` mostly
round away below one count. `tremor` is an absolute amount rather than a
fraction, so it is the one shape setting that still shows there; set it to
`0` if you see a stray sideways count in the middle of a flick.

#### Reaction latency

| Key | Default | Effect |
| --- | --- | --- |
| `move_smooth_react_min_ms` / `_MAX_MS` | `0` / `0` | Wait before the first flick at a fresh target, drawn uniformly from the range. |

Off by default: in trigger mode a command only leaves this process while the
ESP trigger button is held, so the operator's own finger already supplies a
reaction time, and simulating it again would charge for the same latency
twice. Something like `150` / `260` makes sense when running auto aim with no
trigger.

"Fresh" means no flick for 250 ms — and time spent inside `min_zone` counts,
because nothing is flicked there. With the delay on, the hand therefore
re-reacts after every pause, not only on a new target.

### Auto click

Fires a left click once a movement has been played. `smooth` and `direct` both
do this and share the same policy; `mover = "bezier"` never auto-clicks. The
click runs on the mouse worker thread, so the aim loop never blocks on it.

Each profile keeps its own click timestamp, so the rate limit below is counted
per profile rather than across the whole session. A `direct` profile reads only
these four keys out of the `move_smooth_*` set and ignores the rest.

| Key | Default | Effect |
| --- | --- | --- |
| `move_smooth_auto_click` | `false` | Turn it on. |
| `move_smooth_auto_click_lower_ms` / `_UPPER_MS` | `100` / `130` | Wait between the flick ending and the click, drawn uniformly. Stands in for deciding to shoot. |
| `move_smooth_auto_click_rate_limit_ms` | `0` | Smallest gap between two clicks, ms. `0` is no limit. |
| `move_smooth_auto_click_miss_p` | `0.0` | Chance in `[0, 1]` the shot is fired without letting the aim settle — a deliberate miss. |

**Rate limit.** Without one, a held trigger clicks once per flick and a flick is
planned once per frame, so the rate is whatever the delay above works out to:

| `rate_limit` | Clicks/s | Gap | Cycles that skip the click |
| --- | --- | --- | --- |
| `0` | 5.80 | 172 ms | 0% |
| `250` | 3.90 | 256 ms | 87% |
| `500` | 2.00 | 500 ms | 96% |
| `1000` | 1.00 | 1000 ms | 99% |

The gap is measured click to click, and the delay before the click counts toward
it. A refused click is skipped outright rather than deferred, so the worker goes
straight back to tracking instead of sleeping out a delay it will not use —
which is why a limit makes the aim *smoother*, not slower. Pick by weapon: `0`
for automatic, `150`–`250` for semi-auto, `800`–`1500` for a sniper.

**Miss probability.** Both outcomes wait out the same
`lower_ms`–`upper_ms`; what separates them is whether the mouse keeps correcting
during that wait:

| `miss_p` outcome | During the wait | Result |
| --- | --- | --- |
| miss | holds still | the crosshair keeps the ballistic error, so the shot misses |
| hit | plays the corrections still arriving | the crosshair is on target when it fires |

The error it keeps is what `move_smooth_gain` left behind, measured over 4000
flicks at `game_sens = 0.38` against a ~20 px target:

| Reach | Residual p50 | Residual p90 | Overshot |
| --- | --- | --- | --- |
| 20 px | 0.21 px | 4.17 px | 3.5% |
| 60 px | 4.58 px | 8.54 px | 12.2% |
| 135 px | 12.29 px | 20.21 px | 13.4% |

So a miss only really misses on a long flick; a 20 px correction lands on target
either way. Worth setting above zero regardless — never missing is itself a
signature. Not to be confused with `move_smooth_overshoot_p`, which decides
whether a *movement* goes past its target rather than whether a *shot* does.

### Choosing poll_hz

| `[mouse].baud` | Wire time per command | Rate picked by `0` |
| --- | --- | --- |
| `115200` | 1.56 ms | 250 Hz |
| `2000000` | 90 µs | 1000 Hz |
| `4000000` | 45 µs | 1000 Hz |

`poll_hz` decides how finely the speed profile is sampled, not how long the
flick lasts. What matters is how many counts a flick carries against how many
slots it gets, so work out the largest flick first:

```text
max counts = fov × (1920 / screen_width) × 1.07437623 × game_sens^-0.9936827126
```

At `screen_width = 1920`:

| `fov` | at `game_sens = 1.0` | at `game_sens = 0.38` |
| --- | --- | --- |
| 35 px | 38 counts | 98 counts |
| 50 px | 54 counts | 141 counts |
| 120 px | 129 counts | 337 counts |
| 192 px | 206 counts | 540 counts |

Divide by `screen_width / 1920` for a wider screen.

- **Up to ~4 counts:** `250` is free — same timing on the wire as 1000 Hz, four
  times fewer thread wake-ups. At 2M and 4M the automatic choice is 1000 Hz, so
  this has to be set by hand.
- **More than that:** leave it at `0`. At 250 Hz a short flick has so few slots
  that counts pile up into a single report. With `game_sens = 0.38`,
  `[mouse].dpi = 1000` and `fov = 35` (8.8 counts), a 53 ms flick gets 13 slots at
  250 Hz and its biggest report jumps 8.1 px; at 1000 Hz it gets 53 slots and
  no report exceeds one count (4.2 px).
- **Never above 1000.** No real mouse reports faster, so a higher rate is a
  signature in itself.
- **Keep `poll_hz × MAX_MS` ≤ 512 000.** A flick holds at most 512 reports;
  past that the longest flicks get cut short.

### Presets

| Variable | Tight fov (≤ ~50 px) | Default (wide fov) |
| --- | --- | --- |
| `move_smooth_poll_hz` | `250` if flicks are ≤ ~4 counts, else `0` | `0` |
| `move_smooth_fitts_a_ms` | `14` | `35` |
| `move_smooth_fitts_b_ms` | `22` | `55` |
| `move_smooth_min_ms` | `22` | `45` |

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
together. It opens the same `source_stream` the aimbot does.

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

| Distance | Tight fov | Default |
| --- | --- | --- |
| 35 px | 37 ms | 94 ms |
| 60 px | 48 ms | 119 ms |
| 180 px | 74 ms | 185 ms |
| 600 px | 109 ms | 272 ms |

Measured at `fov = 35`, `[mouse].dpi = 1000`, `game_sens = 1.0`, 4M baud, over
the range of `min_zone` a target actually produces:

| | Tight fov | Default |
| --- | --- | --- |
| `MT` | 38–54 ms | 54 ms |
| Reports scheduled | 16 | 16 |
| Tracking | 583–823 px/s | 583 px/s |

The default row is flat because at this fov and sensitivity
`move_smooth_min_speed_ips = 0.6` caps `MT` at 62 ms whatever Fitts asks, so
widening `W` changes nothing. That also means **the tight preset buys much less
than it used to here** — 38 ms against 54, not 39 against 98. Before reaching
for it, try lowering `move_smooth_min_speed_ips` instead; at a tight fov it,
not Fitts, is what you are actually fighting.

Be clear about what the tight-fov preset gives up. A person takes roughly
80–150 ms over a 10–35 px correction, so 39–66 ms is *faster than a hand*. The
speed profile, the undershoot, the curvature and the sub-count carry are all
still there; the duration just no longer follows human Fitts constants. That is
a reasonable trade at a tight fov, where the alternative is falling behind
every strafing target.

### Tuning procedure

1. **Calibrate `aim_scale` first**, with `game_sens` and `game_hfov` set to
   what the game says; see
   [Calibrating aim_scale](#calibrating-aim_scale). Every flick is sized in
   counts derived from these, so a wrong value makes each one the wrong size
   and no smooth setting can compensate. `[mouse].dpi` matters too, but only
   for the speed envelope — it does not change how far a flick goes. All must
   be above zero; startup fails otherwise.
2. **Pick a preset from your fov.**
3. **Check the device keeps up.** Run `mouse_test`, type `s` at the `dx`
   prompt to switch paths, and compare the planned duration with the measured
   one it prints. Measured consistently longer than planned means the host or
   the MAKCU cannot hold the cadence — lower `move_smooth_poll_hz`.
4. **A/B in game.** Give one profile `mover = "bezier"` (or `"direct"`) and the
   other `mover = "smooth"`, keeping every other key identical, and compare the
   same shot under each ESP button. Leaving `aim_mode` unset in both keeps the target
   choice identical, which is what makes the comparison about the movement.
5. **Adjust one symptom at a time:**

   | Symptom | Change |
   | --- | --- |
   | Slow, falls behind moving targets | Lower `fitts_a_ms`, with `min_ms` alongside; then `fitts_b_ms` |
   | Lands short, needs several nudges | Raise `gain` toward `1.0` |
   | Wobbles, reverses too often | Lower `overshoot_p` |
   | Stray sideways counts | `move_smooth_tremor = 0` |
   | Too uniform, looks mechanical | Raise `fitts_a_ms` / `fitts_b_ms`, keep `move_smooth_peak_*` off 0.5 |

### Rarely touched

Settable in a `[[profile]]` like the rest, but left alone unless you have a
reason. The defaults live in `SmoothConfig::default()` in `src/config.rs`.

| Key | Default | Effect |
| --- | --- | --- |
| `move_smooth_mt_jitter` | `0.12` | Spread of the movement time between flicks |
| `move_smooth_gain_sd` | `0.045` | Spread of the ballistic gain |
| `move_smooth_curve_bias` | `0.30` | How strongly the bow direction follows the direction of travel; `0` is a coin flip |
| `move_smooth_kappa_min` / `_kappa_max` | `0.85` / `1.25` | Where along the path the bow peaks |
| `move_smooth_gap_ms` | `250` | Silence that counts as a fresh target |
| `move_smooth_max_counts` | `127` | Ceiling on one report; the excess is deferred, never dropped |

## Configuration reference

Everything is read from a TOML file at startup — `config.toml` next to the
binary, or whatever `--config <path>` names. `config.example.toml` is a working
starting point; copy it to `config.toml` and edit. The only environment variable
left is `RUST_LOG`.

A key that is not recognised is an **error**, in the common section and inside a
`[[profile]]` alike. A misspelled setting that silently does nothing is the
failure this replaces.

### Required

`source_stream`, `model_path`, `model_input_size`, `model_conf_body`,
`model_conf_head`, `model_iou`, `esp_button_1_profile`, `esp_button_2_profile`,
a `[mouse]` table with a `port`, and at least one `[[profile]]` with a `name`
and a `scale_min_zone`.

Unlike the environment version, `trt_*` and `openvino_cache_dir` are **not**
required: they are only read by the provider that uses them, so an OpenVINO or
CPU build no longer needs TensorRT settings to start.

### Core

| Key | Default | Meaning |
| --- | --- | --- |
| `source_stream` | **required** | Frame source; see [Frame sources](#frame-sources) |
| `event_listener_port` | `10000` | Port for the event listener |
| `screen_width` / `screen_height` | `1920` / `1080` | Full frame size; the crosshair is the centre of this |
| `region_left` / `region_top` | `0` / `0` | Detection region origin |
| `region_width` / `region_height` | `0` / `0` | Detection region size. Cropping happens only when this differs from the screen size |
| `intra_threads` | `1` | ONNX Runtime intra-op threads |
| `default_aim_mode` | unset | `0` head, `1` neck, `2` chest, `3` abdomen, `4` horizon |
| `game_sens` | `1.0` | In-game sensitivity, exactly as typed into the game |
| `game_hfov` | `103.0` | Horizontal field of view in degrees |
| `game_yaw` | `0.07` | Degrees of view rotation per mouse count at sensitivity 1 — the engine's own constant |
| `aim_scale` | `1.0` | Measured multiplier on every move; see [Calibrating aim_scale](#calibrating-aim_scale) |
| `RUST_LOG` (env) | `info` | e.g. `info,aimbot=debug` for per-stage timings |

### Profiles

Each ESP button selects one `[[profile]]` by its position in the file, counting
from zero. `esp_button_1_profile` also covers the MAKCU side-4 button and the
no-trigger auto-aim path.

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | **required** | Shown in logs; has no effect on aim |
| `scale_min_zone` | **required** | Dead-zone scale. Below `min_zone * this` no move is queued |
| `fov` | whole frame | Maximum engagement distance in frame pixels. `0` or absent means no limit |
| `mover` | `"smooth"` | `"smooth"` for the human-flick model, `"bezier"` to hand the whole delta to the device, `"direct"` to send it untouched |
| `aim_mode` | follows the runtime mode | `"head"`, `"neck"`, `"chest"`, `"abdomen"`, `"horizon"`. Set, it pins this profile to that mode; absent, it follows whatever the event listener last selected |

Each profile carries its own mover instance, so the sub-count remainder, the
reaction window and the auto-click rate limit are tracked per profile rather
than shared.

```toml
esp_button_1_profile = 0
esp_button_2_profile = 1

[[profile]]
name = "tracking"
scale_min_zone = 0.25
fov = 120.0

[[profile]]
name = "flick"
scale_min_zone = 0.85
aim_mode = "horizon"
move_smooth_auto_click = true
move_smooth_auto_click_miss_p = 0.1
```

### Model

| Key | Default | Meaning |
| --- | --- | --- |
| `model_path` | **required** | Model file. A name starting with `v_light` selects the light pipeline |
| `model_input_size` | **required** | Must match the model input (192 for `v_light_192`) |
| `model_conf_body` | **required** | Confidence threshold for class 0 |
| `model_conf_head` | **required** | Confidence threshold for class 1 |
| `model_iou` | **required** | NMS IoU, both pipelines |
| `model_provider` | `cpu` | `cpu`, `openvino`, `tensorrt`/`trt`, `rocm`, `migraphx`/`mrx` |
| `build_head_iou` | unset | When set, synthesises a head box from each unmatched body box at this IoU |

### Providers

| Key | Default | Meaning |
| --- | --- | --- |
| `openvino_cache_dir` | `""` | Compiled-blob cache directory |
| `openvino_device_type` | `CPU` | `CPU`, `GPU`, `NPU`, `GPU.0`, ... |
| `trt_cache_dir` | `""` | TensorRT engine cache |
| `trt_min_shapes` / `trt_opt_shapes` / `trt_max_shapes` | `""` | e.g. `images:1x3x256x256` |
| `trt_fp16` | unset | Build the engine in FP16 |
| `trt_max_partition_iterations` | `10` | |
| `trt_builder_optimization_level` | `3` | `0`–`5`; below 3 builds faster but runs slower |
| `trt_dla_enable` / `trt_dla_core` | `false` / `0` | |
| `trt_auxiliary_streams` | `-1` | |
| `gpu_id` | `0` | Device index |
| `gpu_mem_limit` | `1073741824` | Workspace / memory limit in bytes |

### NDI

| Key | Default | Meaning |
| --- | --- | --- |
| `ndi_source_name` | unset | Prefer the source whose name matches |
| `ndi_timeout_ms` | `1000` | Receive timeout in ms |

### Output

The mouse link lives in its own `[mouse]` table, documented under
[Mouse devices](#mouse-devices). These two stay at the top level, because
neither is a fact about the mouse:

| Key | Default | Meaning |
| --- | --- | --- |
| `game_sens` | `1.0` | In-game sensitivity, exactly as typed into the game |
| `aim_scale` | `1.0` | Measured multiplier on every move. **Calibrate this first**; see [Calibrating aim_scale](#calibrating-aim_scale) |
| `game_hfov` | `103.0` | Horizontal field of view in degrees |
| `game_yaw` | `0.07` | Degrees of view rotation per count at sensitivity 1 |
| `esp_port` | unset | Serial port of the ESP button board |

`game_hfov` is a game setting you can read off. `game_yaw` is the engine
constant — `m_yaw` in Source-derived engines, `0.022` there — not something to
tune by feel, and the `0.07` default comes out of a reference snippet rather
than from this game. Where it is wrong, `aim_scale` absorbs the difference.
All are refused at startup if non-positive, and `game_hfov` must be under
180°.

### Human-flick movement

These live **inside a `[[profile]]`**, so each ESP button can be tuned on its
own. What each one does, how to compute its effect and presets by fov are under
[Mouse movement](#settings).

| Key | Default | Meaning |
| --- | --- | --- |
| `move_smooth_poll_hz` | `0` | Report cadence; `0` derives it from `[mouse].baud` |
| `move_smooth_fitts_a_ms` / `_b_ms` | `35` / `55` | Fitts intercept and slope, ms and ms per bit |
| `move_smooth_min_ms` / `_max_ms` | `45` / `320` | Movement-time clamp |
| `move_smooth_gain` | `0.90` | Fraction of the distance the ballistic phase commits to |
| `move_smooth_overshoot_p` | `0.15` | Share of flicks that overshoot instead of falling short |
| `move_smooth_peak_min` / `_peak_max` | `0.30` / `0.45` | Where peak speed sits, as a fraction of the movement |
| `move_smooth_bow` | `0.03` | Maximum perpendicular deviation, as a fraction of the amplitude |
| `move_smooth_noise` | `0.022` | Signal-dependent motor noise coefficient |
| `move_smooth_tremor` | `0.35` | Physiological tremor amplitude in counts; `0` disables it |
| `move_smooth_react_min_ms` / `_max_ms` | `0` / `0` | Reaction latency on a fresh target; off by default |
| `move_smooth_counts_per_report` | `2.0` | Mean counts one report carries; caps how many reports a flick plans |
| `move_smooth_min_speed_ips` / `_max_speed_ips` | `0.6` / `40` | Hand-speed envelope in inches per second; the floor sets the duration below a few hundred counts |
| `move_smooth_move_now` | `false` | Ask the device to put the counts in its next report untouched, skipping any interpolation of its own. Probed at connect and ignored where the device has no such command |
| `move_smooth_auto_click` | `false` | Click once a flick has settled |
| `move_smooth_auto_click_lower_ms` / `_upper_ms` | `100` / `130` | Wait between the flick and the click |
| `move_smooth_auto_click_rate_limit_ms` | `0` | Smallest gap between two clicks, ms; `0` is no limit |
| `move_smooth_auto_click_miss_p` | `0.0` | Chance the shot is fired without letting the aim settle |

The remaining `SmoothConfig` fields are listed under
[Rarely touched](#rarely-touched).

Capture-card keys are in
[Capture from a capture card](#capture-from-a-capture-card).

## Cargo features

| Feature | Effect |
| --- | --- |
| `ndi4` (default) | NDI 4 via the `ndi` crate |
| `ndi6` | NDI 6 via `grafton-ndi` |
| `disable-mouse` | Build without mouse output; detection only |
| `debug` | Write annotated frames to `assets/debug` |
| `save-bbox` | `debug` plus YOLO-format label files |
