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
| `SCALE_MIN_ZONE2` | `0.8` | Dead-zone scale while ESP button 2 forces head aim |
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
