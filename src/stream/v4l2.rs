//! Minimal V4L2 `MMAP` streaming capture, written for latency rather than
//! generality. Used by [`crate::stream::Elgato`] for USB capture cards such as
//! the Elgato 4K X, which the kernel exposes through `uvcvideo`.
//!
//! What keeps the latency down:
//!
//! * `MMAP` buffers, so a dequeued frame is read straight out of the DMA
//!   buffer with no copy on the capture side;
//! * a small buffer ring (3 by default) so the driver cannot build a queue of
//!   frames ahead of us;
//! * after the first frame is ready we keep dequeuing until the driver reports
//!   `EAGAIN` and hand back only the newest one, returning the older buffers
//!   immediately. A frame that is already stale by the time we look at it is
//!   worse than no frame at all for aiming;
//! * `O_NONBLOCK` + a single `poll()` per grab, so a dead signal surfaces as a
//!   timeout instead of a hung thread.
//!
//! Struct layouts and ioctl request codes below are transcribed from
//! `linux/videodev2.h`; the `const` size assertions next to each struct are
//! there to catch a mistake at compile time rather than as garbage pixels.

use anyhow::{Context, Result, anyhow, bail};
use std::ffi::CString;
use std::io;
use std::os::raw::{c_int, c_void};
use std::time::Duration;

// ---------------------------------------------------------------------------
// videodev2.h constants
// ---------------------------------------------------------------------------

const VIDIOC_QUERYCAP: u64 = 0x8068_5600;
const VIDIOC_ENUM_FMT: u64 = 0xc040_5602;
const VIDIOC_S_FMT: u64 = 0xc0d0_5605;
const VIDIOC_REQBUFS: u64 = 0xc014_5608;
const VIDIOC_QUERYBUF: u64 = 0xc058_5609;
const VIDIOC_QBUF: u64 = 0xc058_560f;
const VIDIOC_DQBUF: u64 = 0xc058_5611;
const VIDIOC_STREAMON: u64 = 0x4004_5612;
const VIDIOC_STREAMOFF: u64 = 0x4004_5613;
const VIDIOC_S_PARM: u64 = 0xc0cc_5616;

const V4L2_BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
const V4L2_FIELD_NONE: u32 = 1;
const V4L2_MEMORY_MMAP: u32 = 1;

const V4L2_CAP_VIDEO_CAPTURE: u32 = 0x0000_0001;
const V4L2_CAP_STREAMING: u32 = 0x0400_0000;
const V4L2_CAP_DEVICE_CAPS: u32 = 0x8000_0000;
const V4L2_CAP_TIMEPERFRAME: u32 = 0x1000;
const V4L2_BUF_FLAG_ERROR: u32 = 0x0000_0040;

/// Build a FourCC the way `v4l2_fourcc()` does.
pub const fn fourcc(a: u8, b: u8, c: u8, d: u8) -> u32 {
    (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24)
}

pub const V4L2_PIX_FMT_NV12: u32 = fourcc(b'N', b'V', b'1', b'2');
pub const V4L2_PIX_FMT_NV21: u32 = fourcc(b'N', b'V', b'2', b'1');
/// Defined for recognition in logs; there is no OpenCV NV16 -> BGR code, so
/// it is deliberately absent from [`PREFERRED_FOURCC`].
pub const V4L2_PIX_FMT_NV16: u32 = fourcc(b'N', b'V', b'1', b'6');
pub const V4L2_PIX_FMT_YUYV: u32 = fourcc(b'Y', b'U', b'Y', b'V');
pub const V4L2_PIX_FMT_UYVY: u32 = fourcc(b'U', b'Y', b'V', b'Y');
pub const V4L2_PIX_FMT_YUV420: u32 = fourcc(b'Y', b'U', b'1', b'2');
pub const V4L2_PIX_FMT_BGR24: u32 = fourcc(b'B', b'G', b'R', b'3');
pub const V4L2_PIX_FMT_RGB24: u32 = fourcc(b'R', b'G', b'B', b'3');
pub const V4L2_PIX_FMT_ABGR32: u32 = fourcc(b'A', b'R', b'2', b'4');
pub const V4L2_PIX_FMT_MJPEG: u32 = fourcc(b'M', b'J', b'P', b'G');

/// Negotiation order when no FourCC is pinned by configuration.
///
/// Bytes on the wire are latency on a USB capture card, so the 12-bit planar
/// formats come first and the compressed one comes last: MJPEG would add a
/// decode step and a quality loss on top of the transfer.
/// Every entry must have a conversion in `stream::elgato::layout_for`, so that
/// negotiation never lands on a format the pipeline cannot turn into BGR.
pub const PREFERRED_FOURCC: [u32; 6] = [
    V4L2_PIX_FMT_NV12,
    V4L2_PIX_FMT_YUV420,
    V4L2_PIX_FMT_YUYV,
    V4L2_PIX_FMT_UYVY,
    V4L2_PIX_FMT_BGR24,
    V4L2_PIX_FMT_MJPEG,
];

pub fn fourcc_to_string(v: u32) -> String {
    let bytes = [v as u8, (v >> 8) as u8, (v >> 16) as u8, (v >> 24) as u8];
    bytes
        .iter()
        .map(|&b| if b.is_ascii_graphic() { b as char } else { '?' })
        .collect()
}

pub fn parse_fourcc(s: &str) -> Result<u32> {
    let s = s.trim();
    let b = s.as_bytes();
    if b.len() != 4 {
        bail!("fourcc must be exactly 4 characters, got {:?}", s);
    }
    Ok(fourcc(b[0], b[1], b[2], b[3]))
}

// ---------------------------------------------------------------------------
// videodev2.h structs
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct V4l2Capability {
    driver: [u8; 16],
    card: [u8; 32],
    bus_info: [u8; 32],
    version: u32,
    capabilities: u32,
    device_caps: u32,
    reserved: [u32; 3],
}
const _: () = assert!(std::mem::size_of::<V4l2Capability>() == 104);

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct V4l2PixFormat {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
    private: u32,
    flags: u32,
    ycbcr_enc: u32,
    quantization: u32,
    xfer_func: u32,
}
const _: () = assert!(std::mem::size_of::<V4l2PixFormat>() == 48);

/// `struct v4l2_format`: the union starts at offset 8 and is 200 bytes wide.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct V4l2Format {
    type_: u32,
    _pad: u32,
    raw: [u8; 200],
}
const _: () = assert!(std::mem::size_of::<V4l2Format>() == 208);
const _: () = assert!(std::mem::offset_of!(V4l2Format, raw) == 8);

impl V4l2Format {
    fn capture(pix: &V4l2PixFormat) -> Self {
        let mut f = Self {
            type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
            _pad: 0,
            raw: [0; 200],
        };
        f.set_pix(pix);
        f
    }

    fn set_pix(&mut self, pix: &V4l2PixFormat) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (pix as *const V4l2PixFormat).cast::<u8>(),
                std::mem::size_of::<V4l2PixFormat>(),
            )
        };
        self.raw[..bytes.len()].copy_from_slice(bytes);
    }

    fn pix(&self) -> V4l2PixFormat {
        let mut pix = V4l2PixFormat::default();
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.raw.as_ptr(),
                (&mut pix as *mut V4l2PixFormat).cast::<u8>(),
                std::mem::size_of::<V4l2PixFormat>(),
            );
        }
        pix
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct V4l2RequestBuffers {
    count: u32,
    type_: u32,
    memory: u32,
    capabilities: u32,
    reserved: [u32; 1],
}
const _: () = assert!(std::mem::size_of::<V4l2RequestBuffers>() == 20);

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Timeval {
    tv_sec: i64,
    tv_usec: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct V4l2Timecode {
    type_: u32,
    flags: u32,
    frames: u8,
    seconds: u8,
    minutes: u8,
    hours: u8,
    userbits: [u8; 4],
}
const _: () = assert!(std::mem::size_of::<V4l2Timecode>() == 16);

#[repr(C, align(8))]
#[derive(Clone, Copy, Default)]
struct V4l2Buffer {
    index: u32,
    type_: u32,
    bytesused: u32,
    flags: u32,
    field: u32,
    _pad: u32,
    timestamp: Timeval,
    timecode: V4l2Timecode,
    sequence: u32,
    memory: u32,
    /// union { offset, userptr, planes, fd }
    m: u64,
    length: u32,
    reserved2: u32,
    request_fd: i32,
    _pad2: u32,
}
const _: () = assert!(std::mem::size_of::<V4l2Buffer>() == 88);
const _: () = assert!(std::mem::offset_of!(V4l2Buffer, timestamp) == 24);
const _: () = assert!(std::mem::offset_of!(V4l2Buffer, timecode) == 40);
const _: () = assert!(std::mem::offset_of!(V4l2Buffer, m) == 64);
const _: () = assert!(std::mem::offset_of!(V4l2Buffer, length) == 72);

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct V4l2Fract {
    numerator: u32,
    denominator: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct V4l2CaptureParm {
    capability: u32,
    capturemode: u32,
    timeperframe: V4l2Fract,
    extendedmode: u32,
    readbuffers: u32,
    reserved: [u32; 4],
}
const _: () = assert!(std::mem::size_of::<V4l2CaptureParm>() == 40);

/// `struct v4l2_streamparm`: union at offset 4, 200 bytes wide, 4-aligned.
#[repr(C)]
#[derive(Clone, Copy)]
struct V4l2StreamParm {
    type_: u32,
    raw: [u8; 200],
}
const _: () = assert!(std::mem::size_of::<V4l2StreamParm>() == 204);
const _: () = assert!(std::mem::offset_of!(V4l2StreamParm, raw) == 4);

impl V4l2StreamParm {
    fn capture(parm: &V4l2CaptureParm) -> Self {
        let mut s = Self {
            type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
            raw: [0; 200],
        };
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (parm as *const V4l2CaptureParm).cast::<u8>(),
                std::mem::size_of::<V4l2CaptureParm>(),
            )
        };
        s.raw[..bytes.len()].copy_from_slice(bytes);
        s
    }

    fn parm(&self) -> V4l2CaptureParm {
        let mut parm = V4l2CaptureParm::default();
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.raw.as_ptr(),
                (&mut parm as *mut V4l2CaptureParm).cast::<u8>(),
                std::mem::size_of::<V4l2CaptureParm>(),
            );
        }
        parm
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct V4l2FmtDesc {
    index: u32,
    type_: u32,
    flags: u32,
    description: [u8; 32],
    pixelformat: u32,
    mbus_code: u32,
    reserved: [u32; 3],
}
const _: () = assert!(std::mem::size_of::<V4l2FmtDesc>() == 64);

// ---------------------------------------------------------------------------
// ioctl plumbing
// ---------------------------------------------------------------------------

/// `ioctl` retried across `EINTR`, which V4L2 drivers hand out freely.
unsafe fn xioctl<T>(fd: c_int, request: u64, arg: *mut T) -> io::Result<()> {
    loop {
        let r = unsafe { libc::ioctl(fd, request as _, arg.cast::<c_void>()) };
        if r >= 0 {
            return Ok(());
        }
        let err = io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EINTR) {
            continue;
        }
        return Err(err);
    }
}

fn cstr_name(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Config {
    pub device: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// Pinned FourCC, or `None` to negotiate from [`PREFERRED_FOURCC`].
    pub fourcc: Option<u32>,
    /// Size of the MMAP ring. 2 is the lowest the kernel accepts, 3 leaves one
    /// buffer in flight while we hold one and is the better default for UVC.
    pub buffers: u32,
    /// Drain the driver queue on every grab and keep only the newest frame.
    pub drop_stale: bool,
    pub timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            device: String::from("/dev/video0"),
            width: 1920,
            height: 1080,
            fps: 60,
            fourcc: None,
            buffers: 3,
            drop_stale: true,
            timeout: Duration::from_millis(1000),
        }
    }
}

/// What the driver actually agreed to, which is not always what was asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    pub width: u32,
    pub height: u32,
    pub fourcc: u32,
    pub bytes_per_line: u32,
    pub size_image: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FrameMeta {
    pub index: u32,
    pub sequence: u32,
    /// Driver timestamp of the frame, on the same clock as
    /// `CLOCK_MONOTONIC` for `uvcvideo`.
    pub timestamp: Duration,
    pub bytes_used: u32,
    /// Frames thrown away during this grab because a newer one was queued.
    pub dropped_stale: u32,
}

struct Mapping {
    ptr: *mut c_void,
    len: usize,
}

impl Mapping {
    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }
}

pub struct Capture {
    fd: c_int,
    config: Config,
    format: Format,
    fps: u32,
    card: String,
    buffers: Vec<Mapping>,
    streaming: bool,
}

// The fd and the mappings are owned exclusively by this struct and every use
// goes through `&mut self`, so it is safe to hand the whole thing to the
// capture thread.
unsafe impl Send for Capture {}

impl Capture {
    pub fn open(config: Config) -> Result<Self> {
        if config.width == 0 || config.height == 0 {
            bail!("[V4L2] capture resolution must be non-zero");
        }
        if config.buffers < 2 {
            bail!("[V4L2] need at least 2 buffers, got {}", config.buffers);
        }

        let path = CString::new(config.device.as_str())
            .with_context(|| format!("[V4L2] bad device path {:?}", config.device))?;
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_NONBLOCK) };
        if fd < 0 {
            return Err(anyhow!(io::Error::last_os_error()))
                .with_context(|| format!("[V4L2] cannot open {}", config.device));
        }

        let mut this = Self {
            fd,
            config,
            format: Format {
                width: 0,
                height: 0,
                fourcc: 0,
                bytes_per_line: 0,
                size_image: 0,
            },
            fps: 0,
            card: String::new(),
            buffers: Vec::new(),
            streaming: false,
        };

        // Anything that fails from here on must still close the fd; `this`
        // owns it now, so an early return runs `Drop`.
        this.query_capabilities()?;
        let fourcc = this.negotiate_fourcc()?;
        this.set_format(fourcc)?;
        this.set_frame_rate()?;
        this.request_buffers()?;
        this.stream_on()?;

        tracing::info!(
            "[V4L2] {} ({}) streaming {}x{} {} @ {} fps, {} buffers, drop_stale={}",
            this.config.device,
            this.card,
            this.format.width,
            this.format.height,
            fourcc_to_string(this.format.fourcc),
            this.fps,
            this.buffers.len(),
            this.config.drop_stale,
        );
        Ok(this)
    }

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn fps(&self) -> u32 {
        self.fps
    }

    pub fn card(&self) -> &str {
        &self.card
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Tear the stream down and bring it back up with the same configuration.
    pub fn restart(&mut self) -> Result<()> {
        self.teardown();
        let path = CString::new(self.config.device.as_str())?;
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_NONBLOCK) };
        if fd < 0 {
            return Err(anyhow!(io::Error::last_os_error()))
                .with_context(|| format!("[V4L2] cannot reopen {}", self.config.device));
        }
        self.fd = fd;
        self.query_capabilities()?;
        let fourcc = self.negotiate_fourcc()?;
        self.set_format(fourcc)?;
        self.set_frame_rate()?;
        self.request_buffers()?;
        self.stream_on()?;
        Ok(())
    }

    /// Grab the newest available frame and run `f` over the mapped bytes.
    ///
    /// The buffer is handed back to the driver as soon as `f` returns, error or
    /// not, so `f` must copy or consume whatever it needs. Keeping the callback
    /// short is the whole point: the driver is one buffer down until it ends.
    pub fn with_frame<T>(&mut self, f: impl FnOnce(&[u8], &FrameMeta) -> Result<T>) -> Result<T> {
        let (index, meta) = self.dequeue_newest()?;
        let bytes = {
            let mapping = &self.buffers[index as usize];
            // Some drivers leave `bytesused` at zero for uncompressed formats,
            // where the whole buffer is the frame by definition.
            let len = if meta.bytes_used == 0 {
                mapping.len
            } else {
                (meta.bytes_used as usize).min(mapping.len)
            };
            &mapping.as_slice()[..len]
        };
        let result = f(bytes, &meta);
        // Requeue before propagating: a leaked buffer starves the ring.
        let requeue = self.queue_buffer(index);
        let value = result?;
        requeue?;
        Ok(value)
    }

    // -- setup steps --------------------------------------------------------

    fn query_capabilities(&mut self) -> Result<()> {
        let mut cap = V4l2Capability {
            driver: [0; 16],
            card: [0; 32],
            bus_info: [0; 32],
            version: 0,
            capabilities: 0,
            device_caps: 0,
            reserved: [0; 3],
        };
        unsafe { xioctl(self.fd, VIDIOC_QUERYCAP, &mut cap) }
            .map_err(|e| anyhow!("[V4L2] VIDIOC_QUERYCAP on {}: {}", self.config.device, e))?;

        self.card = cstr_name(&cap.card);
        let caps = if cap.capabilities & V4L2_CAP_DEVICE_CAPS != 0 {
            cap.device_caps
        } else {
            cap.capabilities
        };
        if caps & V4L2_CAP_VIDEO_CAPTURE == 0 {
            bail!(
                "[V4L2] {} ({}) is not a single-planar video capture device",
                self.config.device,
                self.card
            );
        }
        if caps & V4L2_CAP_STREAMING == 0 {
            bail!(
                "[V4L2] {} ({}) does not support streaming I/O",
                self.config.device,
                self.card
            );
        }
        tracing::debug!(
            "[V4L2] {} driver={} card={} bus={}",
            self.config.device,
            cstr_name(&cap.driver),
            self.card,
            cstr_name(&cap.bus_info)
        );
        Ok(())
    }

    /// Every FourCC the device advertises for video capture.
    pub fn supported_formats(&self) -> Result<Vec<(u32, String)>> {
        let mut out = Vec::new();
        for index in 0..64u32 {
            let mut desc = V4l2FmtDesc {
                index,
                type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
                flags: 0,
                description: [0; 32],
                pixelformat: 0,
                mbus_code: 0,
                reserved: [0; 3],
            };
            match unsafe { xioctl(self.fd, VIDIOC_ENUM_FMT, &mut desc) } {
                Ok(()) => out.push((desc.pixelformat, cstr_name(&desc.description))),
                Err(e) if e.raw_os_error() == Some(libc::EINVAL) => break,
                Err(e) => return Err(anyhow!("[V4L2] VIDIOC_ENUM_FMT: {}", e)),
            }
        }
        Ok(out)
    }

    fn negotiate_fourcc(&self) -> Result<u32> {
        let supported = self.supported_formats().unwrap_or_default();
        let list = supported
            .iter()
            .map(|(f, d)| format!("{} ({})", fourcc_to_string(*f), d))
            .collect::<Vec<_>>()
            .join(", ");
        tracing::debug!("[V4L2] {} supports: {}", self.config.device, list);

        if let Some(wanted) = self.config.fourcc {
            if !supported.is_empty() && !supported.iter().any(|(f, _)| *f == wanted) {
                tracing::warn!(
                    "[V4L2] {} does not advertise {}; trying it anyway. Advertised: {}",
                    self.config.device,
                    fourcc_to_string(wanted),
                    list
                );
            }
            return Ok(wanted);
        }

        if supported.is_empty() {
            // Nothing enumerated: fall back to the cheapest format and let
            // VIDIOC_S_FMT have the last word.
            return Ok(PREFERRED_FOURCC[0]);
        }
        PREFERRED_FOURCC
            .into_iter()
            .find(|pref| supported.iter().any(|(f, _)| f == pref))
            .ok_or_else(|| {
                anyhow!(
                    "[V4L2] {} advertises no format this pipeline can convert. Advertised: {}",
                    self.config.device,
                    list
                )
            })
    }

    fn set_format(&mut self, fourcc: u32) -> Result<()> {
        let pix = V4l2PixFormat {
            width: self.config.width,
            height: self.config.height,
            pixelformat: fourcc,
            field: V4L2_FIELD_NONE,
            ..Default::default()
        };
        let mut fmt = V4l2Format::capture(&pix);
        unsafe { xioctl(self.fd, VIDIOC_S_FMT, &mut fmt) }
            .map_err(|e| anyhow!("[V4L2] VIDIOC_S_FMT: {}", e))?;

        // S_FMT adjusts silently rather than failing, so read back what we got.
        let got = fmt.pix();
        if got.width != self.config.width || got.height != self.config.height {
            tracing::warn!(
                "[V4L2] requested {}x{} but the driver chose {}x{}",
                self.config.width,
                self.config.height,
                got.width,
                got.height
            );
        }
        if got.pixelformat != fourcc {
            tracing::warn!(
                "[V4L2] requested {} but the driver chose {}",
                fourcc_to_string(fourcc),
                fourcc_to_string(got.pixelformat)
            );
        }
        if got.sizeimage == 0 {
            bail!("[V4L2] driver reported a zero-size image format");
        }
        self.format = Format {
            width: got.width,
            height: got.height,
            fourcc: got.pixelformat,
            bytes_per_line: got.bytesperline,
            size_image: got.sizeimage,
        };
        Ok(())
    }

    fn set_frame_rate(&mut self) -> Result<()> {
        if self.config.fps == 0 {
            self.fps = 0;
            return Ok(());
        }
        let parm = V4l2CaptureParm {
            timeperframe: V4l2Fract {
                numerator: 1,
                denominator: self.config.fps,
            },
            ..Default::default()
        };
        let mut sp = V4l2StreamParm::capture(&parm);
        match unsafe { xioctl(self.fd, VIDIOC_S_PARM, &mut sp) } {
            Ok(()) => {
                let got = sp.parm();
                if got.capability & V4L2_CAP_TIMEPERFRAME == 0 {
                    tracing::warn!(
                        "[V4L2] {} does not support setting the frame interval; \
                         running at whatever the device produces",
                        self.config.device
                    );
                    self.fps = 0;
                    return Ok(());
                }
                let tpf = got.timeperframe;
                self.fps = if tpf.numerator > 0 {
                    tpf.denominator / tpf.numerator
                } else {
                    0
                };
                if self.fps != self.config.fps {
                    tracing::warn!(
                        "[V4L2] requested {} fps but the driver chose {}/{} ({} fps)",
                        self.config.fps,
                        tpf.denominator,
                        tpf.numerator,
                        self.fps
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    "[V4L2] VIDIOC_S_PARM failed ({}); leaving the default rate",
                    e
                );
                self.fps = 0;
            }
        }
        Ok(())
    }

    fn request_buffers(&mut self) -> Result<()> {
        let mut req = V4l2RequestBuffers {
            count: self.config.buffers,
            type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
            memory: V4L2_MEMORY_MMAP,
            ..Default::default()
        };
        unsafe { xioctl(self.fd, VIDIOC_REQBUFS, &mut req) }
            .map_err(|e| anyhow!("[V4L2] VIDIOC_REQBUFS: {}", e))?;
        if req.count < 2 {
            bail!("[V4L2] driver granted only {} buffer(s)", req.count);
        }
        if req.count != self.config.buffers {
            tracing::warn!(
                "[V4L2] requested {} buffers, got {}",
                self.config.buffers,
                req.count
            );
        }

        for index in 0..req.count {
            let mut buf = V4l2Buffer {
                index,
                type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
                memory: V4L2_MEMORY_MMAP,
                ..Default::default()
            };
            unsafe { xioctl(self.fd, VIDIOC_QUERYBUF, &mut buf) }
                .map_err(|e| anyhow!("[V4L2] VIDIOC_QUERYBUF({}): {}", index, e))?;

            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    buf.length as libc::size_t,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    self.fd,
                    buf.m as libc::off_t,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(anyhow!(io::Error::last_os_error()))
                    .with_context(|| format!("[V4L2] mmap of buffer {} failed", index));
            }
            self.buffers.push(Mapping {
                ptr,
                len: buf.length as usize,
            });
        }
        Ok(())
    }

    fn stream_on(&mut self) -> Result<()> {
        for index in 0..self.buffers.len() as u32 {
            self.queue_buffer(index)?;
        }
        let mut type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE as c_int;
        unsafe { xioctl(self.fd, VIDIOC_STREAMON, &mut type_) }
            .map_err(|e| anyhow!("[V4L2] VIDIOC_STREAMON: {}", e))?;
        self.streaming = true;
        Ok(())
    }

    // -- steady state -------------------------------------------------------

    fn queue_buffer(&self, index: u32) -> Result<()> {
        let mut buf = V4l2Buffer {
            index,
            type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
            memory: V4L2_MEMORY_MMAP,
            ..Default::default()
        };
        unsafe { xioctl(self.fd, VIDIOC_QBUF, &mut buf) }
            .map_err(|e| anyhow!("[V4L2] VIDIOC_QBUF({}): {}", index, e))?;
        Ok(())
    }

    /// Dequeue one buffer without blocking. `Ok(None)` means the driver has
    /// nothing ready.
    fn try_dequeue(&self) -> Result<Option<V4l2Buffer>> {
        let mut buf = V4l2Buffer {
            type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
            memory: V4L2_MEMORY_MMAP,
            ..Default::default()
        };
        match unsafe { xioctl(self.fd, VIDIOC_DQBUF, &mut buf) } {
            Ok(()) => Ok(Some(buf)),
            Err(e) if e.raw_os_error() == Some(libc::EAGAIN) => Ok(None),
            Err(e) => Err(anyhow!("[V4L2] VIDIOC_DQBUF: {}", e)),
        }
    }

    fn wait_readable(&self) -> Result<()> {
        let timeout_ms = self
            .config
            .timeout
            .as_millis()
            .try_into()
            .unwrap_or(c_int::MAX);
        loop {
            let mut pfd = libc::pollfd {
                fd: self.fd,
                events: libc::POLLIN,
                revents: 0,
            };
            let n = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
            if n < 0 {
                let err = io::Error::last_os_error();
                if err.raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                return Err(anyhow!("[V4L2] poll: {}", err));
            }
            if n == 0 {
                bail!(
                    "[V4L2] no frame from {} within {:?} (signal lost?)",
                    self.config.device,
                    self.config.timeout
                );
            }
            if pfd.revents & libc::POLLERR != 0 {
                bail!("[V4L2] poll reported POLLERR on {}", self.config.device);
            }
            return Ok(());
        }
    }

    /// Block until a frame is ready, then drain whatever else the driver has
    /// queued and keep only the newest. Returns the held buffer index.
    fn dequeue_newest(&mut self) -> Result<(u32, FrameMeta)> {
        /// Bound on fruitless iterations, so a device stuck producing error
        /// frames (or a driver that reports readable with nothing to dequeue)
        /// surfaces as an error instead of spinning the capture thread.
        const MAX_FRUITLESS: u32 = 64;

        let mut dropped = 0u32;
        let mut fruitless = 0u32;
        let mut newest: Option<V4l2Buffer> = None;

        loop {
            if newest.is_none() {
                self.wait_readable()?;
            }
            match self.try_dequeue()? {
                Some(buf) => {
                    if buf.flags & V4L2_BUF_FLAG_ERROR != 0 {
                        // Corrupt frame: hand it straight back and keep looking.
                        tracing::debug!("[V4L2] discarding buffer {} flagged as error", buf.index);
                        self.queue_buffer(buf.index)?;
                        fruitless += 1;
                        if fruitless >= MAX_FRUITLESS {
                            bail!(
                                "[V4L2] {} returned {} error frames in a row",
                                self.config.device,
                                fruitless
                            );
                        }
                        continue;
                    }
                    if let Some(stale) = newest.replace(buf) {
                        dropped += 1;
                        self.queue_buffer(stale.index)?;
                    }
                    if !self.config.drop_stale {
                        break;
                    }
                }
                None => {
                    if newest.is_some() {
                        break;
                    }
                    // Readable but nothing to dequeue; wait again.
                    fruitless += 1;
                    if fruitless >= MAX_FRUITLESS {
                        bail!(
                            "[V4L2] {} reported readable {} times with no frame to dequeue",
                            self.config.device,
                            fruitless
                        );
                    }
                }
            }
        }

        let buf = newest.expect("loop only exits with a buffer in hand");
        if buf.index as usize >= self.buffers.len() {
            bail!(
                "[V4L2] driver returned out-of-range buffer index {}",
                buf.index
            );
        }
        if dropped > 0 {
            tracing::debug!(
                "[V4L2] dropped {} stale frame(s) to keep the newest",
                dropped
            );
        }
        Ok((
            buf.index,
            FrameMeta {
                index: buf.index,
                sequence: buf.sequence,
                timestamp: Duration::new(
                    buf.timestamp.tv_sec.max(0) as u64,
                    (buf.timestamp.tv_usec.max(0) as u32).saturating_mul(1000),
                ),
                bytes_used: buf.bytesused,
                dropped_stale: dropped,
            },
        ))
    }

    // -- teardown -----------------------------------------------------------

    fn teardown(&mut self) {
        if self.streaming {
            let mut type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE as c_int;
            let _ = unsafe { xioctl(self.fd, VIDIOC_STREAMOFF, &mut type_) };
            self.streaming = false;
        }
        for mapping in self.buffers.drain(..) {
            unsafe { libc::munmap(mapping.ptr, mapping.len) };
        }
        if self.fd >= 0 {
            unsafe { libc::close(self.fd) };
            self.fd = -1;
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.teardown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourcc_round_trips() {
        assert_eq!(fourcc_to_string(V4L2_PIX_FMT_NV12), "NV12");
        assert_eq!(fourcc_to_string(V4L2_PIX_FMT_YUYV), "YUYV");
        assert_eq!(fourcc_to_string(V4L2_PIX_FMT_MJPEG), "MJPG");
        assert_eq!(parse_fourcc("NV12").unwrap(), V4L2_PIX_FMT_NV12);
        assert_eq!(parse_fourcc(" MJPG ").unwrap(), V4L2_PIX_FMT_MJPEG);
        assert!(parse_fourcc("NV1").is_err());
    }

    #[test]
    fn fourcc_values_match_videodev2() {
        // Cross-checked against the values videodev2.h expands to.
        assert_eq!(V4L2_PIX_FMT_NV12, 0x3231_564e);
        assert_eq!(V4L2_PIX_FMT_YUYV, 0x5659_5559);
        assert_eq!(V4L2_PIX_FMT_MJPEG, 0x4750_4a4d);
        assert_eq!(V4L2_PIX_FMT_UYVY, 0x5956_5955);
        assert_eq!(V4L2_PIX_FMT_BGR24, 0x3352_4742);
        assert_eq!(V4L2_PIX_FMT_YUV420, 0x3231_5559);
        assert_eq!(V4L2_PIX_FMT_NV16, 0x3631_564e);
    }

    #[test]
    fn pix_format_survives_the_union_round_trip() {
        let pix = V4l2PixFormat {
            width: 3840,
            height: 2160,
            pixelformat: V4L2_PIX_FMT_NV12,
            field: V4L2_FIELD_NONE,
            bytesperline: 3840,
            sizeimage: 3840 * 2160 * 3 / 2,
            ..Default::default()
        };
        let fmt = V4l2Format::capture(&pix);
        let got = fmt.pix();
        assert_eq!(fmt.type_, V4L2_BUF_TYPE_VIDEO_CAPTURE);
        assert_eq!(got.width, 3840);
        assert_eq!(got.height, 2160);
        assert_eq!(got.pixelformat, V4L2_PIX_FMT_NV12);
        assert_eq!(got.sizeimage, 3840 * 2160 * 3 / 2);
    }

    #[test]
    fn stream_parm_survives_the_union_round_trip() {
        let parm = V4l2CaptureParm {
            timeperframe: V4l2Fract {
                numerator: 1,
                denominator: 144,
            },
            ..Default::default()
        };
        let sp = V4l2StreamParm::capture(&parm);
        assert_eq!(sp.type_, V4L2_BUF_TYPE_VIDEO_CAPTURE);
        assert_eq!(sp.parm().timeperframe.denominator, 144);
    }

    #[test]
    fn preferred_order_puts_compressed_last() {
        assert_eq!(PREFERRED_FOURCC[0], V4L2_PIX_FMT_NV12);
        assert_eq!(*PREFERRED_FOURCC.last().unwrap(), V4L2_PIX_FMT_MJPEG);
    }
}
