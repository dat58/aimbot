//! Raw-frame capture from a USB capture card (Elgato 4K X and friends) through
//! V4L2, wired into the [`StreamCapture`] trait the rest of the pipeline uses.
//!
//! The card shows up as a `uvcvideo` node (`/dev/videoN`) whether it is plugged
//! into a USB4/Thunderbolt port or a plain USB 3.2 one — USB4 ports expose a
//! USB 3.2 host, so nothing here is port-specific. What the port does decide is
//! how much bandwidth is available, and therefore which resolution/rate the
//! card will actually agree to.
//!
//! Latency budget per frame, from the card to the `Mat` handed downstream:
//!
//! 1. `poll()` until the driver has a filled buffer;
//! 2. drain the queue and keep only the newest buffer (see
//!    [`crate::stream::v4l2`]);
//! 3. produce the output frame.
//!
//! Step 3 is where `CAPTURE_ROI` matters. The detection pipeline only looks at
//! `REGION_*`, a few hundred pixels around the crosshair, so decoding or
//! converting the whole frame is work that is thrown away immediately. With
//! `CAPTURE_ROI=true` (the default) only that region is produced:
//!
//! * MJPEG goes through [`crate::stream::roi_jpeg`], which makes libjpeg skip
//!   the IDCT and colour conversion for every block outside the region —
//!   ~0.46 ms for a 192x192 region instead of ~7 ms for a full 1080p
//!   `imdecode`;
//! * packed formats (YUYV, UYVY, BGR3, ...) take a plain sub-`Mat` of the
//!   mapped DMA buffer, so only the region's rows are ever read;
//! * planar 4:2:0 (NV12, NV21, YU12) has its region gathered into a small
//!   contiguous buffer first, because a sub-rect of a planar frame is not
//!   itself a valid planar frame — the Y and chroma rows are not adjacent.
//!   The gather is ~55 KB for a 192x192 region.
//!
//! This is not MJPEG-only: every format above takes the region path, and the
//! saving on the uncompressed ones is just as real (NV12 1080p measured 0.65 ms
//! -> 0.06 ms). It does not make NV12 *fast* overall, because NV12 at 1080p120
//! is limited by the wire, not by conversion.
//!
//! Either way the output frame stays a screen-space region: detections are
//! mapped back through `REGION_LEFT`/`REGION_TOP` in [`crate::model`], so the
//! rest of the pipeline is unchanged. Anything that draws detections onto the
//! frame has to shift them by [`crate::model::Model::frame_origin`] — the
//! `debug` overlay in `main.rs` does. `CAPTURE_ROI=false` restores whole-frame
//! output.

use crate::config::Config;
use crate::stream::roi_jpeg::RoiJpeg;
use crate::stream::v4l2::{self, Capture, Format, fourcc_to_string, monotonic_now};
use crate::stream::{StreamCapture, StreamInfo};
use anyhow::{Result, anyhow, bail};
use opencv::core::{CV_8U, CV_MAKETYPE, Mat, MatTraitConst, MatTraitManual, Rect};
use opencv::imgproc::ColorConversionCodes;
use std::ffi::c_void;
use std::time::{Duration, Instant};

/// What `capture()` hands back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// 8-bit BGR, ready for the detection pipeline.
    Bgr,
    /// The device bytes exactly as they were DMA'd, in their native layout.
    Raw,
}

impl OutputFormat {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "bgr" | "" => Ok(Self::Bgr),
            "raw" => Ok(Self::Raw),
            other => bail!("CAPTURE_OUTPUT must be `bgr` or `raw`, got {:?}", other),
        }
    }
}

/// How the device bytes are arranged, which decides how a region is taken out
/// of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Packing {
    /// One row per image row; a sub-rect is a plain sub-`Mat`.
    Packed { bytes_per_pixel: u32 },
    /// Y plane, then one interleaved chroma plane at the same row pitch.
    SemiPlanar,
    /// Y plane, then a U plane, then a V plane, both at half the row pitch.
    Planar,
    /// Compressed payload; only [`RoiJpeg`] can read a region out of it.
    Jpeg,
}

/// How the device bytes map onto a `Mat` before conversion.
#[derive(Debug, Clone, Copy)]
struct Layout {
    /// Frame geometry the driver agreed to.
    width: i32,
    height: i32,
    /// Whole-frame `Mat` view over the mapped buffer.
    rows: i32,
    cols: i32,
    mat_type: i32,
    step: usize,
    /// `cvtColor` code, or `None` when the bytes are already BGR.
    convert: Option<i32>,
    packing: Packing,
    /// Chroma subsampling. A region has to start and end on these boundaries,
    /// otherwise the chroma samples it needs are not in the bytes it covers.
    sub_x: u32,
    sub_y: u32,
}

impl Layout {
    fn compressed(&self) -> bool {
        self.packing == Packing::Jpeg
    }
}

fn layout_for(format: &Format) -> Result<Layout> {
    let w = format.width as i32;
    let h = format.height as i32;
    if w <= 0 || h <= 0 {
        bail!("[Elgato] driver reported a {}x{} frame", w, h);
    }

    // `bytesperline` is the luma/packed row pitch and may be padded; honour it
    // rather than assuming a tight layout, otherwise a padded frame shears.
    let pitch = |bytes_per_pixel: u32| -> usize {
        if format.bytes_per_line > 0 {
            format.bytes_per_line as usize
        } else {
            format.width as usize * bytes_per_pixel as usize
        }
    };
    let planar_420 = |convert: i32, packing: Packing| Layout {
        width: w,
        height: h,
        rows: h * 3 / 2,
        cols: w,
        mat_type: CV_MAKETYPE(CV_8U, 1),
        step: pitch(1),
        convert: Some(convert),
        packing,
        sub_x: 2,
        sub_y: 2,
    };
    let packed = |bytes_per_pixel: u32, channels: i32, convert: Option<i32>, sub_x: u32| Layout {
        width: w,
        height: h,
        rows: h,
        cols: w,
        mat_type: CV_MAKETYPE(CV_8U, channels),
        step: pitch(bytes_per_pixel),
        convert,
        packing: Packing::Packed { bytes_per_pixel },
        sub_x,
        sub_y: 1,
    };

    let layout = match format.fourcc {
        v4l2::V4L2_PIX_FMT_NV12 => planar_420(
            ColorConversionCodes::COLOR_YUV2BGR_NV12 as i32,
            Packing::SemiPlanar,
        ),
        v4l2::V4L2_PIX_FMT_NV21 => planar_420(
            ColorConversionCodes::COLOR_YUV2BGR_NV21 as i32,
            Packing::SemiPlanar,
        ),
        // I420 and IYUV are the same code; opencv-rust keeps only the
        // canonical name because the aliases share a discriminant.
        v4l2::V4L2_PIX_FMT_YUV420 => planar_420(
            ColorConversionCodes::COLOR_YUV2BGR_IYUV as i32,
            Packing::Planar,
        ),
        // 4:2:2 packed: two pixels share a chroma pair, so a region must start
        // on an even column.
        v4l2::V4L2_PIX_FMT_YUYV => packed(
            2,
            2,
            Some(ColorConversionCodes::COLOR_YUV2BGR_YUY2 as i32),
            2,
        ),
        v4l2::V4L2_PIX_FMT_UYVY => packed(
            2,
            2,
            Some(ColorConversionCodes::COLOR_YUV2BGR_UYVY as i32),
            2,
        ),
        v4l2::V4L2_PIX_FMT_BGR24 => packed(3, 3, None, 1),
        // COLOR_RGB2BGR is an alias of COLOR_BGR2RGB — the same symmetric
        // 3-channel swap, and only the canonical name is generated.
        v4l2::V4L2_PIX_FMT_RGB24 => {
            packed(3, 3, Some(ColorConversionCodes::COLOR_BGR2RGB as i32), 1)
        }
        v4l2::V4L2_PIX_FMT_ABGR32 => {
            packed(4, 4, Some(ColorConversionCodes::COLOR_BGRA2BGR as i32), 1)
        }
        v4l2::V4L2_PIX_FMT_MJPEG => Layout {
            width: w,
            height: h,
            rows: 1,
            cols: 0, // filled in per frame from `bytesused`
            mat_type: CV_MAKETYPE(CV_8U, 1),
            step: 0,
            convert: None,
            packing: Packing::Jpeg,
            sub_x: 1,
            sub_y: 1,
        },
        other => bail!(
            "[Elgato] no BGR conversion for pixel format {} ({:#010x}). \
             Pin a supported one with CAPTURE_FOURCC (NV12, YU12, YUYV, UYVY, BGR3, RGB3, AR24, MJPG).",
            fourcc_to_string(other),
            other
        ),
    };
    Ok(layout)
}

/// The region of the frame to hand downstream, in device pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Region {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
}

impl Region {
    /// The region grown outwards to the chroma grid, which is the smallest
    /// rect that can actually be converted, plus where the requested rect sits
    /// inside it.
    fn aligned(&self, sub_x: u32, sub_y: u32) -> (Region, i32, i32) {
        let (sx, sy) = (sub_x.max(1) as i32, sub_y.max(1) as i32);
        let x = self.x - self.x % sx;
        let y = self.y - self.y % sy;
        let w = (self.w + (self.x - x) + sx - 1) / sx * sx;
        let h = (self.h + (self.y - y) + sy - 1) / sy * sy;
        (Region { x, y, w, h }, self.x - x, self.y - y)
    }
}

pub struct Elgato {
    cap: Capture,
    output: OutputFormat,
    layout: Layout,
    /// Region to hand downstream, or `None` to return whole frames.
    region: Option<Region>,
    /// Configured region, kept so `reconnect` can re-validate it against a
    /// mode the card may have come back on.
    wanted_region: Option<Region>,
    jpeg: Option<RoiJpeg>,
    /// Reusable gather buffer for planar regions.
    scratch: Vec<u8>,
}

impl Elgato {
    /// Build the capture from the process configuration. `device` overrides
    /// `CAPTURE_DEVICE`, which is how `SOURCE_STREAM=elgato:///dev/video1`
    /// selects a specific node.
    pub fn new(config: &Config, device: Option<&str>) -> Result<Self> {
        let v4l2_config = v4l2::Config {
            device: device
                .map(str::to_string)
                .unwrap_or_else(|| config.capture_device.clone()),
            width: config.capture_width,
            height: config.capture_height,
            fps: config.capture_fps,
            fourcc: config.capture_fourcc,
            buffers: config.capture_buffers,
            drop_stale: config.capture_drop_stale,
            timeout: config.capture_timeout,
        };
        let cap = Capture::open(v4l2_config)?;
        let layout = layout_for(&cap.format())?;
        let output = config.capture_output;

        if output == OutputFormat::Raw {
            tracing::warn!(
                "[Elgato] CAPTURE_OUTPUT=raw: frames are handed back as {} bytes in \
                 native {} layout, not BGR. The detection pipeline expects BGR.",
                cap.format().size_image,
                fourcc_to_string(cap.format().fourcc)
            );
        }

        // `raw` is defined as the untouched device bytes, so there is nothing
        // sensible to crop out of it.
        let wanted_region = if config.capture_roi && output == OutputFormat::Bgr {
            Some(Region {
                x: config.region_left as i32,
                y: config.region_top as i32,
                w: config.region_width as i32,
                h: config.region_height as i32,
            })
        } else {
            None
        };

        let mut this = Self {
            cap,
            output,
            layout,
            region: None,
            wanted_region,
            jpeg: None,
            scratch: Vec::new(),
        };
        this.apply_region()?;
        this.log_mode();
        Ok(this)
    }

    pub fn format(&self) -> Format {
        self.cap.format()
    }

    /// Decide whether the configured region can be served from the mode the
    /// card settled on, and set up whatever that mode needs.
    fn apply_region(&mut self) -> Result<()> {
        let layout = self.layout;
        self.region = None;

        let Some(region) = self.wanted_region else {
            self.jpeg = None;
            return Ok(());
        };
        if region.w <= 0 || region.h <= 0 {
            bail!(
                "[Elgato] CAPTURE_ROI needs a non-empty REGION_WIDTH/REGION_HEIGHT, got {}x{}",
                region.w,
                region.h
            );
        }
        // A region that covers the frame is just the frame; skip the machinery.
        if region.x == 0 && region.y == 0 && region.w == layout.width && region.h == layout.height {
            tracing::debug!("[Elgato] REGION covers the whole frame; CAPTURE_ROI is a no-op");
            self.jpeg = None;
            return Ok(());
        }
        if region.x < 0
            || region.y < 0
            || region.x + region.w > layout.width
            || region.y + region.h > layout.height
        {
            bail!(
                "[Elgato] REGION {}x{}+{}+{} does not fit inside the {}x{} frame the card \
                 delivers. Fix REGION_*/CAPTURE_WIDTH/CAPTURE_HEIGHT, or set CAPTURE_ROI=false.",
                region.w,
                region.h,
                region.x,
                region.y,
                layout.width,
                layout.height
            );
        }
        if layout.packing == Packing::Planar && layout.step % 2 != 0 {
            bail!(
                "[Elgato] {} has an odd row pitch ({}), so its chroma planes cannot be \
                 addressed; set CAPTURE_ROI=false",
                fourcc_to_string(self.cap.format().fourcc),
                layout.step
            );
        }

        self.jpeg = if layout.compressed() {
            Some(RoiJpeg::new()?)
        } else {
            None
        };
        self.region = Some(region);
        Ok(())
    }

    fn log_mode(&self) {
        let format = self.cap.format();
        match self.region {
            Some(r) => {
                let (aligned, dx, dy) = r.aligned(self.layout.sub_x, self.layout.sub_y);
                tracing::info!(
                    "[Elgato] {} {}x{} -> region {}x{}+{}+{} ({}{})",
                    fourcc_to_string(format.fourcc),
                    format.width,
                    format.height,
                    r.w,
                    r.h,
                    r.x,
                    r.y,
                    match self.layout.packing {
                        Packing::Jpeg => "ROI jpeg decode",
                        Packing::Packed { .. } => "sub-Mat + convert",
                        Packing::SemiPlanar | Packing::Planar => "gather + convert",
                    },
                    if aligned == r {
                        String::new()
                    } else {
                        format!(
                            ", widened to {}x{}+{}+{} for {}:{} chroma then cropped by +{}+{}",
                            aligned.w,
                            aligned.h,
                            aligned.x,
                            aligned.y,
                            self.layout.sub_x,
                            self.layout.sub_y,
                            dx,
                            dy
                        )
                    }
                );
            }
            None => tracing::info!(
                "[Elgato] {} {}x{} -> whole frame (CAPTURE_ROI off or not applicable)",
                fourcc_to_string(format.fourcc),
                format.width,
                format.height
            ),
        }
    }

    /// Gather a planar region into `scratch` as a small contiguous frame in the
    /// same planar layout, which is what `cvtColor` needs.
    fn gather_planar(
        scratch: &mut Vec<u8>,
        bytes: &[u8],
        layout: &Layout,
        r: Region,
    ) -> Result<()> {
        let step = layout.step;
        let (rw, rh) = (r.w as usize, r.h as usize);
        let (rx, ry) = (r.x as usize, r.y as usize);
        let frame_h = layout.height as usize;

        scratch.clear();
        scratch.resize(rw * rh * 3 / 2, 0);
        let (y_dst, c_dst) = scratch.split_at_mut(rw * rh);

        let take = |offset: usize, len: usize| -> Result<&[u8]> {
            bytes.get(offset..offset + len).ok_or_else(|| {
                anyhow!(
                    "[Elgato] frame is {} bytes; region needs {}..{}",
                    bytes.len(),
                    offset,
                    offset + len
                )
            })
        };

        for row in 0..rh {
            y_dst[row * rw..][..rw].copy_from_slice(take((ry + row) * step + rx, rw)?);
        }

        match layout.packing {
            // One interleaved chroma plane: same row pitch as luma, and a luma
            // column maps to the same byte offset (two columns share a UV pair,
            // which is two bytes).
            Packing::SemiPlanar => {
                let uv_base = step * frame_h;
                for row in 0..rh / 2 {
                    c_dst[row * rw..][..rw]
                        .copy_from_slice(take(uv_base + (ry / 2 + row) * step + rx, rw)?);
                }
            }
            // Two half-pitch chroma planes. `cvtColor` reads them as one
            // contiguous block, so U rows go back to back, then V rows.
            Packing::Planar => {
                let c_step = step / 2;
                let c_rows = frame_h / 2;
                let u_base = step * frame_h;
                let v_base = u_base + c_step * c_rows;
                let (cw, ch) = (rw / 2, rh / 2);
                let (u_dst, v_dst) = c_dst.split_at_mut(cw * ch);
                for row in 0..ch {
                    u_dst[row * cw..][..cw]
                        .copy_from_slice(take(u_base + (ry / 2 + row) * c_step + rx / 2, cw)?);
                    v_dst[row * cw..][..cw]
                        .copy_from_slice(take(v_base + (ry / 2 + row) * c_step + rx / 2, cw)?);
                }
            }
            other => bail!("[Elgato] {:?} is not a planar layout", other),
        }
        Ok(())
    }

    /// Produce the region as BGR. `bytes` is the mapped DMA buffer.
    fn region_bgr(
        jpeg: &mut Option<RoiJpeg>,
        scratch: &mut Vec<u8>,
        bytes: &[u8],
        layout: &Layout,
        region: Region,
    ) -> Result<Mat> {
        // MJPEG has no addressable pixels, so libjpeg does the cropping and
        // writes BGR straight into the output frame.
        if layout.compressed() {
            let decoder = jpeg
                .as_mut()
                .ok_or_else(|| anyhow!("[Elgato] MJPEG region requested with no decoder"))?;
            let mut out = unsafe { Mat::new_rows_cols(region.h, region.w, CV_MAKETYPE(CV_8U, 3)) }?;
            if !out.is_continuous() {
                bail!("[Elgato] freshly allocated Mat is not continuous");
            }
            let stride = region.w as usize * 3;
            decoder.decode_bgr(
                bytes,
                crate::stream::roi_jpeg::Rect {
                    x: region.x,
                    y: region.y,
                    w: region.w,
                    h: region.h,
                },
                out.data_bytes_mut()?,
                stride,
            )?;
            return Ok(out);
        }

        let (aligned, dx, dy) = region.aligned(layout.sub_x, layout.sub_y);
        if aligned.x + aligned.w > layout.width || aligned.y + aligned.h > layout.height {
            bail!(
                "[Elgato] region {}x{}+{}+{} grows past the {}x{} frame once aligned to {}:{} \
                 chroma; move REGION_* inwards",
                region.w,
                region.h,
                region.x,
                region.y,
                layout.width,
                layout.height,
                layout.sub_x,
                layout.sub_y
            );
        }

        let src = match layout.packing {
            // Rows are already contiguous, so the region is a view with the
            // frame's own pitch — no copy at all.
            Packing::Packed { bytes_per_pixel } => {
                let offset = aligned.y as usize * layout.step
                    + aligned.x as usize * bytes_per_pixel as usize;
                let span = (aligned.h as usize - 1) * layout.step
                    + aligned.w as usize * bytes_per_pixel as usize;
                let window = bytes.get(offset..offset + span).ok_or_else(|| {
                    anyhow!(
                        "[Elgato] frame is {} bytes; region needs {}..{}",
                        bytes.len(),
                        offset,
                        offset + span
                    )
                })?;
                unsafe {
                    Mat::new_rows_cols_with_data_unsafe(
                        aligned.h,
                        aligned.w,
                        layout.mat_type,
                        window.as_ptr().cast::<c_void>().cast_mut(),
                        layout.step,
                    )
                }?
            }
            Packing::SemiPlanar | Packing::Planar => {
                Self::gather_planar(scratch, bytes, layout, aligned)?;
                unsafe {
                    Mat::new_rows_cols_with_data_unsafe(
                        aligned.h * 3 / 2,
                        aligned.w,
                        layout.mat_type,
                        scratch.as_ptr().cast::<c_void>().cast_mut(),
                        aligned.w as usize,
                    )
                }?
            }
            Packing::Jpeg => unreachable!("handled above"),
        };

        let mut bgr = Mat::default();
        match layout.convert {
            Some(code) => opencv::imgproc::cvt_color_def(&src, &mut bgr, code)?,
            None => src.copy_to(&mut bgr)?,
        }
        if aligned == region {
            return Ok(bgr);
        }
        // The chroma grid forced a wider rect than asked for; hand back exactly
        // the requested one so downstream geometry stays as configured.
        Ok(Mat::roi(&bgr, Rect::new(dx, dy, region.w, region.h))?.clone_pointee())
    }

    /// Whole-frame output, the pre-`CAPTURE_ROI` behaviour.
    fn whole_frame(
        bytes: &[u8],
        layout: &Layout,
        output: OutputFormat,
        sequence: u32,
    ) -> Result<Mat> {
        if layout.compressed() {
            // One decode pass; imdecode allocates the output itself.
            let src = unsafe {
                Mat::new_rows_cols_with_data_unsafe(
                    1,
                    bytes.len() as i32,
                    layout.mat_type,
                    bytes.as_ptr().cast::<c_void>().cast_mut(),
                    bytes.len(),
                )
            }?;
            if output == OutputFormat::Raw {
                let mut dst = Mat::default();
                src.copy_to(&mut dst)?;
                return Ok(dst);
            }
            let dst = opencv::imgcodecs::imdecode(
                &src,
                opencv::imgcodecs::ImreadModes::IMREAD_COLOR as i32,
            )?;
            if dst.empty() {
                bail!("[Elgato] MJPEG frame {} failed to decode", sequence);
            }
            return Ok(dst);
        }

        let needed = layout.step * layout.rows as usize;
        if bytes.len() < needed {
            bail!(
                "[Elgato] frame {} is {} bytes, need {} for {}x{} {} at pitch {}",
                sequence,
                bytes.len(),
                needed,
                layout.cols,
                layout.rows,
                layout.mat_type,
                layout.step
            );
        }

        // Zero-copy view over the mapped DMA buffer.
        let src = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                layout.rows,
                layout.cols,
                layout.mat_type,
                bytes.as_ptr().cast::<c_void>().cast_mut(),
                layout.step,
            )
        }?;

        let mut dst = Mat::default();
        match (output, layout.convert) {
            (OutputFormat::Raw, _) | (OutputFormat::Bgr, None) => src.copy_to(&mut dst)?,
            (OutputFormat::Bgr, Some(code)) => {
                opencv::imgproc::cvt_color_def(&src, &mut dst, code)?
            }
        }
        Ok(dst)
    }
}

impl StreamCapture for Elgato {
    fn capture(&mut self) -> Result<Mat> {
        let started = Instant::now();
        // Split the borrows: `with_frame` needs `&mut cap` while the callback
        // needs the decoder and the gather buffer.
        let Self {
            cap,
            output,
            layout,
            region,
            jpeg,
            scratch,
            ..
        } = self;
        let output = *output;
        let layout = *layout;
        let region = *region;

        let mut waited_for = Duration::ZERO;
        let mut frame_age = None;
        let (mat, convert_for) = cap.with_frame(|bytes, meta, waited| {
            waited_for = waited;
            if meta.timestamp_monotonic {
                frame_age = monotonic_now().checked_sub(meta.timestamp);
            }
            let convert = Instant::now();
            if bytes.is_empty() {
                bail!("[Elgato] driver returned an empty frame");
            }
            let mat = match region {
                Some(r) => Self::region_bgr(jpeg, scratch, bytes, &layout, r)?,
                None => Self::whole_frame(bytes, &layout, output, meta.sequence)?,
            };
            Ok((mat, convert.elapsed()))
        })?;

        // `waited` is time spent blocked on the sensor, which is bounded below
        // by the frame interval and is not work we can optimise away; `convert`
        // is. `age` is what actually matters: how stale the frame already was
        // when the driver handed it over.
        tracing::debug!(
            "[Elgato] total {:?} = wait {:?} + convert {:?} | frame age {} | {}x{}",
            started.elapsed(),
            waited_for,
            convert_for,
            frame_age.map_or_else(|| "n/a".to_string(), |a| format!("{a:?}")),
            mat.cols(),
            mat.rows()
        );
        Ok(mat)
    }

    fn stream_info(&self) -> Result<StreamInfo> {
        let format = self.cap.format();
        Ok(StreamInfo {
            width: format.width,
            height: format.height,
            fps: self.cap.fps(),
        })
    }

    fn reconnect(&mut self) -> Result<()> {
        self.cap
            .restart()
            .map_err(|e| anyhow!("[Elgato] restart failed: {}", e))?;
        // The card may come back on a different mode after a signal drop.
        self.layout = layout_for(&self.cap.format())?;
        self.apply_region()?;
        tracing::info!(
            "[Elgato] reconnected to {} at {}x{} {}",
            self.cap.config().device,
            self.cap.format().width,
            self.cap.format().height,
            fourcc_to_string(self.cap.format().fourcc)
        );
        self.log_mode();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::MatTraitConstManual;

    fn nv12_format(w: u32, h: u32) -> Format {
        Format {
            width: w,
            height: h,
            fourcc: v4l2::V4L2_PIX_FMT_NV12,
            bytes_per_line: w,
            size_image: w * h * 3 / 2,
        }
    }

    #[test]
    fn an_aligned_region_needs_no_widening() {
        let r = Region {
            x: 864,
            y: 444,
            w: 192,
            h: 192,
        };
        let (aligned, dx, dy) = r.aligned(2, 2);
        assert_eq!(aligned, r);
        assert_eq!((dx, dy), (0, 0));
    }

    #[test]
    fn an_odd_region_grows_to_the_chroma_grid() {
        let r = Region {
            x: 865,
            y: 443,
            w: 191,
            h: 191,
        };
        let (aligned, dx, dy) = r.aligned(2, 2);
        assert_eq!((dx, dy), (1, 1));
        assert_eq!(aligned.x, 864);
        assert_eq!(aligned.y, 442);
        // Must cover the request on both edges and land on the grid.
        assert!(aligned.x + aligned.w >= r.x + r.w);
        assert!(aligned.y + aligned.h >= r.y + r.h);
        assert_eq!(aligned.w % 2, 0);
        assert_eq!(aligned.h % 2, 0);
    }

    /// The gathered buffer has to be byte-identical to cropping the frame after
    /// a whole-frame conversion, which is the property the whole ROI path rests
    /// on.
    #[test]
    fn a_gathered_nv12_region_matches_the_whole_frame_crop() {
        let (w, h) = (64u32, 48u32);
        let layout = layout_for(&nv12_format(w, h)).unwrap();
        // Distinct luma and chroma patterns, so a plane or pitch mix-up shows.
        let mut frame = vec![0u8; (w * h * 3 / 2) as usize];
        for row in 0..h as usize {
            for col in 0..w as usize {
                frame[row * w as usize + col] = ((row * 7 + col * 3) % 256) as u8;
            }
        }
        let uv = (w * h) as usize;
        for row in 0..(h / 2) as usize {
            for col in 0..w as usize {
                frame[uv + row * w as usize + col] = ((row * 11 + col * 5 + 128) % 256) as u8;
            }
        }

        let region = Region {
            x: 16,
            y: 8,
            w: 16,
            h: 16,
        };
        let mut scratch = Vec::new();
        let mut jpeg = None;
        let got = Elgato::region_bgr(&mut jpeg, &mut scratch, &frame, &layout, region).unwrap();
        assert_eq!((got.cols(), got.rows()), (region.w, region.h));

        let whole = Elgato::whole_frame(&frame, &layout, OutputFormat::Bgr, 0).unwrap();
        let want = Mat::roi(&whole, Rect::new(region.x, region.y, region.w, region.h))
            .unwrap()
            .clone_pointee();

        let a = got.data_bytes().unwrap();
        let b = want.data_bytes().unwrap();
        assert_eq!(a.len(), b.len());
        // COLOR_YUV2BGR_NV12 replicates one chroma pair across each 2x2 luma
        // block rather than interpolating, so a gather that lands on the chroma
        // grid has to be bit-identical — not merely close.
        assert_eq!(a, b, "gathered NV12 region is not identical to the crop");
    }

    #[test]
    fn a_gathered_i420_region_matches_the_whole_frame_crop() {
        let (w, h) = (64u32, 48u32);
        let format = Format {
            fourcc: v4l2::V4L2_PIX_FMT_YUV420,
            ..nv12_format(w, h)
        };
        let layout = layout_for(&format).unwrap();
        let mut frame = vec![0u8; (w * h * 3 / 2) as usize];
        for (i, b) in frame.iter_mut().enumerate() {
            *b = ((i * 13) % 256) as u8;
        }
        let region = Region {
            x: 8,
            y: 16,
            w: 24,
            h: 16,
        };
        let mut scratch = Vec::new();
        let mut jpeg = None;
        let got = Elgato::region_bgr(&mut jpeg, &mut scratch, &frame, &layout, region).unwrap();
        let whole = Elgato::whole_frame(&frame, &layout, OutputFormat::Bgr, 0).unwrap();
        let want = Mat::roi(&whole, Rect::new(region.x, region.y, region.w, region.h))
            .unwrap()
            .clone_pointee();
        assert_eq!(
            got.data_bytes().unwrap(),
            want.data_bytes().unwrap(),
            "gathered I420 region is not identical to the crop"
        );
    }

    #[test]
    fn a_packed_region_is_a_view_of_the_frame() {
        let (w, h) = (32u32, 16u32);
        let format = Format {
            width: w,
            height: h,
            fourcc: v4l2::V4L2_PIX_FMT_BGR24,
            bytes_per_line: w * 3,
            size_image: w * h * 3,
        };
        let layout = layout_for(&format).unwrap();
        let mut frame = vec![0u8; (w * h * 3) as usize];
        for (i, b) in frame.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }
        let region = Region {
            x: 5,
            y: 3,
            w: 8,
            h: 4,
        };
        let mut scratch = Vec::new();
        let mut jpeg = None;
        let got = Elgato::region_bgr(&mut jpeg, &mut scratch, &frame, &layout, region).unwrap();
        assert_eq!((got.cols(), got.rows()), (region.w, region.h));
        let data = got.data_bytes().unwrap();
        for row in 0..region.h as usize {
            let src = (region.y as usize + row) * (w as usize * 3) + region.x as usize * 3;
            let dst = row * region.w as usize * 3;
            assert_eq!(
                &data[dst..dst + region.w as usize * 3],
                &frame[src..src + region.w as usize * 3],
                "row {} differs",
                row
            );
        }
    }
}
