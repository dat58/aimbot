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
//! 3. wrap the mapped bytes in a `Mat` with the driver's own row pitch — no
//!    copy, no allocation;
//! 4. one pass to produce the output frame: `cvtColor` for the packed YUV
//!    formats, a plain copy in `raw` mode.
//!
//! Step 4 is the only pass over the pixels. `CAPTURE_OUTPUT=raw` skips even the
//! colour conversion and hands back the untouched device bytes, which is the
//! lowest-latency mode available but leaves the consumer to interpret the
//! layout.

use crate::config::Config;
use crate::stream::v4l2::{self, Capture, Format, fourcc_to_string};
use crate::stream::{StreamCapture, StreamInfo};
use anyhow::{Result, anyhow, bail};
use opencv::core::{CV_8U, CV_MAKETYPE, Mat, MatTraitConst};
use opencv::imgproc::ColorConversionCodes;
use std::ffi::c_void;
use std::time::Instant;

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

/// How the device bytes map onto a `Mat` before conversion.
#[derive(Debug, Clone, Copy)]
struct Layout {
    rows: i32,
    cols: i32,
    mat_type: i32,
    step: usize,
    /// `cvtColor` code, or `None` when the bytes are already BGR.
    convert: Option<i32>,
    /// Compressed payload that has to go through `imdecode` instead.
    compressed: bool,
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

    let layout = match format.fourcc {
        v4l2::V4L2_PIX_FMT_NV12 => Layout {
            rows: h * 3 / 2,
            cols: w,
            mat_type: CV_MAKETYPE(CV_8U, 1),
            step: pitch(1),
            convert: Some(ColorConversionCodes::COLOR_YUV2BGR_NV12 as i32),
            compressed: false,
        },
        v4l2::V4L2_PIX_FMT_NV21 => Layout {
            rows: h * 3 / 2,
            cols: w,
            mat_type: CV_MAKETYPE(CV_8U, 1),
            step: pitch(1),
            convert: Some(ColorConversionCodes::COLOR_YUV2BGR_NV21 as i32),
            compressed: false,
        },
        v4l2::V4L2_PIX_FMT_YUV420 => Layout {
            rows: h * 3 / 2,
            cols: w,
            mat_type: CV_MAKETYPE(CV_8U, 1),
            step: pitch(1),
            // I420 and IYUV are the same code; opencv-rust keeps only the
            // canonical name because the aliases share a discriminant.
            convert: Some(ColorConversionCodes::COLOR_YUV2BGR_IYUV as i32),
            compressed: false,
        },
        v4l2::V4L2_PIX_FMT_YUYV => Layout {
            rows: h,
            cols: w,
            mat_type: CV_MAKETYPE(CV_8U, 2),
            step: pitch(2),
            convert: Some(ColorConversionCodes::COLOR_YUV2BGR_YUY2 as i32),
            compressed: false,
        },
        v4l2::V4L2_PIX_FMT_UYVY => Layout {
            rows: h,
            cols: w,
            mat_type: CV_MAKETYPE(CV_8U, 2),
            step: pitch(2),
            convert: Some(ColorConversionCodes::COLOR_YUV2BGR_UYVY as i32),
            compressed: false,
        },
        v4l2::V4L2_PIX_FMT_BGR24 => Layout {
            rows: h,
            cols: w,
            mat_type: CV_MAKETYPE(CV_8U, 3),
            step: pitch(3),
            convert: None,
            compressed: false,
        },
        v4l2::V4L2_PIX_FMT_RGB24 => Layout {
            rows: h,
            cols: w,
            mat_type: CV_MAKETYPE(CV_8U, 3),
            step: pitch(3),
            // COLOR_RGB2BGR is an alias of COLOR_BGR2RGB — the same symmetric
            // 3-channel swap, and only the canonical name is generated.
            convert: Some(ColorConversionCodes::COLOR_BGR2RGB as i32),
            compressed: false,
        },
        v4l2::V4L2_PIX_FMT_ABGR32 => Layout {
            rows: h,
            cols: w,
            mat_type: CV_MAKETYPE(CV_8U, 4),
            step: pitch(4),
            convert: Some(ColorConversionCodes::COLOR_BGRA2BGR as i32),
            compressed: false,
        },
        v4l2::V4L2_PIX_FMT_MJPEG => Layout {
            rows: 1,
            cols: 0, // filled in per frame from `bytesused`
            mat_type: CV_MAKETYPE(CV_8U, 1),
            step: 0,
            convert: None,
            compressed: true,
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

pub struct Elgato {
    cap: Capture,
    output: OutputFormat,
    layout: Layout,
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
        Ok(Self {
            cap,
            output,
            layout,
        })
    }

    pub fn format(&self) -> Format {
        self.cap.format()
    }
}

impl StreamCapture for Elgato {
    fn capture(&mut self) -> Result<Mat> {
        let output = self.output;
        let layout = self.layout;
        let started = Instant::now();

        let mat = self.cap.with_frame(|bytes, meta| {
            if bytes.is_empty() {
                bail!("[Elgato] driver returned an empty frame");
            }

            if layout.compressed {
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
                    bail!("[Elgato] MJPEG frame {} failed to decode", meta.sequence);
                }
                return Ok(dst);
            }

            let needed = layout.step * layout.rows as usize;
            if bytes.len() < needed {
                bail!(
                    "[Elgato] frame {} is {} bytes, need {} for {}x{} {} at pitch {}",
                    meta.sequence,
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
        })?;

        tracing::debug!(
            "[Elgato] frame ready in {:?} ({}x{})",
            started.elapsed(),
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
        tracing::info!(
            "[Elgato] reconnected to {} at {}x{} {}",
            self.cap.config().device,
            self.cap.format().width,
            self.cap.format().height,
            fourcc_to_string(self.cap.format().fourcc)
        );
        Ok(())
    }
}
