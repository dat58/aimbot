pub mod elgato;
mod ndi;
#[cfg(feature = "ndi4")]
mod ndi4;
#[cfg(feature = "ndi6")]
mod ndi6;
pub mod roi_jpeg;
mod udp;
pub mod v4l2;
pub use elgato::Elgato;
pub use ndi::*;
#[cfg(feature = "ndi4")]
pub use ndi4::*;
#[cfg(feature = "ndi6")]
pub use ndi6::*;
pub use udp::*;

use crate::config::Config;
use anyhow::Result;
use crossbeam::queue::ArrayQueue;
use opencv::core::Mat;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Debug)]
pub struct StreamInfo {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

pub trait StreamCapture: Send {
    fn capture(&mut self) -> Result<Mat>;
    fn stream_info(&self) -> Result<StreamInfo>;
    fn reconnect(&mut self) -> Result<()>;
}

/// Open the frame source `SOURCE_STREAM` names: NDI for `ndi://`, the capture
/// card for `elgato://` or `v4l2://`, and FFmpeg for anything else.
pub fn open_source(config: &Config) -> Result<Box<dyn StreamCapture>> {
    Ok(if config.source_stream.starts_with("ndi://") {
        let source_stream = config
            .source_stream
            .trim()
            .split(',')
            .map(|source| source.trim_start_matches("ndi://"))
            .collect::<Vec<&str>>()
            .join(",");
        Box::new(NDI::new(
            &source_stream,
            config.ndi_source_name.clone(),
            config.ndi_timeout,
        )?)
    } else if let Some(device) = config
        .source_stream
        .trim()
        .strip_prefix("elgato://")
        .or_else(|| config.source_stream.trim().strip_prefix("v4l2://"))
    {
        // `elgato://` on its own falls back to CAPTURE_DEVICE.
        let device = (!device.is_empty()).then_some(device);
        Box::new(Elgato::new(config, device)?)
    } else {
        Box::new(UDP::new(config.source_stream.as_str())?)
    })
}

pub fn handle_capture(
    mut cap: Box<dyn StreamCapture>,
    queue: Arc<ArrayQueue<Mat>>,
    retry_time: usize,
    retry_interval: Duration,
) {
    loop {
        let now = Instant::now();
        match cap.capture() {
            Ok(mat) => {
                tracing::debug!("[Stream] captured took: {:?}", now.elapsed());
                queue.force_push(mat);
            }
            Err(e) => {
                tracing::error!("[Stream] {}, try reconnecting", e);
                let mut reconnect_success = false;
                for _ in 0..retry_time {
                    if cap.reconnect().is_ok() {
                        reconnect_success = true;
                        break;
                    }
                    std::thread::sleep(retry_interval);
                }
                if reconnect_success {
                    continue;
                } else {
                    tracing::error!("[Stream] reconnect to the stream timed out, break the loop.");
                    break;
                }
            }
        }
    }
}
