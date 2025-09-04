mod ndi;
#[cfg(feature = "ndi4")]
mod ndi4;
#[cfg(feature = "ndi6")]
mod ndi6;
mod udp;
pub use ndi::*;
#[cfg(feature = "ndi4")]
pub use ndi4::*;
#[cfg(feature = "ndi6")]
pub use ndi6::*;
pub use udp::*;

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

pub fn handle_capture(
    mut cap: Box<dyn StreamCapture>,
    queue: Arc<ArrayQueue<Mat>>,
    retry_time: usize,
    retry_interval: Duration,
    sleep_interval: Option<Duration>,
) {
    loop {
        let now = Instant::now();
        match cap.capture() {
            Ok(mat) => {
                tracing::debug!("[Stream] captured took: {:?}", now.elapsed());
                queue.force_push(mat);
                if let Some(sleep_interval) = sleep_interval {
                    std::thread::sleep(sleep_interval);
                }
            }
            Err(e) => {
                tracing::error!("[Stream] {}, try reconnecting...", e);
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
