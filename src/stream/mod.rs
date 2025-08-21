mod ndi;
mod udp;
pub use ndi::*;
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
) {
    loop {
        let now = Instant::now();
        match cap.capture() {
            Ok(mat) => {
                tracing::debug!("[Stream] captured took: {:?}", now.elapsed());
                queue.force_push(mat);
            }
            Err(e) => {
                tracing::error!("[Stream] failed to capture next frame: {}, try reconnecting", e);
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
