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
use crossbeam::queue::{ArrayQueue, SegQueue};
use opencv::core::Mat;
use std::{
    sync::Arc,
    thread,
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
    let first_delay = std::env::var("CC_DELAY_MS")
        .unwrap_or("50".to_string())
        .parse::<u64>()
        .expect("CC_DELAY must be a number");
    let fake_cc_queue = Arc::new(SegQueue::<(Mat, Duration)>::new());
    let cc_queue = fake_cc_queue.clone();
    thread::spawn(move || {
        loop {
            if let Some((mat, duration)) = cc_queue.pop() {
                thread::sleep(duration);
                queue.force_push(mat);
            }
        }
    });
    loop {
        let now = Instant::now();
        match cap.capture() {
            Ok(mat) => {
                tracing::debug!("[Stream] captured took: {:?}", now.elapsed());
                if fake_cc_queue.is_empty() {
                    tracing::warn!("[Stream] EMPTY CC QUEUE");
                    fake_cc_queue.push((mat, Duration::from_millis(first_delay)));
                } else {
                    fake_cc_queue.push((mat, now.elapsed()));
                }
            }
            Err(e) => {
                tracing::error!("[Stream] {}, try reconnecting", e);
                let mut reconnect_success = false;
                for _ in 0..retry_time {
                    if cap.reconnect().is_ok() {
                        reconnect_success = true;
                        break;
                    }
                    thread::sleep(retry_interval);
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
