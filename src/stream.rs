use anyhow::{Result, bail};
use crossbeam::queue::ArrayQueue;
use opencv::{
    core::Mat,
    videoio::{
        CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, VideoCapture, VideoCaptureTrait,
        VideoCaptureTraitConst,
    },
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

pub struct StreamInfo {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

pub trait StreamCapture {
    fn capture(&mut self) -> Result<Mat>;
    fn stream_info(&self) -> Result<StreamInfo>;
    fn reconnect(&mut self) -> Result<()>;
}

pub struct UDP {
    cap: VideoCapture,
    url: String,
}

impl UDP {
    pub fn new(url: &str) -> Result<Self> {
        let cap = VideoCapture::from_file_def(url)?;
        if !cap.is_opened()? {
            bail!("[Stream] unable to open capture stream at: {}", url);
        } else {
            tracing::info!(
                "[Stream] established connection to stream at {} successfully.",
                url
            );
        }
        Ok(Self {
            cap,
            url: url.to_string(),
        })
    }
}

impl StreamCapture for UDP {
    fn capture(&mut self) -> Result<Mat> {
        let mut frame = Mat::default();
        let ret = self.cap.read(&mut frame)?;
        if ret {
            Ok(frame)
        } else {
            let error = "[Stream] unable to read from the stream.";
            tracing::error!("{}", error);
            bail!(error);
        }
    }

    fn stream_info(&self) -> Result<StreamInfo> {
        let fps = self.cap.get(CAP_PROP_FPS)? as u32;
        let width = self.cap.get(CAP_PROP_FRAME_WIDTH)? as u32;
        let height = self.cap.get(CAP_PROP_FRAME_HEIGHT)? as u32;
        Ok(StreamInfo { width, height, fps })
    }

    fn reconnect(&mut self) -> Result<()> {
        let cap = VideoCapture::from_file_def(&self.url)?;
        if !cap.is_opened()? {
            let error = "[Stream] unable to reconnect to the stream.";
            tracing::error!("{}", error);
            bail!(error);
        } else {
            tracing::info!("[Stream] reconnect to stream at {} successfully.", self.url);
            self.cap = cap;
        }
        Ok(())
    }
}

pub fn handle_capture(
    mut cap: Box<dyn StreamCapture>,
    queue: Arc<ArrayQueue<Mat>>,
    retry_time: usize,
    retry_interval: Duration,
) {
    loop {
        let now = Instant::now();
        if let Ok(mat) = cap.capture() {
            tracing::debug!("[Stream] captured took: {:?}", now.elapsed());
            queue.force_push(mat);
        } else {
            tracing::warn!("[Stream] unable to capture from the stream, try reconnecting...");
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
