use anyhow::{Result, bail};
use crossbeam::queue::ArrayQueue;
use opencv::{
    core::Mat,
    videoio::{
        CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, VideoCapture, VideoCaptureTrait,
        VideoCaptureTraitConst,
    },
};
use std::{sync::Arc, time::Instant};

pub struct StreamInfo {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

pub trait StreamCapture {
    fn capture(&mut self) -> Result<Mat>;
    fn stream_info(&self) -> Result<StreamInfo>;
}

pub struct UDP(VideoCapture);

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
        Ok(Self(cap))
    }
}

impl StreamCapture for UDP {
    fn capture(&mut self) -> Result<Mat> {
        let mut frame = Mat::default();
        let ret = self.0.read(&mut frame)?;
        if ret {
            Ok(frame)
        } else {
            let error = "[Stream] unable to read from stream.";
            tracing::error!(error);
            bail!(error);
        }
    }

    fn stream_info(&self) -> Result<StreamInfo> {
        let fps = self.0.get(CAP_PROP_FPS)? as u32;
        let width = self.0.get(CAP_PROP_FRAME_WIDTH)? as u32;
        let height = self.0.get(CAP_PROP_FRAME_HEIGHT)? as u32;
        Ok(StreamInfo { width, height, fps })
    }
}

pub fn handle_capture(mut cap: Box<dyn StreamCapture>, queue: Arc<ArrayQueue<Mat>>) {
    loop {
        let now = Instant::now();
        if let Ok(mat) = cap.capture() {
            tracing::debug!("[Stream] captured took: {:?}", now.elapsed());
            queue.force_push(mat);
        } else {
            tracing::debug!("[Stream] no captured & no pushed took: {:?}", now.elapsed());
            break;
        }
    }
}
