use crate::stream::{StreamCapture, StreamInfo};
use anyhow::{Result, bail};
use opencv::{
    core::Mat,
    videoio::{
        CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, VideoCapture, VideoCaptureTrait,
        VideoCaptureTraitConst,
    },
};

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
