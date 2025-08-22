use crate::stream::{StreamCapture, StreamInfo};
use anyhow::{Result, bail};
use opencv::{
    core::{CV_8UC4, Mat, Mat_AUTO_STEP},
    imgproc::{self, COLOR_RGBA2BGR},
};
use std::{os::raw::c_void, time::Duration};

pub struct NDI4 {
    extra_ips: String,
    source_name: Option<String>,
    recv: ndi::recv::Recv,
    timeout: Duration,
}

impl NDI4 {
    // Create new instance of NDI with extra IPs
    // Ex: "192.168.1.2,192.168.1.20"
    pub fn new(extra_ips: &str, source_name: Option<String>, timeout: Duration) -> Result<Self> {
        let find = ndi::FindBuilder::new()
            .extra_ips(extra_ips.to_string())
            .build()?;
        let sources = find.current_sources(timeout.as_millis() * 2)?;
        if sources.is_empty() {
            let error = "[Stream] unable to find sources.";
            tracing::error!("{}", error);
            bail!(error);
        }
        let source_index = match source_name {
            Some(ref source_name) => {
                let mut index = 0;
                for (i, source) in sources.iter().enumerate() {
                    if source_name == &source.get_name() {
                        index = i;
                        break;
                    }
                }
                index
            }
            None => 0,
        };
        let mut recv = ndi::RecvBuilder::new()
            .color_format(ndi::RecvColorFormat::RGBX_RGBA)
            .bandwidth(ndi::RecvBandwidth::Highest)
            .ndi_recv_name(sources[source_index].get_name())
            .allow_video_fields(false)
            .build()?;
        recv.connect(&sources[source_index]);

        let mut video_data = None;
        let mut count = 0usize;
        let mut connected = false;
        while count < 10 {
            let response = recv.capture_video(&mut video_data, timeout.as_millis() as u32);
            if response == ndi::FrameType::Video {
                connected = true;
                break;
            }
            count += 1;
            tracing::debug!("In a loop...{}", count);
        }
        if !connected {
            bail!("Unable to connect to the NDI.");
        }

        tracing::info!(
            "Connected to NDI device {}",
            sources[source_index].get_name()
        );

        Ok(Self {
            extra_ips: extra_ips.to_string(),
            source_name: source_name.and_then(|s| Some(s.to_string())),
            recv,
            timeout,
        })
    }

    pub fn recv_video(&self, timeout: Option<Duration>) -> Result<ndi::VideoData> {
        let mut video_data = None;
        self.recv.capture_video(
            &mut video_data,
            timeout.unwrap_or(self.timeout).as_millis() as u32,
        );
        match video_data {
            Some(video_data) if !video_data.p_data().is_null() => Ok(video_data),
            _ => bail!("Failed to capture video data"),
        }
    }

    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }
}

impl StreamCapture for NDI4 {
    #[inline]
    fn capture(&mut self) -> Result<Mat> {
        let video = self.recv_video(None)?;
        let data = video.p_data();
        let bgra = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                video.height() as i32,
                video.width() as i32,
                CV_8UC4, // Use 4 channels for the BGRA data
                data as *mut c_void,
                Mat_AUTO_STEP,
            )?
        };
        let mut bgr = Mat::default();
        imgproc::cvt_color(&bgra, &mut bgr, COLOR_RGBA2BGR, 0)?;
        Ok(bgr)
    }

    fn stream_info(&self) -> Result<StreamInfo> {
        let video = self.recv_video(None)?;
        Ok(StreamInfo {
            width: video.width(),
            height: video.height(),
            fps: video.frame_rate_n(),
        })
    }

    fn reconnect(&mut self) -> Result<()> {
        let _self = Self::new(&self.extra_ips, self.source_name.clone(), self.timeout)?;
        self.recv = _self.recv;
        Ok(())
    }
}
