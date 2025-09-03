use crate::stream::{StreamCapture, StreamInfo};
use anyhow::{Result, bail};
use grafton_ndi as ndi;
use lazy_static::lazy_static;
use opencv::{
    core::{CV_8UC4, Mat, Mat_AUTO_STEP},
    imgproc::{self, COLOR_RGBA2BGR},
};
use std::{os::raw::c_void, time::Duration};

lazy_static! {
    pub static ref NDI_CORE: ndi::NDI = ndi::NDI::new().expect("Failed to initialize NDI");
}

pub struct NDI6 {
    source: ndi::Source,
    timeout: Duration,
    recv: ndi::Receiver<'static>,
}

impl NDI6 {
    pub fn new(extra_ips: &str, source_name: Option<String>, timeout: Duration) -> Result<Self> {
        let wait_time = timeout.as_millis() as u32;

        let source = {
            // Find sources on the network
            let finder_options = ndi::FinderOptions::builder()
                .extra_ips(extra_ips)
                .groups("Public,Private")
                .show_local_sources(true)
                .build();
            let finder = ndi::Finder::new(&NDI_CORE, &finder_options)?;

            // Wait for sources
            finder.wait_for_sources(wait_time);
            let sources = finder.get_sources(wait_time)?;
            tracing::info!("All sources: {:?}", sources);

            if sources.is_empty() {
                let error = "[Stream] unable to find sources.";
                tracing::error!("{}", error);
                bail!(error);
            }
            let source_index = match source_name {
                Some(ref source_name) => {
                    let mut index = 0;
                    for (i, source) in sources.iter().enumerate() {
                        if source_name == &source.name {
                            index = i;
                            break;
                        }
                    }
                    index
                }
                None => 0,
            };
            sources[source_index].clone()
        };
        let receiver = ndi::ReceiverOptions::builder(source.clone())
            .color(ndi::ReceiverColorFormat::RGBX_RGBA)
            .bandwidth(ndi::ReceiverBandwidth::Highest)
            .name(&source.name)
            .build(&NDI_CORE)?;

        let mut count = 0usize;
        let mut connected = false;
        while count < 10 {
            if let Some(video) = receiver.capture_video(timeout.as_millis() as u32)? {
                if video.data.len() > 0 {
                    connected = true;
                    break;
                }
            }
            count += 1;
            tracing::debug!("In a loop...{}", count);
        }
        if !connected {
            bail!("Unable to connect to the NDI.");
        }

        tracing::info!("[NDI] connected to source: {}", source);
        Ok(Self {
            recv: receiver,
            source,
            timeout,
        })
    }

    pub fn recv_video(&self) -> Result<ndi::VideoFrame<'_>> {
        for _ in 0..2 {
            let response = self.recv.capture_video(self.timeout.as_millis() as u32)?;
            match response {
                Some(video) if video.data.len() > 0 => return Ok(video),
                _ => {}
            }
        }
        bail!("NDI received a response without video.");
    }

    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }
}

impl StreamCapture for NDI6 {
    #[inline]
    fn capture(&mut self) -> Result<Mat> {
        let video = self.recv_video()?;
        let data = video.data.as_ptr();
        let bgra = unsafe {
            Mat::new_rows_cols_with_data_unsafe(
                video.height,
                video.width,
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
        let video = self.recv_video()?;
        Ok(StreamInfo {
            width: video.width as u32,
            height: video.height as u32,
            fps: video.frame_rate_n as u32,
        })
    }

    fn reconnect(&mut self) -> Result<()> {
        let receiver = ndi::ReceiverOptions::builder(self.source.clone())
            .color(ndi::ReceiverColorFormat::RGBX_RGBA)
            .bandwidth(ndi::ReceiverBandwidth::Highest)
            .name(&self.source.name)
            .build(&NDI_CORE)?;
        tracing::info!("[NDI] reconnected to source: {}", self.source);
        self.recv = receiver;
        Ok(())
    }
}
