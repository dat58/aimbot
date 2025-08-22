#[cfg(feature = "ndi4")]
use crate::stream::ndi4;
#[cfg(feature = "ndi6")]
use crate::stream::ndi6;
use crate::stream::{StreamCapture, StreamInfo};
use anyhow::Result;
use opencv::core::Mat;
use std::time::Duration;

pub struct NDI {
    #[cfg(feature = "ndi4")]
    inner: ndi4::NDI4,
    #[cfg(feature = "ndi6")]
    inner: ndi6::NDI6,
}

impl NDI {
    pub fn new(extra_ips: &str, source_name: Option<String>, timeout: Duration) -> Result<Self> {
        #[cfg(feature = "ndi4")]
        let inner = ndi4::NDI4::new(extra_ips, source_name, timeout)?;
        #[cfg(feature = "ndi6")]
        let inner = ndi6::NDI6::new(extra_ips, source_name, timeout)?;
        Ok(Self { inner })
    }
}

impl StreamCapture for NDI {
    fn capture(&mut self) -> Result<Mat> {
        self.inner.capture()
    }

    fn stream_info(&self) -> Result<StreamInfo> {
        self.inner.stream_info()
    }

    fn reconnect(&mut self) -> Result<()> {
        self.inner.reconnect()
    }
}
