//! Safe wrapper over the libjpeg-turbo shim in `roi_jpeg.c`.
//!
//! One [`RoiJpeg`] owns one libjpeg decompressor and reuses it for every frame,
//! which is where most of the speed comes from: `imdecode` builds and tears one
//! down per call. Every method takes `&mut self` because the decompressor is
//! stateful, so the type is `Send` but deliberately not `Sync`.

use anyhow::{Result, anyhow, bail};
use std::ffi::{CStr, c_char, c_int, c_uchar, c_void};

unsafe extern "C" {
    fn roi_jpeg_new() -> *mut c_void;
    fn roi_jpeg_free(rj: *mut c_void);
    fn roi_jpeg_error(rj: *const c_void) -> *const c_char;
    fn roi_jpeg_size(
        rj: *mut c_void,
        data: *const c_uchar,
        len: usize,
        width: *mut c_int,
        height: *mut c_int,
    ) -> c_int;
    fn roi_jpeg_decode_bgr(
        rj: *mut c_void,
        data: *const c_uchar,
        len: usize,
        x: c_int,
        y: c_int,
        w: c_int,
        h: c_int,
        out: *mut c_uchar,
        out_stride: usize,
    ) -> c_int;
}

/// A rect in frame pixels. Deliberately not `opencv::core::Rect`: this module
/// talks to libjpeg and has no reason to depend on OpenCV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

pub struct RoiJpeg {
    raw: *mut c_void,
}

// The decompressor is reached only through `&mut self` and the pointer is owned
// exclusively by this struct, so it can move to the capture thread.
unsafe impl Send for RoiJpeg {}

impl RoiJpeg {
    pub fn new() -> Result<Self> {
        let raw = unsafe { roi_jpeg_new() };
        if raw.is_null() {
            bail!("[RoiJpeg] could not create a libjpeg decompressor");
        }
        Ok(Self { raw })
    }

    fn last_error(&self) -> String {
        let msg = unsafe { roi_jpeg_error(self.raw) };
        if msg.is_null() {
            return String::from("unknown libjpeg error");
        }
        unsafe { CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned()
    }

    /// Frame dimensions from the JPEG header, without decoding any pixels.
    pub fn size(&mut self, data: &[u8]) -> Result<(i32, i32)> {
        let mut w: c_int = 0;
        let mut h: c_int = 0;
        let rc = unsafe { roi_jpeg_size(self.raw, data.as_ptr(), data.len(), &mut w, &mut h) };
        match rc {
            0 => Ok((w, h)),
            -2 => Err(anyhow!("[RoiJpeg] bad arguments to roi_jpeg_size")),
            _ => Err(anyhow!("[RoiJpeg] header: {}", self.last_error())),
        }
    }

    /// Decode `rect` of `data` as BGR into `out`, which must hold `rect.h` rows
    /// of `stride` bytes with `stride >= rect.w * 3`.
    pub fn decode_bgr(
        &mut self,
        data: &[u8],
        rect: Rect,
        out: &mut [u8],
        stride: usize,
    ) -> Result<()> {
        let Rect { x, y, w, h } = rect;
        if w <= 0 || h <= 0 {
            bail!("[RoiJpeg] region must be non-empty, got {}x{}", w, h);
        }
        let needed = stride
            .checked_mul(h as usize)
            .ok_or_else(|| anyhow!("[RoiJpeg] output size overflow"))?;
        if stride < (w as usize) * 3 || out.len() < needed {
            bail!(
                "[RoiJpeg] output buffer holds {} bytes at pitch {}, need {} for {}x{} BGR",
                out.len(),
                stride,
                needed,
                w,
                h
            );
        }
        let rc = unsafe {
            roi_jpeg_decode_bgr(
                self.raw,
                data.as_ptr(),
                data.len(),
                x,
                y,
                w,
                h,
                out.as_mut_ptr(),
                stride,
            )
        };
        match rc {
            0 => Ok(()),
            -2 => Err(anyhow!("[RoiJpeg] bad arguments to roi_jpeg_decode_bgr")),
            _ => Err(anyhow!(
                "[RoiJpeg] decode of {}x{}+{}+{}: {}",
                w,
                h,
                x,
                y,
                self.last_error()
            )),
        }
    }
}

impl Drop for RoiJpeg {
    fn drop(&mut self) {
        unsafe { roi_jpeg_free(self.raw) };
        self.raw = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::Vector;
    use opencv::imgcodecs;
    use opencv::prelude::VectorToVec;

    /// A synthetic JPEG, so the test needs no capture card.
    fn sample(w: i32, h: i32) -> (Vec<u8>, opencv::core::Mat) {
        use opencv::core::{CV_8UC3, Mat, MatTraitManual};
        let mut src =
            Mat::new_rows_cols_with_default(h, w, CV_8UC3, opencv::core::Scalar::all(0.)).unwrap();
        // A gradient plus a hard edge: flat colour would hide any misalignment.
        {
            let bytes = src.data_bytes_mut().unwrap();
            for row in 0..h as usize {
                for col in 0..w as usize {
                    let p = (row * w as usize + col) * 3;
                    bytes[p] = (col % 251) as u8;
                    bytes[p + 1] = (row % 241) as u8;
                    bytes[p + 2] = if col > w as usize / 2 { 200 } else { 40 };
                }
            }
        }
        let mut buf = Vector::<u8>::new();
        imgcodecs::imencode(".jpg", &src, &mut buf, &Vector::new()).unwrap();
        // Round-trip through imdecode so the reference has the same quantisation.
        let reference =
            imgcodecs::imdecode(&buf, imgcodecs::ImreadModes::IMREAD_COLOR as i32).unwrap();
        (buf.to_vec(), reference)
    }

    #[test]
    fn reports_frame_size() {
        let (jpeg, _) = sample(320, 240);
        let mut dec = RoiJpeg::new().unwrap();
        assert_eq!(dec.size(&jpeg).unwrap(), (320, 240));
    }

    #[test]
    fn roi_matches_imdecode_then_crop() {
        use opencv::core::MatTraitConstManual;
        let (jpeg, reference) = sample(640, 480);
        let mut dec = RoiJpeg::new().unwrap();

        // Deliberately not MCU-aligned on either axis, which is what exercises
        // the lead-offset arithmetic in the shim.
        for &(x, y, w, h) in &[(0, 0, 64, 64), (99, 61, 96, 96), (576, 416, 64, 64)] {
            let stride = (w as usize) * 3;
            let mut out = vec![0u8; stride * h as usize];
            dec.decode_bgr(&jpeg, Rect { x, y, w, h }, &mut out, stride)
                .unwrap();

            let want = reference.data_bytes().unwrap();
            let mut worst = 0i32;
            for row in 0..h as usize {
                for col in 0..(w as usize) * 3 {
                    let a = want[(y as usize + row) * 640 * 3 + (x as usize) * 3 + col] as i32;
                    let b = out[row * stride + col] as i32;
                    worst = worst.max((a - b).abs());
                }
            }
            // Chroma upsampling reads a block beyond the crop edge, so the two
            // paths differ slightly there; they must not differ structurally.
            assert!(worst <= 40, "{}x{}+{}+{}: max diff {}", w, h, x, y, worst);
        }
    }

    #[test]
    fn whole_frame_is_a_valid_region() {
        use opencv::core::MatTraitConstManual;
        let (jpeg, reference) = sample(256, 192);
        let mut dec = RoiJpeg::new().unwrap();
        let stride = 256 * 3;
        let mut out = vec![0u8; stride * 192];
        dec.decode_bgr(
            &jpeg,
            Rect {
                x: 0,
                y: 0,
                w: 256,
                h: 192,
            },
            &mut out,
            stride,
        )
        .unwrap();
        let want = reference.data_bytes().unwrap();
        let worst = out
            .iter()
            .zip(want)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        assert!(worst <= 2, "full-frame decode differs by {}", worst);
    }

    #[test]
    fn rejects_a_region_outside_the_frame() {
        let (jpeg, _) = sample(128, 128);
        let mut dec = RoiJpeg::new().unwrap();
        let mut out = vec![0u8; 64 * 64 * 3];
        assert!(
            dec.decode_bgr(
                &jpeg,
                Rect {
                    x: 100,
                    y: 100,
                    w: 64,
                    h: 64,
                },
                &mut out,
                64 * 3
            )
            .is_err()
        );
    }

    #[test]
    fn survives_a_truncated_frame() {
        let (jpeg, _) = sample(320, 240);
        let mut dec = RoiJpeg::new().unwrap();
        let mut out = vec![0u8; 64 * 64 * 3];
        // Half a frame must come back as an error, not as a process exit.
        let half = &jpeg[..jpeg.len() / 2];
        let small = Rect {
            x: 0,
            y: 0,
            w: 64,
            h: 64,
        };
        let _ = dec.decode_bgr(half, small, &mut out, 64 * 3);
        // The decoder has to stay usable for the next good frame.
        dec.decode_bgr(&jpeg, small, &mut out, 64 * 3).unwrap();
    }
}
