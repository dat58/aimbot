//! Builds the region-of-interest MJPEG decoder in `src/stream/roi_jpeg.c` and
//! links it against the system libjpeg (libjpeg-turbo).
//!
//! `jpeg_crop_scanline` and `jpeg_skip_scanlines` need libjpeg-turbo >= 1.5;
//! every distro build of it exports them. On Debian/Ubuntu the headers come
//! from `libjpeg-turbo8-dev` (or `libjpeg62-turbo-dev`).

fn main() {
    println!("cargo:rerun-if-changed=src/stream/roi_jpeg.c");

    cc::Build::new()
        .file("src/stream/roi_jpeg.c")
        .opt_level(2)
        .warnings(true)
        .compile("roi_jpeg");

    println!("cargo:rustc-link-lib=jpeg");
}
