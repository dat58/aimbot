/* Region-of-interest MJPEG decoding on top of the system libjpeg-turbo.
 *
 * OpenCV's `imdecode` always decodes the whole frame and always re-creates the
 * decompressor, which costs ~7 ms for a 1080p frame from an Elgato 4K X. The
 * detection pipeline only ever looks at a small region around the crosshair,
 * so almost all of that work is thrown away.
 *
 * `jpeg_crop_scanline` + `jpeg_skip_scanlines` let libjpeg-turbo skip the IDCT,
 * chroma upsampling and colour conversion for every block outside the region:
 * a 192x192 region out of 1080p drops from ~7 ms to ~0.46 ms. Reusing one
 * decompressor across frames keeps libjpeg's memory pool warm as well, which is
 * most of the difference between the full-frame path here (~1.9 ms) and
 * `imdecode`.
 *
 * This lives in C rather than as Rust FFI declarations on purpose: the layout
 * of `struct jpeg_decompress_struct` depends on the libjpeg version and on
 * jconfig.h, so letting the C compiler see the real headers is what keeps the
 * ABI correct. Nothing here allocates a frame buffer — the caller owns the
 * output and we write BGR straight into it.
 */

/* jpeglib.h does not include its own prerequisites: it names `size_t` and
 * `FILE` without declaring them, so these have to come first. */
#include <setjmp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <jpeglib.h>

/* libjpeg's default error handler calls exit(). A capture card hands out the
 * occasional truncated frame, so every error has to come back as a value. */
struct roi_jpeg_err {
    struct jpeg_error_mgr pub;
    jmp_buf escape;
    char msg[JMSG_LENGTH_MAX];
};

struct roi_jpeg {
    struct jpeg_decompress_struct cinfo;
    struct roi_jpeg_err err;
    /* One decoded scanline at the MCU-aligned crop width. */
    unsigned char *row;
    size_t row_cap;
};

static void roi_jpeg_error_exit(j_common_ptr cinfo)
{
    struct roi_jpeg_err *e = (struct roi_jpeg_err *)cinfo->err;
    (*cinfo->err->format_message)(cinfo, e->msg);
    longjmp(e->escape, 1);
}

/* Warnings (corrupt/truncated data) would otherwise be printed to stderr on
 * every bad frame. Keep the text for the caller and stay quiet. */
static void roi_jpeg_output_message(j_common_ptr cinfo)
{
    struct roi_jpeg_err *e = (struct roi_jpeg_err *)cinfo->err;
    (*cinfo->err->format_message)(cinfo, e->msg);
}

/* `volatile` in the two functions below is not decoration: a local whose value
 * is needed after a `longjmp` may otherwise live in a call-clobbered register,
 * which is what -Wclobbered warns about. */
struct roi_jpeg *roi_jpeg_new(void)
{
    struct roi_jpeg *volatile rj = calloc(1, sizeof(struct roi_jpeg));
    if (!rj)
        return NULL;
    rj->cinfo.err = jpeg_std_error(&rj->err.pub);
    rj->err.pub.error_exit = roi_jpeg_error_exit;
    rj->err.pub.output_message = roi_jpeg_output_message;
    if (setjmp(rj->err.escape)) {
        free(rj);
        return NULL;
    }
    jpeg_create_decompress(&rj->cinfo);
    return rj;
}

void roi_jpeg_free(struct roi_jpeg *rj_in)
{
    struct roi_jpeg *volatile rj = rj_in;
    if (!rj)
        return;
    if (!setjmp(rj->err.escape))
        jpeg_destroy_decompress(&rj->cinfo);
    free(rj->row);
    free(rj);
}

const char *roi_jpeg_error(const struct roi_jpeg *rj)
{
    return rj ? rj->err.msg : "no decoder";
}

static int roi_jpeg_reserve_row(struct roi_jpeg *rj, size_t bytes)
{
    if (rj->row_cap >= bytes)
        return 0;
    unsigned char *grown = realloc(rj->row, bytes);
    if (!grown)
        return -1;
    rj->row = grown;
    rj->row_cap = bytes;
    return 0;
}

/* Dimensions of the next frame, without decoding it. */
int roi_jpeg_size(struct roi_jpeg *rj, const unsigned char *data, size_t len,
                  int *width, int *height)
{
    if (!rj || !data || !width || !height)
        return -2;
    rj->err.msg[0] = '\0';
    if (setjmp(rj->err.escape)) {
        jpeg_abort_decompress(&rj->cinfo);
        return -1;
    }
    jpeg_mem_src(&rj->cinfo, data, (unsigned long)len);
    if (jpeg_read_header(&rj->cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_abort_decompress(&rj->cinfo);
        strcpy(rj->err.msg, "incomplete JPEG header");
        return -1;
    }
    *width = (int)rj->cinfo.image_width;
    *height = (int)rj->cinfo.image_height;
    jpeg_abort_decompress(&rj->cinfo);
    return 0;
}

/* Decode the (x, y, w, h) rect as BGR into `out`, `out_stride` bytes per row.
 *
 * Passing the whole frame as the rect is supported and is what the non-ROI
 * path uses; it stays cheaper than `imdecode` because the decompressor and its
 * memory pool are reused.
 *
 * Returns 0, -1 on a JPEG error (see roi_jpeg_error), -2 on bad arguments. */
int roi_jpeg_decode_bgr(struct roi_jpeg *rj, const unsigned char *data, size_t len,
                        int x, int y, int w, int h,
                        unsigned char *out, size_t out_stride)
{
    if (!rj || !data || !out || w <= 0 || h <= 0 || x < 0 || y < 0)
        return -2;
    if (out_stride < (size_t)w * 3)
        return -2;

    rj->err.msg[0] = '\0';
    if (setjmp(rj->err.escape)) {
        jpeg_abort_decompress(&rj->cinfo);
        return -1;
    }

    jpeg_mem_src(&rj->cinfo, data, (unsigned long)len);
    if (jpeg_read_header(&rj->cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_abort_decompress(&rj->cinfo);
        strcpy(rj->err.msg, "incomplete JPEG header");
        return -1;
    }

    /* JCS_EXT_BGR is a libjpeg-turbo extension: it emits BGR directly, so the
     * pipeline needs no colour-conversion pass of its own. */
    rj->cinfo.out_color_space = JCS_EXT_BGR;
    jpeg_start_decompress(&rj->cinfo);

    if ((unsigned)x + (unsigned)w > rj->cinfo.output_width ||
        (unsigned)y + (unsigned)h > rj->cinfo.output_height) {
        jpeg_abort_decompress(&rj->cinfo);
        strcpy(rj->err.msg, "region does not fit inside the frame");
        return -1;
    }

    /* Snaps `crop_x` down to an MCU boundary and widens `crop_w` to cover the
     * request, so the decoded row starts before the region we asked for. */
    JDIMENSION crop_x = (JDIMENSION)x;
    JDIMENSION crop_w = (JDIMENSION)w;
    jpeg_crop_scanline(&rj->cinfo, &crop_x, &crop_w);

    if (crop_x > (JDIMENSION)x || crop_x + crop_w < (JDIMENSION)(x + w)) {
        jpeg_abort_decompress(&rj->cinfo);
        strcpy(rj->err.msg, "crop window does not cover the region");
        return -1;
    }
    if (roi_jpeg_reserve_row(rj, (size_t)crop_w * 3) != 0) {
        jpeg_abort_decompress(&rj->cinfo);
        strcpy(rj->err.msg, "out of memory for the scanline buffer");
        return -1;
    }

    if (y > 0)
        jpeg_skip_scanlines(&rj->cinfo, (JDIMENSION)y);

    const size_t lead = (size_t)((JDIMENSION)x - crop_x) * 3;
    const size_t span = (size_t)w * 3;
    for (int row = 0; row < h; row++) {
        JSAMPROW dst = rj->row;
        if (jpeg_read_scanlines(&rj->cinfo, &dst, 1) != 1) {
            jpeg_abort_decompress(&rj->cinfo);
            strcpy(rj->err.msg, "frame ended before the region was filled");
            return -1;
        }
        memcpy(out + (size_t)row * out_stride, rj->row + lead, span);
    }

    /* Deliberately abort instead of finish: the scanlines past the region are
     * exactly the work this function exists to avoid. */
    jpeg_abort_decompress(&rj->cinfo);
    return 0;
}
