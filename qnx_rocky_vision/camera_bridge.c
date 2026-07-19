/*
 * camera_bridge: a small native helper that captures frames from QNX
 * camera1 (via libcamapi, in callback mode), converts them to plain RGB24,
 * and streams them to stdout as a simple binary protocol.
 *
 * Why this exists: invoking a ctypes callback from a thread libcamapi
 * creates internally (i.e. not a thread Python itself spawned) reliably
 * segfaults this QNX Python 3.14 build. Plain C has no such problem --
 * camera_example1_callback (shipped with the OS) already proves native
 * callback-mode capture works fine on this hardware. So this helper does
 * the camera_open/camera_start_viewfinder callback dance AND the pixel
 * format conversion in pure C, and just pipes ready-to-use RGB24 frames to
 * Python over stdout, which Python reads with ordinary blocking I/O (no
 * native callback involved on the Python side at all).
 *
 * All pixel-format handling lives here instead of in Python so that:
 *   (a) it can be compiled directly against the real camera_defs.h headers
 *       on the target (no guessing enum values or struct layouts by hand),
 *       in particular the confirmed real camera_frame_nv12_t layout, and
 *   (b) Python's side of the protocol never has to branch on frame type at
 *       all -- it just gets ready-to-display RGB24 rows.
 *
 * Wire format, written to stdout once per frame:
 *   char     magic[4]     ("QCF1")
 *   uint32_t height
 *   uint32_t width
 *   uint8_t  rgb[height * width * 3]   (packed RGB24, no row padding)
 *
 * Build (on the QNX target):
 *   gcc -o camera_bridge camera_bridge.c -lcamapi
 *   (or, if gcc isn't available directly: qcc -Vgcc_ntoaarch64le -o
 *   camera_bridge camera_bridge.c -lcamapi -- run `qcc -V` with no args to
 *   list installed variants and substitute the right aarch64 one)
 *
 * Run standalone (for testing):
 *   timeout 3 ./camera_bridge > /tmp/test_frames.bin
 *   ls -la /tmp/test_frames.bin
 */

#include <camera/camera_api.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    char magic[4];
    uint32_t height;
    uint32_t width;
} frame_header_t;

static void write_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t written = 0;
    while (written < len) {
        ssize_t n = write(fd, p + written, len - written);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) {
                continue;
            }
            fprintf(stderr, "camera_bridge: write failed: %s\n", strerror(errno));
            exit(1);
        }
        written += (size_t)n;
    }
}

/* Reused across callbacks instead of malloc/free per frame. The viewfinder
 * callback runs on a single dedicated libcamapi thread (never concurrently
 * with itself), so a plain static buffer is safe here. */
static uint8_t *g_rgb_buf = NULL;
static size_t g_rgb_buf_size = 0;

static uint8_t *get_rgb_buffer(size_t needed) {
    if (needed > g_rgb_buf_size) {
        uint8_t *new_buf = realloc(g_rgb_buf, needed);
        if (!new_buf) {
            return NULL;
        }
        g_rgb_buf = new_buf;
        g_rgb_buf_size = needed;
    }
    return g_rgb_buf;
}

static inline uint8_t clip_u8(float v) {
    if (v < 0.0f) return 0;
    if (v > 255.0f) return 255;
    return (uint8_t)v;
}

/* RGB8888 / BGR8888: 4 bytes/pixel, `stride` bytes between rows. */
static void convert_packed_rgb(const uint8_t *framebuf, uint32_t height, uint32_t width,
                                uint32_t stride, int is_bgr, uint8_t *out) {
    for (uint32_t y = 0; y < height; y++) {
        const uint8_t *row = framebuf + (size_t)y * stride;
        uint8_t *out_row = out + (size_t)y * width * 3;
        for (uint32_t x = 0; x < width; x++) {
            const uint8_t *px = row + (size_t)x * 4;
            uint8_t *out_px = out_row + (size_t)x * 3;
            if (is_bgr) {
                out_px[0] = px[2];
                out_px[1] = px[1];
                out_px[2] = px[0];
            } else {
                out_px[0] = px[0];
                out_px[1] = px[1];
                out_px[2] = px[2];
            }
        }
    }
}

/* YCbYCr / CbYCrY: packed 4:2:2, 2 bytes/pixel on average (4 bytes per
 * horizontal pixel pair), `stride` bytes between rows. BT.601 coefficients. */
static void convert_yuv422(const uint8_t *framebuf, uint32_t height, uint32_t width,
                            uint32_t stride, int is_cbycry, uint8_t *out) {
    for (uint32_t y = 0; y < height; y++) {
        const uint8_t *row = framebuf + (size_t)y * stride;
        uint8_t *out_row = out + (size_t)y * width * 3;
        for (uint32_t x = 0; x < width; x += 2) {
            const uint8_t *quad = row + (size_t)x * 2;
            uint8_t y0, cb, y1, cr;
            if (is_cbycry) {
                cb = quad[0]; y0 = quad[1]; cr = quad[2]; y1 = quad[3];
            } else {
                y0 = quad[0]; cb = quad[1]; y1 = quad[2]; cr = quad[3];
            }
            float cbf = (float)cb - 128.0f;
            float crf = (float)cr - 128.0f;

            float r0 = (float)y0 + 1.402f * crf;
            float g0 = (float)y0 - 0.344136f * cbf - 0.714136f * crf;
            float b0 = (float)y0 + 1.772f * cbf;
            uint8_t *px0 = out_row + (size_t)x * 3;
            px0[0] = clip_u8(r0);
            px0[1] = clip_u8(g0);
            px0[2] = clip_u8(b0);

            if (x + 1 < width) {
                float r1 = (float)y1 + 1.402f * crf;
                float g1 = (float)y1 - 0.344136f * cbf - 0.714136f * crf;
                float b1 = (float)y1 + 1.772f * cbf;
                uint8_t *px1 = out_row + (size_t)(x + 1) * 3;
                px1[0] = clip_u8(r1);
                px1[1] = clip_u8(g1);
                px1[2] = clip_u8(b1);
            }
        }
    }
}

/* NV12: semi-planar 4:2:0 -- one full-resolution Y plane, followed (at
 * uv_offset bytes into the same buffer) by a half-resolution plane of
 * interleaved U,V byte pairs. BT.601 coefficients. This is the format
 * camera1 is actually configured for on this hardware (camera_module3.conf
 * sets default_video_format = nv12), and camera_frame_nv12_t's layout is
 * confirmed directly from the real header, unlike the other formats above. */
static void convert_nv12(const uint8_t *framebuf, uint32_t height, uint32_t width,
                          uint32_t y_stride, int64_t uv_offset, int64_t uv_stride,
                          uint8_t *out) {
    const uint8_t *uv_plane = framebuf + uv_offset;
    for (uint32_t y = 0; y < height; y++) {
        const uint8_t *y_row = framebuf + (size_t)y * y_stride;
        const uint8_t *uv_row = uv_plane + (size_t)(y / 2) * uv_stride;
        uint8_t *out_row = out + (size_t)y * width * 3;
        for (uint32_t x = 0; x < width; x++) {
            uint8_t yv = y_row[x];
            uint8_t u = uv_row[(x / 2) * 2];
            uint8_t v = uv_row[(x / 2) * 2 + 1];
            float uf = (float)u - 128.0f;
            float vf = (float)v - 128.0f;
            float r = (float)yv + 1.402f * vf;
            float g = (float)yv - 0.344136f * uf - 0.714136f * vf;
            float b = (float)yv + 1.772f * uf;
            uint8_t *px = out_row + (size_t)x * 3;
            px[0] = clip_u8(r);
            px[1] = clip_u8(g);
            px[2] = clip_u8(b);
        }
    }
}

/* Logged for the first several callbacks (frames are frequent, so we
 * throttle after that) to make the callback's behaviour visible instead of
 * silently dropping frames whose dimensions come out to 0. */
static int g_callbacks_logged = 0;
#define MAX_CALLBACKS_LOGGED 5

static void viewfinder_callback(camera_handle_t handle, camera_buffer_t *buf, void *arg) {
    (void)handle;
    (void)arg;

    uint32_t height = 0, width = 0;
    uint8_t *rgb;
    int should_log = g_callbacks_logged < MAX_CALLBACKS_LOGGED;
    if (should_log) {
        g_callbacks_logged++;
        fprintf(stderr,
                "camera_bridge: callback fired #%d: frametype=%d (NV12=%d, RGB8888=%d, "
                "BGR8888=%d, YCBYCR=%d, CBYCRY=%d) framesize=%llu\n",
                g_callbacks_logged, buf->frametype, CAMERA_FRAMETYPE_NV12,
                CAMERA_FRAMETYPE_RGB8888, CAMERA_FRAMETYPE_BGR8888, CAMERA_FRAMETYPE_YCBYCR,
                CAMERA_FRAMETYPE_CBYCRY, (unsigned long long)buf->framesize);
    }

    if (buf->frametype == CAMERA_FRAMETYPE_NV12) {
        height = buf->framedesc.nv12.height;
        width = buf->framedesc.nv12.width;
        if (should_log) {
            fprintf(stderr,
                    "camera_bridge:   nv12 framedesc: height=%u width=%u stride=%u "
                    "uv_offset=%lld uv_stride=%lld\n",
                    height, width, buf->framedesc.nv12.stride,
                    (long long)buf->framedesc.nv12.uv_offset,
                    (long long)buf->framedesc.nv12.uv_stride);
        }
        if (height == 0 || width == 0) {
            if (should_log) {
                fprintf(stderr, "camera_bridge:   dropping frame: zero height/width\n");
            }
            return;
        }
        rgb = get_rgb_buffer((size_t)height * width * 3);
        if (!rgb) {
            fprintf(stderr, "camera_bridge: out of memory for RGB buffer\n");
            return;
        }
        convert_nv12(buf->framebuf, height, width, buf->framedesc.nv12.stride,
                     buf->framedesc.nv12.uv_offset, buf->framedesc.nv12.uv_stride, rgb);
    } else {
        /* Fallback for other frame types: we don't have a confirmed struct
         * layout for these on this BSP (unlike NV12 above), so assume
         * framedesc's first three uint32_t fields are height, width, stride
         * in that order -- matches camera_frame_nv12_t's own leading fields,
         * so it's a reasonable bet, but has not been directly confirmed. */
        const uint8_t *desc = (const uint8_t *)&buf->framedesc;
        uint32_t stride = 0;
        memcpy(&height, desc + 0, sizeof(uint32_t));
        memcpy(&width, desc + sizeof(uint32_t), sizeof(uint32_t));
        memcpy(&stride, desc + 2 * sizeof(uint32_t), sizeof(uint32_t));
        if (should_log) {
            fprintf(stderr,
                    "camera_bridge:   guessed framedesc (first 3 uint32s): height=%u width=%u "
                    "stride=%u; first 32 raw bytes:",
                    height, width, stride);
            for (int i = 0; i < 32; i++) {
                fprintf(stderr, " %02x", desc[i]);
            }
            fprintf(stderr, "\n");
        }
        if (height == 0 || width == 0 || stride == 0) {
            if (should_log) {
                fprintf(stderr, "camera_bridge:   dropping frame: zero height/width/stride\n");
            }
            return;
        }

        rgb = get_rgb_buffer((size_t)height * width * 3);
        if (!rgb) {
            fprintf(stderr, "camera_bridge: out of memory for RGB buffer\n");
            return;
        }

        switch (buf->frametype) {
        case CAMERA_FRAMETYPE_RGB8888:
            convert_packed_rgb(buf->framebuf, height, width, stride, 0, rgb);
            break;
        case CAMERA_FRAMETYPE_BGR8888:
            convert_packed_rgb(buf->framebuf, height, width, stride, 1, rgb);
            break;
        case CAMERA_FRAMETYPE_YCBYCR:
            convert_yuv422(buf->framebuf, height, width, stride, 0, rgb);
            break;
        case CAMERA_FRAMETYPE_CBYCRY:
            convert_yuv422(buf->framebuf, height, width, stride, 1, rgb);
            break;
        default:
            fprintf(stderr, "camera_bridge: dropping unsupported frametype=%d\n", buf->frametype);
            return;
        }
    }

    frame_header_t header;
    memcpy(header.magic, "QCF1", 4);
    header.height = height;
    header.width = width;

    write_all(1, &header, sizeof(header));
    write_all(1, rgb, (size_t)height * width * 3);
}

int main(void) {
    camera_handle_t handle = CAMERA_HANDLE_INVALID;
    camera_error_t err;

    err = camera_open(CAMERA_UNIT_1, CAMERA_MODE_PREAD | CAMERA_MODE_PWRITE | CAMERA_MODE_DREAD, &handle);
    if (err != CAMERA_EOK) {
        fprintf(stderr, "camera_bridge: camera_open failed: err=%d\n", err);
        return 1;
    }

    err = camera_set_vf_mode(handle, CAMERA_VFMODE_VIDEO);
    if (err != CAMERA_EOK) {
        fprintf(stderr, "camera_bridge: camera_set_vf_mode failed: err=%d\n", err);
        camera_close(handle);
        return 1;
    }

    err = camera_set_vf_property(handle, CAMERA_IMGPROP_CREATEWINDOW, 0);
    if (err != CAMERA_EOK) {
        fprintf(stderr, "camera_bridge: camera_set_vf_property(CREATEWINDOW) failed: err=%d\n", err);
        camera_close(handle);
        return 1;
    }

    err = camera_start_viewfinder(handle, viewfinder_callback, NULL, NULL);
    if (err != CAMERA_EOK) {
        fprintf(stderr, "camera_bridge: camera_start_viewfinder failed: err=%d\n", err);
        camera_close(handle);
        return 1;
    }

    fprintf(stderr, "camera_bridge: streaming RGB24 frames on stdout (Ctrl+C or close stdin to stop)\n");

    /* Run until stdin closes (parent process exiting/closing the pipe) or
     * we're killed. Reading from stdin doubles as a way for the Python
     * parent to signal shutdown just by closing its end of the pipe. */
    char discard[64];
    while (1) {
        ssize_t n = read(0, discard, sizeof(discard));
        if (n <= 0) {
            break;
        }
    }

    camera_stop_viewfinder(handle);
    camera_close(handle);
    return 0;
}
