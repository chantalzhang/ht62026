/*
 * camera_bridge: a small native helper that captures frames from QNX
 * camera1 (via libcamapi, in callback mode) and streams them to stdout as
 * a simple binary protocol.
 *
 * Why this exists: invoking a ctypes callback from a thread libcamapi
 * creates internally (i.e. not a thread Python itself spawned) reliably
 * segfaults this QNX Python 3.14 build. Plain C has no such problem --
 * camera_example1_callback (shipped with the OS) already proves native
 * callback-mode capture works fine on this hardware. So this helper does
 * the camera_open/camera_start_viewfinder callback dance in pure C, and
 * just pipes raw frame bytes to Python over stdout, which Python reads
 * with ordinary blocking I/O (no native callback involved on the Python
 * side at all).
 *
 * Wire format, written to stdout once per frame:
 *   char     magic[4]     ("QCF1")
 *   int32_t  frametype    (camera_frametype_t)
 *   uint32_t height
 *   uint32_t width
 *   uint32_t stride
 *   uint8_t  data[height * stride]   (raw framebuf bytes)
 *
 * Build (on the QNX target):
 *   qcc -Vgcc_ntoaarch64le -o camera_bridge camera_bridge.c -lcamapi
 *   (if that -V variant doesn't exist, run `qcc -V` with no args to list
 *   the variants actually installed and substitute the right aarch64 one)
 *
 * Run standalone (for testing):
 *   ./camera_bridge > /tmp/test_frames.bin
 *   (Ctrl+C after a couple seconds, then check /tmp/test_frames.bin grew)
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
    int32_t frametype;
    uint32_t height;
    uint32_t width;
    uint32_t stride;
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

static void viewfinder_callback(camera_handle_t handle, camera_buffer_t *buf, void *arg) {
    (void)handle;
    (void)arg;

    /* framedesc's exact typed layout varies by frametype/BSP; we only rely
     * on the first three uint32_t fields being height, width, stride in
     * that order, which is the same assumption already used elsewhere in
     * this project (and is what we're about to validate for the first
     * time, since capture never got this far via ctypes). */
    const uint8_t *desc = (const uint8_t *)&buf->framedesc;
    uint32_t height = 0, width = 0, stride = 0;
    memcpy(&height, desc + 0, sizeof(uint32_t));
    memcpy(&width, desc + sizeof(uint32_t), sizeof(uint32_t));
    memcpy(&stride, desc + 2 * sizeof(uint32_t), sizeof(uint32_t));

    if (height == 0 || width == 0 || stride == 0) {
        return;
    }

    frame_header_t header;
    memcpy(header.magic, "QCF1", 4);
    header.frametype = buf->frametype;
    header.height = height;
    header.width = width;
    header.stride = stride;

    write_all(1, &header, sizeof(header));
    write_all(1, buf->framebuf, (size_t)height * stride);
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

    fprintf(stderr, "camera_bridge: streaming frames on stdout (Ctrl+C or close stdin to stop)\n");

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
