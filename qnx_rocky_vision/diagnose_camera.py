"""One-shot, verbose diagnostic for the QNX camera1 (libcamapi) pipeline.

Exercises camera_open -> camera_get_supported_vf_modes -> camera_set_vf_mode ->
camera_start_viewfinder -> frame callback -> camera_stop_viewfinder ->
camera_close end-to-end, logging every stage (with flush=True) so a single
run pinpoints exactly where and why anything fails, without needing another
round trip to add print statements.

Run with:
    python3 -m qnx_rocky_vision.diagnose_camera [--callbacks real|null|vf-null|status-null]

--callbacks real        (default) real Python callbacks for both viewfinder and status.
--callbacks null        true NULL function pointers (via ctypes.cast) for BOTH callbacks --
                        isolates whether the crash is in the callback machinery at all.
--callbacks vf-null     real status callback, but NULL viewfinder callback.
--callbacks status-null real viewfinder callback, but NULL status callback.
"""

import argparse
import ctypes
import time
import traceback

from qnx_rocky_vision.landmark_viewer import (
    CAMERA_EOK,
    CAMERA_HANDLE_INVALID,
    CAMERA_IMGPROP_CREATEWINDOW,
    CAMERA_IMGPROP_END,
    CAMERA_MODE_VIEWFINDER,
    CAMERA_UNIT_1,
    CAMERA_VFMODE_DEFAULT,
    _load_camapi,
    _qnx_frame_to_rgb,
    get_supported_vf_modes,
)


def log(step, msg):
    print(f"[{step}] {msg}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--callbacks",
        choices=("real", "null", "vf-null", "status-null"),
        default="real",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log("1", "loading libcamapi and binding functions...")
    camapi = _load_camapi()
    lib = camapi["lib"]
    log("1", f"OK. camera_buffer_t size = {ctypes.sizeof(camapi['camera_buffer_t'])} bytes")

    log("2", "camera_open(CAMERA_UNIT_1, CAMERA_MODE_VIEWFINDER)...")
    handle = ctypes.c_int32(CAMERA_HANDLE_INVALID)
    err = lib.camera_open(CAMERA_UNIT_1, CAMERA_MODE_VIEWFINDER, ctypes.byref(handle))
    log("2", f"err={err} handle={handle.value}")
    if err != CAMERA_EOK or handle.value == CAMERA_HANDLE_INVALID:
        log("2", "FAILED to open camera. Stopping here.")
        return
    h = handle.value

    try:
        log("3", "camera_get_supported_vf_modes...")
        modes = get_supported_vf_modes(h)
        log("3", f"supported vf modes = {modes}")

        vf_mode = next((m for m in modes if m != CAMERA_VFMODE_DEFAULT), None)
        if vf_mode is None:
            log("4", f"No non-default vf mode available (modes={modes}). Stopping here.")
            return

        log("4", f"camera_set_vf_mode({vf_mode})...")
        err = lib.camera_set_vf_mode(h, vf_mode)
        log("4", f"err={err}")
        if err != CAMERA_EOK:
            log("4", "FAILED to set vf mode. Stopping here.")
            return

        log("4b", f"camera_set_vf_property(CREATEWINDOW={CAMERA_IMGPROP_CREATEWINDOW}, 0)...")
        err = lib.camera_private_set_vf_property(
            h, CAMERA_IMGPROP_CREATEWINDOW, 0, CAMERA_IMGPROP_END
        )
        log("4b", f"err={err}")
        if err != CAMERA_EOK:
            log("4b", "FAILED to disable auto-created viewfinder window. Stopping here.")
            return

        frames_seen = []
        callback_errors = []

        def _on_frame(_handle, buf_ptr, _arg):
            try:
                buf = buf_ptr.contents
                frametype = buf.frametype
                framesize = buf.framesize
                if len(frames_seen) == 0:
                    log(
                        "6",
                        f"first callback fired: frametype={frametype} framesize={framesize}",
                    )
                rgb = _qnx_frame_to_rgb(buf)
                frames_seen.append(rgb)
            except Exception:
                callback_errors.append(traceback.format_exc())

        def _on_status(_handle, status, ext_status, _arg):
            log("status", f"status callback: status={status} ext_status={ext_status}")

        real_viewfinder_cb = camapi["viewfinder_cb_t"](_on_frame)
        real_status_cb = camapi["status_cb_t"](_on_status)
        null_viewfinder_cb = ctypes.cast(None, camapi["viewfinder_cb_t"])
        null_status_cb = ctypes.cast(None, camapi["status_cb_t"])

        if args.callbacks == "null":
            viewfinder_cb, status_cb = null_viewfinder_cb, null_status_cb
        elif args.callbacks == "vf-null":
            viewfinder_cb, status_cb = null_viewfinder_cb, real_status_cb
        elif args.callbacks == "status-null":
            viewfinder_cb, status_cb = real_viewfinder_cb, null_status_cb
        else:
            viewfinder_cb, status_cb = real_viewfinder_cb, real_status_cb

        log("5", f"camera_start_viewfinder (callbacks={args.callbacks})...")
        err = lib.camera_start_viewfinder(h, viewfinder_cb, status_cb, None)
        log("5", f"err={err}")
        if err != CAMERA_EOK:
            log("5", "FAILED to start viewfinder. Stopping here.")
            return

        log("6", "waiting up to 3s for frames...")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not frames_seen and not callback_errors:
            time.sleep(0.05)

        if callback_errors:
            log("6", f"callback raised {len(callback_errors)} exception(s); first one:")
            print(callback_errors[0], flush=True)
        elif not frames_seen:
            log("6", "no frames received within 3s (callback never fired, no exceptions).")
        else:
            rgb = next((f for f in frames_seen if f is not None), None)
            log("6", f"received {len(frames_seen)} callback(s) in 3s")
            if rgb is None:
                log("6", "all frames decoded to None (unrecognized frametype?)")
            else:
                log(
                    "6",
                    f"frame OK: shape={rgb.shape} dtype={rgb.dtype} "
                    f"min={rgb.min()} max={rgb.max()}",
                )

        log("7", "camera_stop_viewfinder...")
        err = lib.camera_stop_viewfinder(h)
        log("7", f"err={err}")
    finally:
        log("8", "camera_close...")
        err = lib.camera_close(h)
        log("8", f"err={err}")

    log("done", "diagnostic complete")


if __name__ == "__main__":
    main()
