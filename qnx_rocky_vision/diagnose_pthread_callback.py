"""Isolated test: does invoking a ctypes CFUNCTYPE callback from a genuine
foreign (non-Python-created) pthread work at all on this system?

This is completely independent of libcamapi -- it spawns a plain OS thread
via libc's pthread_create() and has that thread invoke a Python callback, to
determine whether camera_start_viewfinder's crash-on-first-frame is a
generic ctypes/foreign-thread limitation on this QNX Python build, or
something specific to libcamapi's call/buffer types.

Run with:
    python3 -m qnx_rocky_vision.diagnose_pthread_callback
"""

import ctypes
import time


def log(step, msg):
    print(f"[{step}] {msg}", flush=True)


def main():
    log("1", "loading libc...")
    libc = ctypes.CDLL("libc.so")
    log("1", "OK")

    called = []

    thread_func_t = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)

    def _on_thread(_arg):
        log("3", "callback invoked from foreign pthread!")
        called.append(True)
        return None

    cb = thread_func_t(_on_thread)

    libc.pthread_create.argtypes = [
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_void_p,
        thread_func_t,
        ctypes.c_void_p,
    ]
    libc.pthread_create.restype = ctypes.c_int

    thread_id = ctypes.c_ulong(0)
    log("2", "pthread_create...")
    err = libc.pthread_create(ctypes.byref(thread_id), None, cb, None)
    log("2", f"err={err}")

    log("2b", "waiting up to 2s for the foreign thread to invoke the callback...")
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not called:
        time.sleep(0.05)

    if called:
        log("done", "SUCCESS: a ctypes callback invoked from a foreign pthread worked fine.")
    else:
        log("done", "callback never fired within 2s (but no crash either).")


if __name__ == "__main__":
    main()
