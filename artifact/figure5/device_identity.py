#!/usr/bin/env python3
"""Identify logical CUDA device 0 without creating a CUDA context or launching work."""
import ctypes
import uuid


def selected_uuid(driver=None):
    driver = driver or ctypes.CDLL("libcuda.so.1")
    check = lambda result: result == 0
    if not check(driver.cuInit(0)):
        raise RuntimeError("cuInit failed")
    device = ctypes.c_int()
    if not check(driver.cuDeviceGet(ctypes.byref(device), 0)):
        raise RuntimeError("logical CUDA device 0 is unavailable")
    value = (ctypes.c_ubyte * 16)()
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None) or driver.cuDeviceGetUuid
    if not check(get_uuid(ctypes.byref(value), device)):
        raise RuntimeError("cuDeviceGetUuid failed")
    return "GPU-" + str(uuid.UUID(bytes=bytes(value)))


if __name__ == "__main__":
    try:
        print("GPU_UUID=" + selected_uuid())
    except (OSError, RuntimeError) as error:
        raise SystemExit(str(error))
