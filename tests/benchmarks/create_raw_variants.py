#!/usr/bin/env python3
"""
Create raw binary test data files in different data types from the existing
UINT8 raw file (100x100x100). These files are used by tRawReader tests.

Input:  data/SampleData_2Phase_stack_3d_uint8.raw  (100x100x100 UINT8)
Output: data/SampleData_2Phase_stack_3d_uint16le.raw  (UINT16 little-endian)
        data/SampleData_2Phase_stack_3d_int16le.raw   (INT16 little-endian)
        data/SampleData_2Phase_stack_3d_float32le.raw  (FLOAT32 little-endian)
"""

import os
import struct
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INPUT_FILE = os.path.join(DATA_DIR, "SampleData_2Phase_stack_3d_uint8.raw")
NX, NY, NZ = 100, 100, 100
NVOXELS = NX * NY * NZ


def main():
    # Read the source UINT8 data
    print(f"Reading {INPUT_FILE} ({NX}x{NY}x{NZ} UINT8, {NVOXELS} voxels)...")
    with open(INPUT_FILE, "rb") as f:
        raw_bytes = f.read()

    if len(raw_bytes) != NVOXELS:
        print(
            f"ERROR: Expected {NVOXELS} bytes, got {len(raw_bytes)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    uint8_values = struct.unpack(f"<{NVOXELS}B", raw_bytes)
    print(f"  Value range: [{min(uint8_values)}, {max(uint8_values)}]")

    # --- UINT16_LE: scale 0-255 -> 0-65535 ---
    out_path = os.path.join(DATA_DIR, "SampleData_2Phase_stack_3d_uint16le.raw")
    print(f"Writing {out_path} ...")
    uint16_values = [v * 257 for v in uint8_values]  # 255*257 = 65535
    with open(out_path, "wb") as f:
        f.write(struct.pack(f"<{NVOXELS}H", *uint16_values))
    expected_size = NVOXELS * 2
    actual_size = os.path.getsize(out_path)
    print(f"  Size: {actual_size} bytes (expected {expected_size})")
    assert actual_size == expected_size

    # --- INT16_LE: scale 0-255 -> 0-32767 ---
    out_path = os.path.join(DATA_DIR, "SampleData_2Phase_stack_3d_int16le.raw")
    print(f"Writing {out_path} ...")
    int16_values = [int(v * 32767.0 / 255.0 + 0.5) for v in uint8_values]
    with open(out_path, "wb") as f:
        f.write(struct.pack(f"<{NVOXELS}h", *int16_values))
    actual_size = os.path.getsize(out_path)
    print(f"  Size: {actual_size} bytes (expected {expected_size})")
    assert actual_size == expected_size

    # --- FLOAT32_LE: values as 0.0-255.0 ---
    out_path = os.path.join(DATA_DIR, "SampleData_2Phase_stack_3d_float32le.raw")
    print(f"Writing {out_path} ...")
    float32_values = [float(v) for v in uint8_values]
    with open(out_path, "wb") as f:
        f.write(struct.pack(f"<{NVOXELS}f", *float32_values))
    expected_size = NVOXELS * 4
    actual_size = os.path.getsize(out_path)
    print(f"  Size: {actual_size} bytes (expected {expected_size})")
    assert actual_size == expected_size

    print("\nDone. All files created successfully.")


if __name__ == "__main__":
    main()
