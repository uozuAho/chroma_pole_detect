# ESP32 S3 CAM Development Board + OV3660 Camera

Source: https://zaitronics.com.au/products/esp32-s3-cam-development-board-ov3660-camera

## Overview
A vision-enabled microcontroller platform built around the ESP32-S3-WROOM N16R8
module paired with the OV5640 camera. Designed for image capture, video
streaming, and edge AI. Suited to smart cameras, IoT vision devices, and rapid
prototyping needing wireless connectivity plus camera support.

## Key features
- ESP32-S3-WROOM N16R8 module with dual-core processor
- Integrated Wi-Fi and Bluetooth Low Energy
- OV5640 camera sensor for image capture and video
- Hardware acceleration for AI and DSP workloads
- Large onboard flash and PSRAM for image processing
- Suitable for edge AI, streaming, and vision projects

## Specifications
- Chip: ESP32-S3-WROOM
- single precision FPU
- Flash memory: 16 MB
- PSRAM: 8 MB
- Camera: OV5640
- Wireless: Wi-Fi 802.11 b/g/n and Bluetooth LE
- USB for power and programming

## Points relevant to image processing
- Dual-core ESP32-S3: dedicated processing for vision/DSP alongside
  application logic.
- Hardware acceleration for AI and DSP tasks — useful for fast colour-threshold
  / centroid calculations. Real SIMD via the PIE (Processor Instruction
  Extension) — 128-bit vector registers, SIMD on 8/16/32-bit elements, MAC and
  saturation ops. Algorithmic efficiency (fixed-point, integer arithmetic) is
  still important.
- 8 MB PSRAM provides a working buffer for captured frames, enabling larger
  resolutions and frame staging without exhausting limited internal SRAM.
- 16 MB flash is generous for storing firmware and potentially image assets.
- OV5640 camera: image capture and video applications; colour sensor well suited
  to detecting the green/pink/blue LED colours on the beacon pole. Also has on
  board processing and manual control such as resolution, ROI, binning,
  exposure, white balance
- This is an MCU-class processor, not a general-purpose CPU — the final C
  implementation must stay within constrained compute and memory, favouring
  simple integer/threshold/blob-detection algorithms (e.g. OpenCV-style
  centroid detection ported to hand-rolled C).

## References
- Board source: [Zaitronics ESP32-S3 CAM](https://zaitronics.com.au/products/esp32-s3-cam-development-board-ov3660-camera)
    - note this board has OV3660, we have OV5640
- ESP32-S3 Technical Reference Manual (PIE / SIMD details):
  https://documentation.espressif.com/esp32-s3_technical_reference_manual_en.pdf
- ESP-DSP library (optimized vector/FIR/FFT primitives):
  https://github.com/espressif/esp-dsp , such as:
    - vector arithmetic
    - dot products
    - matrix multiplication
    - FIR/IIR filters
    - FFTs
    - Kalman filters
