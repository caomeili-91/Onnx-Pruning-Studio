# Changelog

## v1.0.0 - 2026-11-14

### Added

- ONNX-only static graph preflight for [1,3,520,520] -> [1,21,520,520] VOC segmentation.
- Channel dependency grouping and protobuf-level pruning.
- Cumulative group exploration with ORT float32 mIoU.
- Minimal Gradio UI: upload, red/green preflight, one-click analysis, cumulative mIoU plot, collapsed log.
- Verification scripts for single-group, cumulative semantics, and round_to alignment.
- Packaged Windows executable with bundled VOCdevkit and sample ONNX.

### Changed

- UI reduced to a practical student-course layout: profile/sort/output/seed hidden or fixed.
- Upload handling waits for the Gradio temp file to finish writing before preflight.
- README and repository docs aligned with v1.0.0.

### Known limits

- Feature Concat, Depthwise Conv, and Flatten->Gemm are UNRESOLVABLE in v1.
- L1 ordering can cause non-monotonic or cliff-like mIoU drops at higher ratios.
- CPU-only ORT evaluation; full sweeps are slow.
