# Changelog

## v1.1.0 - 2026-11-20

### Changed

- Replaced the hand-written shape constant folder with onnxsim 0.7.3.
- Replaced raw ONNX protobuf graph surgery with onnx-graphsurgeon 0.6.1.
- Updated the PyInstaller spec to console=False (no black console window).
- Updated requirements with pinned onnxsim/onnx-graphsurgeon versions.
- Kept preflight, grouping, mask, cumulative exploration, CSV, and UI semantics unchanged.

### Verified

- Anchor preflight status=OK, 39 groups, 1 protected group, 0 UNRESOLVABLE.
- First group cut=7, 20-image mIoU 68.49%, matching the previous backend.
- Cumulative 3 groups x 2 ratios and round_to=1/8 checks passed.
- onnxsim output is numerically equivalent to the original on the anchor: max_abs_diff=0.0, argmax agreement=1.0.
- GraphSurgeon import/export/checker/shape inference passed.

### Packaged

- Rebuilt the windowed exe with console=False; no console window appears.
- Added outputs/app.log and a startup-failure MessageBox.
- Bundled VOCdevkit and samples/fcn_resnet50_voc_520.onnx.
- Verified packaged exe: HTTP 200, green preflight, 2-image / 1-group analysis in 6.5s.

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
