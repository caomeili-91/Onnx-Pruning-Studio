# ONNX 静态图剪枝可行性分析平台 (v1.0.0)

产品定位：**静态 CNN 分割 ONNX 的规则预检 + 逐组剪枝敏感度分析平台。**

v1 只接受：

- 输入 `[1,3,520,520]`、float32 的 ONNX；
- 输出 `[1,21,520,520]` 的 VOC 21 类语义分割 logits；
- 评估指标为 ONNX Runtime float32 下的 mIoU。

> 预检为基于 target profile 的静态规则判断，不运行厂商转换器；mIoU 来自 ONNX Runtime float32 推理，不代表 NPU/MCU 量化后精度。

## v1.0.0 定版状态（2026-11-14）

v1 已具备定版条件。定版范围是“静态 520x520 VOC 分割 ONNX 的规则预检 + 按当前策略的累计精度分析工具”，不是目标硬件交付工具，也不承诺剪枝后的目标硬件精度。

验收证据：

- UI 验收完成；psutil 7.2.2 已随 exe 打包并生效。示例 baseline 的 RSS 为 592.7 MB -> 699.0 MB，已写入 ui_baseline.json / rss.log。
- L1 升序、随机、层深度升序三种排序均可在 UI 切换；日志和 CSV 的 sort_strategy 字段正确变化，组顺序明显不同。
- 完整 L1 探索实测 689s；曲线出现的非单调/断崖掉点已被 UI 固定警示覆盖。
- exe clean build 完成；HTTP 200；打包内预检 status=OK；小规模探索 smoke test 通过。
- 打包目录已包含 VOCdevkit 与 samples/fcn_resnet50_voc_520.onnx。

## 快速运行

已经构建好的软件：

```text
dist\OnnxPruneStudio\OnnxPruneStudio.exe
```

exe 同目录已包含 `VOCdevkit\` 和 `samples\fcn_resnet50_voc_520.onnx`。
双击启动后浏览器打开 `http://127.0.0.1:7860`。

重新构建：

```text
D:\anaconda\envs\FCN\python.exe build.py --noconfirm --clean build.spec
```

依赖：

```text
D:\anaconda\envs\FCN\python.exe -m pip install -r requirements-onnx.txt
```

## v1 实现说明

- 预检前做常数折叠：v1 使用自写的 shape 子图常量折叠（`src/onnx_app/constant_fold.py`），
  而不是 onnxsim；当前环境未安装 onnxsim。
- 图编辑：v1 直接操作 ONNX protobuf（`src/onnx_app/pruner.py`），
  没有使用 onnx-graphsurgeon；当前环境未安装 GraphSurgeon。
- 这两者原计划作为 v1 的技术栈，但受当前 FCN 环境限制采用了等价实现。
- v2 不把 GraphSurgeon 作为第一步：优先做校准集激活统计、排序策略评估和未覆盖算子测试；
  GraphSurgeon 作为后续可选编辑层，保持 preflight/grouper/explorer 接口不变。
- 目标 profile 是规则近似（`cpu_ort` / `rknn_like` / `espdl_like`），
  不运行 RKNN/ESP-DL 转换器，也不验证量化后的精度。
- CPU ORT 下 FCN 520x520 单张约 0.8s；完整 39 组 × 5 ratio × 200 张会很久，
  UI 默认缩小子集，建议先跑 20 张 / 单 ratio 或安装 onnxruntime-gpu。
- v1 支持 Conv(groups=1)/BN/Relu/LeakyRelu/Clip/MaxPool/Add/Resize 等；
  feature Concat、Depthwise Conv、Flatten→Gemm 会标 UNRESOLVABLE，不参与剪枝。

## 补验脚本 (v1)

三个脚本都从项目根目录运行，输出 JSON 到 `outputs/`：

```text
D:\anaconda\envs\FCN\python.exe scripts\verify_single_group_20img.py
D:\anaconda\envs\FCN\python.exe scripts\verify_cumulative.py
D:\anaconda\envs\FCN\python.exe scripts\verify_round_to.py
```

- `verify_single_group_20img.py`：固定 20 张/seed=42，剪第一组，输出 `verify_single_group_20img.json`。
- `verify_cumulative.py`：3 组 × 2 ratio，断言前序 mask 生效、累计通道比单调、失败候选回滚；输出 `verify_curve.csv` 和 `verify_cumulative.json`。
- `verify_round_to.py`：对比 cpu_ort(round_to=1) 与 rknn_like(round_to=8)，校验 curve.csv 的 round_to 列；输出 `verify_round_to.json`。


## 曲线怎么读（v1 的边界）

- UI 中的曲线名为 **按当前策略的累计精度曲线**：x 轴是累计通道剪枝率，y 轴是当前顺序逐组剪枝后的累计 mIoU。
- 它不表示“哪一组最不敏感”，也不代表最优剪枝顺序；只表示“按这个顺序剪，累计掉多少”。
- 排序策略提供三种：**L1 升序（默认）**、**随机**、**层深度升序**。切换后重新运行即可对照；CSV 会覆盖，请先下载保存。
- 固定警示：当前按权重 L1 升序逐组剪枝，该策略在部分模型的中高剪枝率下可能导致非单调掉点，结果仅供探索参考，不代表最优剪枝顺序。
- 20% 档的补验已观察到 L1 选通道会造成明显掉点；这是重要性指标问题，不是累计/回滚语义问题。
- v2 计划引入校准集激活统计（替代/补充权重 L1），这是长期方向，不阻塞 v1。

## v1.0.0 定版边界

v1 可以定版，但只在这个边界内：

- 单输入、静态 shape、[1,3,520,520] 输入、[1,21,520,520] 分割 logits；
- 只做 ORT float32 mIoU 与累计剪枝探索；
- 只支持 Conv(groups=1)/BN/ReLU/MaxPool/Add/Resize 等已覆盖结构；
- feature Concat、Depthwise Conv、Flatten→Gemm 标记 UNRESOLVABLE，不参与剪枝；
- 曲线是“按当前策略逐组剪枝后的累计 mIoU”，不是组级敏感度、不是最优顺序、不是目标硬件精度；
- L1 中高剪枝率下的非单调掉点是已知边界，UI 已固定警示；
- 不运行 RKNN/ESP-DL 转换器，不保证 NPU/MCU 量化后的精度。

定版不等于停止改进；以上边界之外的问题不阻塞 v1.0.0。

## v2 建议顺序

1. **校准集激活统计 + 排序评估**（第一优先）：
   用固定校准集的逐通道激活统计（方差/均值/Taylor 近似等）替代或补充权重 L1；
   用 L1/随机/层深度作为对照基线，确认新指标是否真的降低“断崖”。
2. **覆盖与回归测试**：
   为 feature Concat、Depthwise Conv、Flatten→Gemm 建立最小 ONNX fixture 和 golden 检查；
   把 UNRESOLVABLE 从“跳过”推进到“可处理或明确拒绝”。
3. **可选 GraphSurgeon 编辑层**：
   当前 raw protobuf 路径已经验证；GraphSurgeon 主要降低复杂图编辑的维护成本，
   不是 v2 的价值瓶颈。若引入，应放在 GraphEditor 适配层后面，加 golden tests，
   不改变 preflight/grouper/explorer 语义；不建议作为 v2 的第一步。


