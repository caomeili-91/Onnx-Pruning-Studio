# ONNX 静态图剪枝可行性分析平台 (v1.0.0)

产品定位：**静态 CNN 分割 ONNX 的规则预检 + 逐组剪枝敏感度分析平台。**

v1 只接受：

- 输入 [1,3,520,520]、float32 的 ONNX；
- 输出 [1,21,520,520] 的 VOC 21 类语义分割 logits；
- 评估指标为 ONNX Runtime float32 下的 mIoU。

> 预检为基于 target profile 的静态规则判断，不运行厂商转换器；mIoU 来自 ONNX Runtime float32 推理，不代表 NPU/MCU 量化后精度。

## v1.0.0 状态

v1 已具备定版条件。定版范围是“静态 520x520 VOC 分割 ONNX 的规则预检 + 累计精度分析工具”，不是目标硬件交付工具，也不承诺剪枝后的目标硬件精度。

已验收：

- UI 极简版完成：上传即预检，通过/拒绝红绿色展示，一键开始分析，曲线为主视图，日志默认收起。
- psutil 7.2.2 已随 exe 打包并生效；示例 baseline 的 RSS 为 592.7 MB -> 699.0 MB，写入 ui_baseline.json / rss.log。
- 完整 L1 探索实测 689s；曲线非单调/断崖掉点已在发布说明中标注。
- exe clean build 完成；HTTP 200；打包内预检 status=OK；小规模探索 smoke test 通过。
- 打包目录包含 VOCdevkit 与 samples/fcn_resnet50_voc_520.onnx。
- GitHub 仓库已完成首次提交与推送。

## 功能概览

- ONNX 静态图预检：输入/输出契约、动态 shape、QDQ、IR/opset、算子白名单、target profile 资源粗估。
- 通道分组：Conv/BN/ReLU/MaxPool/Add/Resize 等结构贯通，最终输出头保护。
- 逐组累计剪枝：每步从原始 folded ONNX + 已接受 mask 重建候选，不落盘中间模型。
- ONNX Runtime float32 mIoU 评估，支持固定子集与 seed。
- CSV 输出：累计通道比、参数量、FLOPs、峰值激活估算、mIoU、delta、状态、profile/model 上下文。
- UI 极简版：上传、预检、开始分析、累计 mIoU 曲线、可折叠日志。
- 补验脚本：单组对照、累计语义断言、round_to 对齐检查。

## 运行环境

已验证环境：

    操作系统: Windows
    Python:   3.10
    onnx:     1.23.2
    onnxruntime: 1.23.2 (CPU)
    gradio:   6.32.0
    numpy:    1.26.4
    psutil:   7.2.2

安装依赖：

    D:\anaconda\envs\FCN\python.exe -m pip install -r requirements-onnx.txt

> requirements-onnx.txt 中 ml-dtypes 固定为小于 0.6，避免与 numpy 1.26.4 的 pip check 冲突。当前环境使用 CPU ONNX Runtime，没有 CUDA provider。

## 数据准备

评估需要 VOC2012 的验证集，目录如下：

    VOCdevkit/
      VOC2012/
        JPEGImages/
        SegmentationClass/
        ImageSets/Segmentation/val.txt

VOCdevkit 不进入 Git 仓库，需要本地准备；已构建的 dist 目录中自带一份，可直接运行。

## 快速开始

### 1. 源码运行

    cd /d "D:\python project\first_app"
    D:\anaconda\envs\FCN\python.exe app.py

浏览器打开：

    http://127.0.0.1:7860

### 2. 打包 exe

Git 仓库不跟踪 build.spec，先复制示例：

    copy build.spec.example build.spec

然后构建：

    D:\anaconda\envs\FCN\python.exe build.py --noconfirm --clean build.spec

产物：

    dist\OnnxPruneStudio\OnnxPruneStudio.exe

### 3. 直接使用已构建版本

    dist\OnnxPruneStudio\OnnxPruneStudio.exe

同目录已包含：

- VOCdevkit/
- samples/fcn_resnet50_voc_520.onnx
- 运行所需依赖

## UI 操作

最终界面只保留必要操作：

1. 上传 ONNX 模型。
2. 上传完成后自动预检：
   - 绿色：预检通过；
   - 红色：预检拒绝，显示主要原因和下一步。
3. 点击“开始分析”。
4. 查看主视图“累计 mIoU 曲线”。
5. 需要排查时展开“运行日志”；需要改参数时展开“高级设置”。

UI 固定项：

- target profile：固定 cpu_ort，不展示；
- 排序：固定默认排序，不展示；
- 随机种子：固定 42，不展示；
- 评估输出名：自动选择第一个合法的 [1,21,520,520] 输出，不展示；
- VOC 路径：在“高级设置”中，默认 VOCdevkit。

## 目录结构

    app.py                        极简 Gradio UI
    src/onnx_app/
      preflight.py                规则预检
      constant_fold.py            shape 子图常量折叠
      grouper.py                  通道依赖分组
      pruner.py                   ONNX protobuf 图编辑
      evaluator.py                ORT VOC mIoU 评估
      explorer.py                 逐组累计探索 + CSV
      profile_loader.py           target profile 加载
      profiles/                   cpu_ort / rknn_like / espdl_like
    scripts/
      verify_single_group_20img.py
      verify_cumulative.py
      verify_round_to.py
    requirements-onnx.txt          推理/UI 依赖
    requirements.txt               旧版训练依赖（非 UI 必需）
    build.spec.example             PyInstaller 示例配置
    model/                         旧版训练模型定义
    src/core/ src/eval/ src/models/ 旧版 PyTorch 训练/评估代码

说明：v1 产品链路只依赖 src/onnx_app 与 app.py。旧版 PyTorch 训练代码保留用于锚定模型导出，不参与 UI 运行。

## 补验脚本

三个脚本都从项目根目录运行，输出到 outputs/：

    D:\anaconda\envs\FCN\python.exe scripts\verify_single_group_20img.py
    D:\anaconda\envs\FCN\python.exe scripts\verify_cumulative.py
    D:\anaconda\envs\FCN\python.exe scripts\verify_round_to.py

- verify_single_group_20img.py：固定 20 张 / seed=42，剪第一组，输出 verify_single_group_20img.json。
- verify_cumulative.py：3 组 x 2 ratio，断言前序 mask 生效、累计通道比单调、失败候选回滚；输出 verify_curve.csv 和 verify_cumulative.json。
- verify_round_to.py：对比 cpu_ort(round_to=1) 与 rknn_like(round_to=8)，校验 curve.csv 的 round_to 列；输出 verify_round_to.json。

注意：outputs/ 不进入 Git 仓库，补验结果需要本地重跑生成。

## 输出说明

UI 结果 CSV 包含：

    step, group_id, group_nodes, nominal_ratio, actual_channel_ratio,
    cumulative_channel_ratio, cumulative_param_ratio, round_to,
    miou, delta_vs_baseline, params, flops_est, peak_act_est,
    status, error_msg, elapsed_ms, profile_name, profile_version,
    model_hash, input_signature, provider, sort_strategy, sort_seed

其中：

- miou / delta_vs_baseline 单位为百分比（%）。
- cumulative_channel_ratio / cumulative_param_ratio 为 0-1 比例。
- flops_est / peak_act_est 为粗略估算，不是厂商工具结果。
- sort_strategy 记录本次运行的组排序；UI 极简版固定默认排序。
- rss.log 单独记录探索前后和每步的 RSS。

## 曲线怎么读

- 曲线是“按当前策略逐组剪枝后的累计 mIoU”，不是组级敏感度，也不是最优剪枝顺序。
- x 轴是累计通道剪枝率，y 轴是累计 mIoU。
- 曲线出现非单调或断崖，不代表累计/回滚语义有 bug；可能是排序策略或重要性指标选到了破坏性通道。
- v1 的验证已观察到 L1 在 20% 档可能选出破坏性通道；这是已知边界，UI 极简版不再展示排序词，发布说明保留该风险。
- v2 第一优先是校准集激活统计，不是更换图编辑库。

## 当前边界

- 单输入、静态 shape、[1,3,520,520] 输入、[1,21,520,520] 分割 logits；
- 只做 ORT float32 mIoU 与累计剪枝探索；
- 支持 Conv(groups=1)/BN/ReLU/MaxPool/Add/Resize 等已覆盖结构；
- feature Concat、Depthwise Conv、Flatten→Gemm 标记 UNRESOLVABLE，不参与剪枝；
- 不支持 QDQ / 动态 shape / 控制流 / 多输入；
- 不运行 RKNN/ESP-DL 转换器，不保证 NPU/MCU 量化后的精度；
- CPU ORT 下 FCN 520x520 单张约 0.8s，完整曲线耗时较长，建议先缩小评估子集。

## v2 路线

1. 校准集激活统计 + 排序评估（第一优先）；
2. feature Concat / Depthwise Conv / Flatten→Gemm 的覆盖与 golden tests；
3. 可选 GraphSurgeon 编辑层，放在 GraphEditor 适配层之后，不作为第一步。

## 技术栈说明

v1 使用：

- 自写 shape 子图常量折叠（constant_fold.py），未使用 onnxsim；
- 直接操作 ONNX protobuf（pruner.py），未使用 onnx-graphsurgeon；
- ONNX Runtime CPU float32 评估；未接 CUDA / NPU runtime。

这是当前 FCN 环境下的等价实现。v2 可替换，但需要保持 preflight / grouper / explorer 的语义不变。

## 常见问题

### 上传 125MB 模型被预检拒绝

旧版 UI 曾在 Gradio 临时文件未写完时读取，导致 ONNX protobuf 截断。v1.0.0 已改为上传完成事件 + 文件大小稳定检查，并兼容路径/bytes/FileData 三种输入；上传上限设置为 500MB。

### 为什么 mIoU 和 PyTorch 结果不完全一致

v1 使用固定 520x520 stretch 预处理 + ONNX Runtime float32 推理；PyTorch 端的 resize 实现可能不同。锚定模型验收时 argmax agreement 为 99.98%，mIoU 以 ONNX 侧为准。

### 为什么 build.spec 不在仓库里

.gitignore 按项目要求忽略了 *.spec。仓库保留 build.spec.example，构建前复制为 build.spec 即可。

## 版本

- v1.0.0
- 定版日期：2026-11-14
- 仓库整理：2026-11-15
- 初始提交：7a76758
