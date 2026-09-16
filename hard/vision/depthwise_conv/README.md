# DepthwiseConv（`hard/vision/depthwise_conv`）

| 项 | 内容 |
|----|------|
| 分类 | vision |
| 难度 | hard |
| 实现模型 | 滑窗 + 分块 |
| 数学定义 | `逐通道卷积` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 50 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- 直接卷积的访存放大 vs im2col 的内存放大（两个账本都算）
- NHWC vs NCHW 在 GPU 上的真实差别
- pooling/resize 的边界插值分支；NMS 的迭代抑制结构

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

## 参考

- cuDNN（library 对照）
