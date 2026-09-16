# AttentionBackward（`hard/attention/attention_backward`）

| 项 | 内容 |
|----|------|
| 分类 | attention |
| 难度 | hard |
| 实现模型 | 分块 + online softmax |
| 数学定义 | `注意力反向` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 13 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- naive 的中间矩阵（S、P）显存与访存记账；FLOPs/字节 证明 memory-bound
- flash 思路：分块 + online softmax（running max 与 rescale 先手算两块例子）
- KV 分页（paged）、共享 KV（MQA/GQA）、低秩压缩（MLA）各自的访存变化
- 反向与环形（ring）序列切分：多卡视角（衔接 AI-Infra 路线）

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

## 参考

- PyTorch F.scaled_dot_product_attention（library 对照）
