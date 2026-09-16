# Adam（`hard/optimizer/adam`）

| 项 | 内容 |
|----|------|
| 分类 | optimizer |
| 难度 | hard |
| 实现模型 | SIMT map（状态更新） |
| 数学定义 | `一阶/二阶动量更新` |
| 输入 / 输出 | 见 design.md（实现时定 dtype/shape） |
| 关键考点 | 见下 |
| 书写顺序 | 本级第 64 题 |
| 状态 | ⬜ |

## 关键考点（CUDA）

- 逐元素状态更新：参数/动量/方差的读写带宽记账
- 多组张量同尺寸遍历：一个 kernel 全吃
- 混合精度下的主/从权重（fp32 master + fp16 计算）

## 说明

- 本目录为**算子骨架**：`README.md`(卡片) + `design.md`(设计模板)；实现时先填 design，再长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（工作流见[根 README](../../README.md)）。
- 验收：CPU golden 通过 + compute-sanitizer 通过；**完成以优化闭环为标志**（三版对比有数据）。

