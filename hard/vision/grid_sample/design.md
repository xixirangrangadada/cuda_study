# GridSample 设计文档

> 骨架占位。实现前先补齐以下各节（样板见 `easy/vector_add/design.md`）。

## 1. 需求与接口
- 数学定义：`按坐标网格采样`
- 输入 / 输出（dtype / shape / 布局）：
- 用例集（边界 / 非整块 / 空 / 小 / 随机）：

## 2. 线程映射与 Launch 配置
- grid / block 维度选择及理由：
- 1:1 映射还是 grid-stride：
- 边界与尾块策略：

## 3. 内存布局与访存
- 读 / 写访问模式（地址序列手推一遍，是否合并访存）：
- shared memory 用量与生命周期（不用写"无"）：
- 向量化（float4 / half2）与 cp.async 可行性：

## 4. Kernel 结构与同步
- 计算步骤分解：
- `__syncthreads` / 原子 / 跨块同步需求：
- 竞态排查点（racecheck 先行）：

## 5. 精度方案
- CPU 参考实现写法：
- rtol / atol 与特殊值（除零、溢出、NaN）：

## 6. 测试与验收
- golden 与用例清单：
- 性能基线与指标（GB/s 或 GFLOP/s）：

## 7. 优化假设（写给 `optimized.cu`）
- 假设 1（瓶颈猜测 + ncu 验证指标）：
- 假设 2：

## 8. 库版对照（写给 `library.md`）
- 对应库与 API：
- 对照口径（同数据、同精度、预热后计时）：
