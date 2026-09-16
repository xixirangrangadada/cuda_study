# vector_add — T1 元素映射（全库样板 · easy 初版工程）

| 项 | 内容 |
|----|------|
| 分类 | easy（初版工程） |
| 范式 | T1 map：一线程一元素，无通信 |
| 数学定义 | `y[i] = x1[i] + x2[i]` |
| 瓶颈预判 | memory-bound（3n×4 字节 / 次） |
| 状态 | 🟡 naive / optimized 均已跑通（T4/sm_75）；基线见 baseline.md；float4 未提速，结论待写 bench.md |

## 说明

- `design.md` **已填满**，是所有算子纸面推演的样板：线程映射、边界策略、
  带宽记账、可证伪的优化假设（float4 向量化），照这个深度填。
- 代码（`naive.cu` 等）**在首次上机时编写**：本机无 GPU 与 Toolkit，写码必须
  与编译、运行、sanitizer 在同一环境闭环，不预先写未经编译验证的代码。
- 测试与计时代码也在此刻按真实需要建立；只有当至少两个算子重复了同一段
  逻辑，才抽公共头文件（`common/` 的诞生条件，见根 README 纪律）。
## 上机时的步骤（Kaggle，实测 T4 / sm_75）

1. 按 `notebooks/00-kaggle-cuda-environment.ipynb` 记录环境（GPU/驱动/nvcc 版本）。
2. `naive.cu`：kernel + 错误检查 + CPU 参考实现 + 判定
   + 用例集（边界/非整块/空/小/随机；n=0 跳过）。**已跑通**，结果记入 `baseline.md`。
3. `compute-sanitizer --tool memcheck` 通过后才谈性能（本平台待验证）。
4. CUDA events 计时（warmup + repeat），记录 GB/s 与设备峰值之比。
   ⚠️ ncu 被封，性能分析走「cudaEvent + 利用率 + A/B 对比」路线。
5. `optimized.cu`：只改 design.md 假设 1（float4），对比后结论写进 `bench.md`。
6. Thrust `transform` 对照，写 `library.md`；三版数据齐全 → 状态 ✅。

## 参考

- 官方示例：`examples/cuda-samples/cpp/0_Introduction/vectorAdd/`
- 上机流程：`notebooks/00-kaggle-cuda-environment.ipynb`
- 学习计划对应节：`CUDA学习计划.md` §1 T1
