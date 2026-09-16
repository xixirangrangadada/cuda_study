# T1 元素映射 · vector_add 设计文档

> 本文件是**已填写完成的样板**：其他算子的 design.md 照这个深度填。

## 1. 需求与接口
- 数学定义：`y[i] = x1[i] + x2[i]`，i ∈ [0, n)
- 输入 / 输出：`float* x1, x2`（读）、`float* y`（写），连续一维，行主序无影响
- 用例集：`test_sizes(1<<20)` = {0, 1, 31, 255, 256, 257, 1000, 4096, 1M}
  - 0：空输入（不 launch，直接对比空向量）
  - 31/255/257：非整块（边界判断生效的证明）
  - 256：恰好整块（边界分支从未触发的对照组）
  - 1000/4K/1M：常规与较大规模

## 2. 线程映射与 Launch 配置
- 1:1 映射：`i = blockIdx.x * blockDim.x + threadIdx.x`，`blocks = ceil(n/256)`
- blockDim = 256：每 block 8 个 warp，占用与 launch 开销的常见折中（256/512 都可，bench 可证）
- 边界策略：`if (i < n)` 块内尾部线程空转；n=0 时 blocks=0，跳过 launch
- grid 上限：1D grid 最大 2^31-1 块，n 上限 256×2^31 ≈ 5×10^11，float 数组先到显存上限，
  本算子不需要 grid-stride；**换更大规模数据结构时要重算**（这是适用边界）

## 3. 内存布局与访存
- 三个数组各自连续，相邻线程访问相邻地址 → 天然合并访存
- shared memory：无（逐元素无复用）
- 向量化：`float4` 可行（4 元素/线程），n 非 4 倍数时尾部单独处理 —— 留给 optimized.cu

## 4. 同步与正确性
- 无 `__syncthreads`（线程间独立）、无原子、无跨块依赖
- 两处错误检查：launch 后 `CUDA_CHECK_LAST()`；同步后查 `cudaDeviceSynchronize` 返回
- sanitizer：memcheck（越界）+ racecheck（本算子应无竞争，跑一遍作为习惯）

## 5. 精度方案
- CPU 参考：同式逐元素相加（同为 float，舍入一致）
- 判定：rtol=1e-6, atol=1e-6（同式运算，理应逐位相等；放宽到 1e-6 防编译器重排）
- 特殊值：随机数据 [-2,2] 不产生溢出；除零/NaN 不适用于本算子

## 6. 优化假设（写给 `optimized.cu`）
- 假设 1：naive 版带宽未打满设备峰值（实测环境 **T4，参考 ~320 GB/s**，见 environment.md；
  基线实测 n=1<<20 利用率 **72.7%**，见 baseline.md），瓶颈在每线程 1 元素的
  指令/访存请求数 → 改 `float4` 向量化，预期 GB/s 提升；验证指标：耗时 + 实际带宽利用率
  （**ncu 在本平台被封，改用 cudaEvent + 利用率**，原计划指标
  `dram__throughput.avg.pct_of_peak_sustained_elapsed` 留待有权限环境再采）
- 假设 2：n 较小时 launch 开销占比大 → 同 kernel 无法改善，属固有下限；
  记录小 n 的耗时下限作为"launch 税"基线（**实测 ≈ 0.006 ms**，见 baseline.md），供后续算子引用

## 7. 库版对照（写给 `library.md`）
- Thrust：`thrust::transform(a, a+n, b, c, thrust::plus<float>())`
- 对照口径：同 n、同数据种子、预热后 CUDA events 计时，比较 GB/s 与代码量
