# 上机环境快照 — Kaggle

> 关卡纪律 6：每次上机先记录 GPU/驱动/Toolkit 版本；环境以实测为准。
> Kaggle 免费档 GPU 池会变动，**不同会话可能拿到不同卡**，务必每次重新记录。

---

## 当前环境（2026-09-16）

- 日期：2026-09-16
- 平台：Kaggle Notebook GPU session
- GPU：**Tesla T4 × 2**（Turing TU104），compute capability **7.5**（实际使用 device 0）
- 显存：15360 MiB / 卡，GDDR6，**256-bit**，显存最高频率 **5001 MHz**
- **峰值带宽：≈ 320 GB/s**（5001 × 2 × 256 / 8 / 1000 = 320.06）
- 驱动：580.159.04（`nvidia-smi` 显示 CUDA 13.0 = 驱动支持上限，非 toolkit 版本）
- 编译参数：**`-arch=sm_75`**
- 工具链：
  - **ncu ❌**：`ERR_NVGPUCTRPERM`，普通用户无性能计数器权限（容器限制）
  - compute-sanitizer / cuda-gdb：待验证
- host：Ubuntu（Kaggle 容器）

---

## 历史记录（2026-09-14，P100）

- GPU：Tesla P100-PCIE-16GB（Pascal GP100），sm_60，16 GB HBM2，≈ **732 GB/s**
- Toolkit：CUDA 12.8（nvcc V12.8.93），`-arch=sm_60`
- 工具链：compute-sanitizer ✅ / ncu ✅ / cuda-gdb ✅ / nsys ❌

---

## 环境对策略的影响

| 项 | P100（旧） | T4（当前） | 影响 |
|---|---|---|---|
| 架构 | sm_60 | **sm_75** | 编译 `-arch=sm_75` |
| 峰值带宽 | ~732 GB/s | **~320 GB/s** | 「打满」标准降低 |
| 算力/带宽比 | ~12.7 | ~25.3 | T4 访存更易成瓶颈，更适合练访存优化 |
| ncu | ✅ 可用 | ❌ 被封 | 改用 `cudaEvent + 利用率 + A/B 对比` |

**结论：** design.md 的映射/边界/用例结构不变，环境常数按 **T4** 修正；
性能分析路线从「ncu 定位」改为「cudaEvent 计时 + 利用率推算」。
