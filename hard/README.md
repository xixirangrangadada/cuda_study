# hard — 高级：复用/融合/不规则（99 题）

> 高级档。分块复用、Tensor Core、融合、量化与不规则并行；矩阵/注意力/通信等工业级主题，全部要求三版对比闭环。

**算子总数：99**

## 分类总览

| 类别 | 目录 | 算子数 | 代表算子 |
|------|------|:---:|------|
| 矩阵/GEMM | `matrix_gemm/` | 4 | MatMul / GroupedMatMul / QuantGemm |
| 注意力与高效变体 | `attention/` | 10 | Attention / MLAttention / RingAttention |
| 融合算子 | `fusion/` | 5 | MatmulActivation / GeGLU / FusedMoE |
| 量化/推理 | `quantization/` | 10 | Quant / W8A8 / PerTokenQuant |
| 低秩与矩阵分解 | `low_rank/` | 5 | SVD / LoRA / LowRankApprox |
| 频域方法 | `frequency/` | 6 | FFT / DCT / SpectralConv |
| 稀疏计算 | `sparse/` | 6 | SparseMatMul / SparseConv / SparseEmbedding |
| 视觉/多模态 | `vision/` | 10 | Conv2D / Pooling / GridSample |
| 训练侧 | `training/` | 7 | CrossEntropy / LayerNormGrad / AttentionGrad |
| 高阶优化器 | `optimizer/` | 9 | Adam / RMSprop / GradientClip |
| 迭代与聚类 | `iterative_clustering/` | 5 | KMeans / ConjugateGradient / NewtonIteration |
| 通信/并行 | `communication/` | 10 | AllReduce / Broadcast(comm) / GemmAllReduce |
| 原子/不规则并行 | `atomics_irregular/` | 5 | Histogram / GatherAdds / IntegerFastDiv |
| 前沿算子 | `frontier/` | 7 | SelectiveScan / FP8Quant / SpeculativeSampling |

## 各类算子清单

### 矩阵/GEMM（`matrix_gemm/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 1 | **MatMul** | `matrix_gemm/matmul/` | `C = A x B (+ bias)` | ⬜ |
| 2 | **BatchMatMul** | `matrix_gemm/batch_matmul/` | `C_b = A_b x B_b` | ⬜ |
| 3 | **GroupedMatMul** | `matrix_gemm/grouped_matmul/` | `C_g = A_g x B_g` | ⬜ |
| 4 | **QuantGemm** | `matrix_gemm/quant_gemm/` | `C = dequant(A) x dequant(B)` | ⬜ |

### 注意力与高效变体（`attention/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 5 | **Attention** | `attention/attention/` | `O = softmax(QK^T/sqrt(d))V` | ⬜ |
| 6 | **FlashAttention** | `attention/flash_attention/` | `online-softmax 分块注意力` | ⬜ |
| 7 | **PagedAttention** | `attention/paged_attention/` | `KV 分页注意力` | ⬜ |
| 8 | **MultiQueryAttention** | `attention/multi_query_attention/` | `Q 多头共享单组 KV` | ⬜ |
| 9 | **GroupQueryAttention** | `attention/group_query_attention/` | `Q 分组共享 KV` | ⬜ |
| 10 | **MLAttention** | `attention/multi_head_latent_attention/` | `低秩压缩 KV (MLA)` | ⬜ |
| 11 | **LinearAttention** | `attention/linear_attention/` | `核化线性注意力` | ⬜ |
| 12 | **SlidingWindowAttention** | `attention/sliding_window_attention/` | `滑窗局部注意力` | ⬜ |
| 13 | **AttentionBackward** | `attention/attention_backward/` | `注意力反向` | ⬜ |
| 14 | **RingAttention** | `attention/ring_attention/` | `序列维切分环形注意力` | ⬜ |

### 融合算子（`fusion/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 15 | **MatmulActivation** | `fusion/matmul_activation/` | `y = act(A x B)` | ⬜ |
| 16 | **SwiGLU** | `fusion/swiglu/` | `y = Swish(x W1) * (x W3)` | ⬜ |
| 17 | **GeGLU** | `fusion/geglu/` | `y = GELU(x W1) * (x W3)` | ⬜ |
| 18 | **QkvRmsNormRopeCache** | `fusion/qkv_rmsnorm_rope_cache/` | `QKV+RMSNorm+RoPE+Cache 融合` | ⬜ |
| 19 | **FusedMoE** | `fusion/fused_moe/` | `y = sum_g gate_g * Expert_g(x)` | ⬜ |

### 量化/推理（`quantization/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 20 | **Quant** | `quantization/quant/` | `round(x/scale) + zp` | ⬜ |
| 21 | **Dequant** | `quantization/dequant/` | `(x - zp) * scale` | ⬜ |
| 22 | **Requant** | `quantization/requant/` | `量化值域再打包` | ⬜ |
| 23 | **FixedQuant** | `quantization/fixed_quant/` | `定点量化` | ⬜ |
| 24 | **FixedDequant** | `quantization/fixed_dequant/` | `定点反量化` | ⬜ |
| 25 | **W8A8** | `quantization/w8a8/` | `int8 矩阵乘 + 反量化` | ⬜ |
| 26 | **W4A16** | `quantization/w4a16/` | `4bit 权重 + 16bit 激活` | ⬜ |
| 27 | **W8A16** | `quantization/w8a16/` | `8bit 权重 + 16bit 激活` | ⬜ |
| 28 | **DynamicQuant** | `quantization/dynamic_quant/` | `运行时动态 scale` | ⬜ |
| 29 | **PerTokenQuant** | `quantization/per_token_quant/` | `按 token 粒度量化` | ⬜ |

### 低秩与矩阵分解（`low_rank/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 30 | **SVD** | `low_rank/svd/` | `A = U S V^T` | ⬜ |
| 31 | **QR** | `low_rank/qr/` | `A = Q R` | ⬜ |
| 32 | **LoRA** | `low_rank/low_rank_adaptation/` | `y = x W + (alpha/r) x A B` | ⬜ |
| 33 | **MatrixFactorization** | `low_rank/matrix_factorization/` | `A ≈ U V^T` | ⬜ |
| 34 | **LowRankApprox** | `low_rank/low_rank_approx/` | `截断低秩近似` | ⬜ |

### 频域方法（`frequency/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 35 | **FFT** | `frequency/fft/` | `y = FFT(x)` | ⬜ |
| 36 | **IFFT** | `frequency/ifft/` | `y = IFFT(X)` | ⬜ |
| 37 | **RFFT** | `frequency/rfft/` | `实数输入 FFT` | ⬜ |
| 38 | **DCT** | `frequency/dct/` | `离散余弦变换` | ⬜ |
| 39 | **STFT** | `frequency/stft/` | `短时傅里叶变换` | ⬜ |
| 40 | **SpectralConv** | `frequency/spectral_conv/` | `频域卷积 (FFT 定理)` | ⬜ |

### 稀疏计算（`sparse/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 41 | **SparseMatMul** | `sparse/sparse_matmul/` | `稀疏矩阵乘 (SpMM)` | ⬜ |
| 42 | **SDDMM** | `sparse/sddmm/` | `稀疏采样矩阵乘` | ⬜ |
| 43 | **SparseAttention** | `sparse/sparse_attention/` | `稀疏注意力` | ⬜ |
| 44 | **SparseConv** | `sparse/sparse_conv/` | `稀疏卷积` | ⬜ |
| 45 | **EmbeddingBag** | `sparse/embedding_bag/` | `变长查表求和/均值` | ⬜ |
| 46 | **SparseEmbedding** | `sparse/sparse_embedding/` | `稀疏梯度嵌入` | ⬜ |

### 视觉/多模态（`vision/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 47 | **Conv2D** | `vision/conv2d/` | `y = sum_k x*w + b` | ⬜ |
| 48 | **Conv3D** | `vision/conv3d/` | `3D 卷积` | ⬜ |
| 49 | **Deconv** | `vision/deconv/` | `转置卷积` | ⬜ |
| 50 | **DepthwiseConv** | `vision/depthwise_conv/` | `逐通道卷积` | ⬜ |
| 51 | **GroupConv** | `vision/group_conv/` | `分组卷积` | ⬜ |
| 52 | **Pooling** | `vision/pooling/` | `max/avg(window)` | ⬜ |
| 53 | **Resize** | `vision/resize/` | `双线性/最近邻插值` | ⬜ |
| 54 | **NMS** | `vision/nms/` | `按 IoU 抑制重叠框` | ⬜ |
| 55 | **RoiAlign** | `vision/roi_align/` | `ROI 双线性池化` | ⬜ |
| 56 | **GridSample** | `vision/grid_sample/` | `按坐标网格采样` | ⬜ |

### 训练侧（`training/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 57 | **CrossEntropy** | `training/cross_entropy/` | `L = -sum y*log(softmax(x))` | ⬜ |
| 58 | **MSELoss** | `training/mse_loss/` | `mean((x-y)^2)` | ⬜ |
| 59 | **BCELoss** | `training/bce_loss/` | `-[y log p +(1-y)log(1-p)]` | ⬜ |
| 60 | **LayerNormGrad** | `training/layernorm_grad/` | `LayerNorm 反向` | ⬜ |
| 61 | **GeluGrad** | `training/gelu_grad/` | `GELU 反向` | ⬜ |
| 62 | **SoftmaxGrad** | `training/softmax_grad/` | `Softmax 反向` | ⬜ |
| 63 | **AttentionGrad** | `training/attention_grad/` | `Attention 反向` | ⬜ |

### 高阶优化器（`optimizer/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 64 | **Adam** | `optimizer/adam/` | `一阶/二阶动量更新` | ⬜ |
| 65 | **AdamW** | `optimizer/adam_w/` | `Adam + 解耦权重衰减` | ⬜ |
| 66 | **ApplyMomentum** | `optimizer/apply_momentum/` | `动量 SGD` | ⬜ |
| 67 | **SGD** | `optimizer/sgd/` | `随机梯度下降` | ⬜ |
| 68 | **RMSprop** | `optimizer/rmsprop/` | `均方根传播` | ⬜ |
| 69 | **Adagrad** | `optimizer/adagrad/` | `累积梯度平方` | ⬜ |
| 70 | **LAMB** | `optimizer/lamb/` | `分层自适应大批量优化` | ⬜ |
| 71 | **Lion** | `optimizer/lion/` | `符号动量优化` | ⬜ |
| 72 | **GradientClip** | `optimizer/gradient_clip/` | `梯度裁剪` | ⬜ |

### 迭代与聚类（`iterative_clustering/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 73 | **KMeans** | `iterative_clustering/kmeans/` | `迭代聚类` | ⬜ |
| 74 | **KNN** | `iterative_clustering/knn/` | `k 近邻` | ⬜ |
| 75 | **ConjugateGradient** | `iterative_clustering/conjugate_gradient/` | `共轭梯度解 Ax=b` | ⬜ |
| 76 | **JacobiSolver** | `iterative_clustering/jacobi_solver/` | `Jacobi 迭代` | ⬜ |
| 77 | **NewtonIteration** | `iterative_clustering/newton_iteration/` | `牛顿迭代` | ⬜ |

### 通信/并行（`communication/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 78 | **AllReduce** | `communication/all_reduce/` | `y = sum_r x_r` | ⬜ |
| 79 | **AllGather** | `communication/all_gather/` | `y = concat_r x_r` | ⬜ |
| 80 | **ReduceScatter** | `communication/reduce_scatter/` | `y_r = sum(x_r) 分片` | ⬜ |
| 81 | **AllToAll** | `communication/all_to_all/` | `各 rank 全交换` | ⬜ |
| 82 | **AllToAllV** | `communication/all_to_all_v/` | `变长 AllToAll` | ⬜ |
| 83 | **Broadcast(comm)** | `communication/broadcast/` | `根 rank 广播` | ⬜ |
| 84 | **Reduce** | `communication/reduce/` | `归约到根 rank` | ⬜ |
| 85 | **AllGatherMM** | `communication/all_gather_mm/` | `AllGather + MatMul 融合` | ⬜ |
| 86 | **ReduceScatterMM** | `communication/reduce_scatter_mm/` | `MatMul + ReduceScatter 融合` | ⬜ |
| 87 | **GemmAllReduce** | `communication/gemm_allreduce/` | `Matmul + 集合通信融合` | ⬜ |

### 原子/不规则并行（`atomics_irregular/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 88 | **Histogram** | `atomics_irregular/histogram/` | `按值域统计频次` | ⬜ |
| 89 | **HashTable** | `atomics_irregular/hash_table/` | `键值映射` | ⬜ |
| 90 | **GatherAdds** | `atomics_irregular/gather_adds/` | `y = gather(x, idx) + adds` | ⬜ |
| 91 | **AdaptiveMaxPool3DGrad** | `atomics_irregular/adaptive_max_pool3d_grad/` | `3D 自适应最大池化反向` | ⬜ |
| 92 | **IntegerFastDiv** | `atomics_irregular/integer_fast_div/` | `整数快速除法近似` | ⬜ |

### 前沿算子（`frontier/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 93 | **SelectiveScan** | `frontier/selective_scan/` | `Mamba 选择性扫描` | ⬜ |
| 94 | **StateSpaceModel** | `frontier/state_space_model/` | `状态空间模型` | ⬜ |
| 95 | **RWKV** | `frontier/rwkv/` | `RWKV 线性注意力/递推` | ⬜ |
| 96 | **FP8Quant** | `frontier/fp8_quant/` | `FP8 量化` | ⬜ |
| 97 | **MXQuant** | `frontier/mx_quant/` | `MX 微缩放格式量化` | ⬜ |
| 98 | **MoEDispatch** | `frontier/moe_dispatch/` | `MoE token 分发/收集` | ⬜ |
| 99 | **SpeculativeSampling** | `frontier/speculative_sampling/` | `投机采样` | ⬜ |

## 建议书写顺序

按上表序号顺序推进；同类内先做代表算子打通结构，再做变体练复用。
初版工程 [`vector_add/`](vector_add/README.md) 是全库样板（easy 档）。
