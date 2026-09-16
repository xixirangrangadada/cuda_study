# operator — 个人 CUDA 算子作品集

> 收录本人独立完成的 CUDA 算子实践，按**难度**分级、按**算子种类**分类。
> 难度判据：**并行范式（map / 归约 / 扫描 / 分块 / 原子 / Tensor Core）+ 瓶颈类型（memory / compute）+ 是否融合**。
> 岗位驱动：对齐业界「AI 算子开发 / 异构计算」招聘要求，覆盖**归约/扫描、归一化、矩阵/卷积、注意力全谱、极值、并行前缀、量化/低秩、频域、位置编码、高阶优化、排序搜索、稀疏、迭代聚类、前沿算子**等完整算法域。

## 难度分级

| 档 | 范式 / 模式 | 算子数 | 分类目录 |
|----|------------|:---:|----------|
| **easy** | T1 元素映射（算术 / 激活 / 类型转换 / 复合原语） | 28 | [`easy/`](easy/README.md) |
| **medium** | 归约归一化 + 扫描 + 重排索引 + 排序搜索 + 位置编码 + 随机 | 65 | [`medium/`](medium/README.md) |
| **hard** | GEMM/Tensor Core + 注意力全谱 + 融合 + 量化 + 低秩 + 频域 + 稀疏 + 视觉 + 训练/优化 + 迭代聚类 + 通信 + 原子不规则 + 前沿 | 99 | [`hard/`](hard/README.md) |

**合计 192 个算子骨架。**

> **实现模型**：每个算子卡片含 `实现模型` 字段（SIMT map / warp 原语 / 分块+共享内存 / Tensor Core / 原子）。

## 目录结构

```
operator/
├── README.md
├── easy/                     (28)
│   ├── README.md             ★ 分类 + 清单 + 书写顺序
│   ├── elementwise_arithmetic/{add,mul,sub,div,abs,exp,log,sqrt,pow,clamp,min,max}/
│   ├── activation/{relu,leaky_relu,sigmoid,tanh,gelu,silu,hard_sigmoid,erf}/
│   ├── cast_compare/{cast,compare,select,where}/
│   ├── composite_fused/{axpy,fused_mul_add,mul_add_dst,add_relu}/
│   └── vector_add/           ★ 初版工程（全库样板，未优化，不算完成）
├── medium/                   (65)
│   ├── README.md
│   ├── reduction_normalization/{reduce_sum..welford_finalize}/
│   ├── scan_prefix/{cumsum..cumlogsumexp}/
│   ├── data_rearrange_index/{transpose,layout_convert..deinterleave}/
│   ├── sort_search/{topk,sort,mrg_sort,argsort,searchsorted,kth_value,unique,nonzero,radix_sort,beam_search}/
│   ├── position_encoding/{rope,rope_partial,rope_ntk,alibi,relative_position_encoding,sinusoidal_encoding,learned_position_embedding}/
│   └── random/{philox_random,dropout}/
└── hard/                     (99)
    ├── README.md
    ├── matrix_gemm/{matmul,batch_matmul,grouped_matmul,quant_gemm}/
    ├── attention/{attention,flash_attention,paged_attention,multi_query_attention,group_query_attention,multi_head_latent_attention,linear_attention,sliding_window_attention,attention_backward,ring_attention}/
    ├── fusion/{matmul_activation,swiglu,geglu,qkv_rmsnorm_rope_cache,fused_moe}/
    ├── quantization/{quant,dequant,requant,fixed_quant,fixed_dequant,w8a8,w4a16,w8a16,dynamic_quant,per_token_quant}/
    ├── low_rank/{svd,qr,low_rank_adaptation,matrix_factorization,low_rank_approx}/
    ├── frequency/{fft,ifft,rfft,dct,stft,spectral_conv}/
    ├── sparse/{sparse_matmul,sddmm,sparse_attention,sparse_conv,embedding_bag,sparse_embedding}/
    ├── vision/{conv2d,conv3d,deconv,depthwise_conv,group_conv,pooling,resize,nms,roi_align,grid_sample}/
    ├── training/{cross_entropy,mse_loss,bce_loss,layernorm_grad,gelu_grad,softmax_grad,attention_grad}/
    ├── optimizer/{adam,adam_w,apply_momentum,sgd,rmsprop,adagrad,lamb,lion,gradient_clip}/
    ├── iterative_clustering/{kmeans,knn,conjugate_gradient,jacobi_solver,newton_iteration}/
    ├── communication/{all_reduce,all_gather,reduce_scatter,all_to_all,all_to_all_v,broadcast,reduce,all_gather_mm,reduce_scatter_mm,gemm_allreduce}/
    ├── atomics_irregular/{histogram,hash_table,gather_adds,adaptive_max_pool3d_grad,integer_fast_div}/
    └── frontier/{selective_scan,state_space_model,rwkv,fp8_quant,mx_quant,moe_dispatch,speculative_sampling}/
└── notes/
    └── selection_table.md    # 阶段四产出：范式 → 场景 → 选型对照表（个人见解的落点）
```

> **归档规则**：同一算子的多个变种/版本统一收进同名族目录（如 `medium/reduction_normalization/softmax/`），不平铺。
> **骨架规则**：每个算子骨架 = `README.md`(卡片) + `design.md`(设计模板)；实现时先填 design 再落代码，长出 `naive.cu` / `optimized.cu` / `library.md` / `bench.md`（见 `easy/vector_add/` 样板）。

> **状态说明**：`🔧` = 有初版但**未做优化闭环，不算完成**；`⬜` = 待实现。完成标志见下方纪律。

## 单算子工作流（CUDA 形态，七步）

1. 填 `design.md`：线程映射 / launch config / 内存布局 / 边界策略 / 优化假设。
2. **上机时**写 `naive.cu`：自包含（kernel + CPU 参考 + rtol/atol + 用例集：边界/非整块/空/小/随机）；写码与编译、运行、sanitizer 在同一环境闭环，本地不预写代码。
3. 正确性：CPU 参考对比全用例通过 → `compute-sanitizer`（memcheck/racecheck）干净 → 才谈性能。
4. 性能：CUDA events 计时（warmup + repeat），GB/s 或 GFLOP/s 对设备峰值；瓶颈假设用 Nsight 证实/证伪。
5. `optimized.cu`：一次只改一处，数据落 `bench.md`。
6. `library.md`：与 cuBLAS/CUB/cuDNN/Thrust 对照，解释差距。
7. 三版数据齐全 → 状态 ✅（优化闭环）。

## 书写顺序（各级详见对应 README）

- **easy**：Add → Mul → Sub/Div/Abs/Exp/Log/Sqrt/Pow/Clamp/Min/Max → Cast/Compare/Select/Where → Relu…Erf → Axpy/FusedMulAdd/MulAddDst/AddRelu
- **medium**：归约(ReduceSum…ArgMin) → 归一化(Softmax/LayerNorm/RMSNorm/…) → 扫描(CumSum…CumLogSumExp) → 重排(Transpose…DeInterleave) → 排序搜索(TopK…BeamSearch) → 位置编码(RoPE…LearnedPE) → 随机
- **hard**：MatMul/Batch/Grouped/QuantGemm → 注意力全谱(Attention→Flash/Paged→MQA/GQA/MLA→Linear/Sliding→Backward/Ring) → 融合(SwiGLU/GeGLU/QkvRmsNormRopeCache/FusedMoE) → 量化 → 低秩 → 频域 → 稀疏 → 视觉 → 训练 → 优化器 → 迭代聚类 → 通信 → 原子不规则 → 前沿

## 纪律

1. **算子目录即实验单元**：代码、优化、测试、性能采集、版本迭代、笔记全部在该算子目录内完成，不外溢。
2. **写与优化同地**：`optimized.cu` 与 `naive.cu` 并存，就地留版本、就地实测、就地记数据，形成「改动 → 实测 → 结论」闭环（`bench.md`）。
3. 新算子 → 归入对应难度/分类骨架目录，并在对应级别 README 登记。
4. 难度升级 → 移动到更高档，并在该算子目录的 design.md 中说明理由。
5. 板上 / 云端副本分叉时，先比 mtime 再取舍，仲裁结论写进该算子目录。
6. **正确性先于性能**：sanitizer 不干净的 kernel 不测性能；每次上机先记录 GPU/驱动/Toolkit 版本。
7. **不预写基建**：测试/计时/公共头文件在实现算子的真实编译需求中出现；`common/` 只有在至少两个算子重复了同一段逻辑后才建立。未经编译验证的"框架"代码一律不落盘。本地推演结论在上机前不得写成实测。

## 环境

本地无 GPU 与 Toolkit：纸面推演、CPU 参考实现、代码走查。真机验证集中在 **Kaggle（Tesla T4，sm_75，约 30h/周）**；编译参数 `-arch` 以 `deviceQuery`/`nvidia-smi` 实测为准。样板工程：[`easy/vector_add/`](easy/vector_add/README.md)。
