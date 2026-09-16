# medium — 协作范式进阶（65 题）

> 进阶档。引入 warp 原语（shuffle）、树形归约、扫描、分治与原子：线程开始协作，同步与竞争成为主要考点。

**算子总数：65**

## 分类总览

| 类别 | 目录 | 算子数 | 代表算子 |
|------|------|:---:|------|
| 归约与归一化 | `reduction_normalization/` | 23 | ReduceSum / LogSoftmax / WelfordFinalize |
| 扫描/前缀算法 | `scan_prefix/` | 8 | CumSum / InclusiveScan / CumLogSumExp |
| 数据重排与索引 | `data_rearrange_index/` | 15 | Transpose / Split / DeInterleave |
| 排序与搜索 | `sort_search/` | 10 | TopK / KthValue / BeamSearch |
| 位置编码 | `position_encoding/` | 7 | RoPE / ALiBi / LearnedPE |
| 随机 | `random/` | 2 | PhiloxRandom / DropOut / DropOut |

## 各类算子清单

### 归约与归一化（`reduction_normalization/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 1 | **ReduceSum** | `reduction_normalization/reduce_sum/` | `sum(x, axis)` | ⬜ |
| 2 | **ReduceMax** | `reduction_normalization/reduce_max/` | `max(x, axis)` | ⬜ |
| 3 | **ReduceMin** | `reduction_normalization/reduce_min/` | `min(x, axis)` | ⬜ |
| 4 | **ReduceProd** | `reduction_normalization/reduce_prod/` | `prod(x, axis)` | ⬜ |
| 5 | **ReduceMean** | `reduction_normalization/reduce_mean/` | `mean(x, axis)` | ⬜ |
| 6 | **ReduceAll** | `reduction_normalization/reduce_all/` | `all(x!=0, axis)` | ⬜ |
| 7 | **ReduceAny** | `reduction_normalization/reduce_any/` | `any(x!=0, axis)` | ⬜ |
| 8 | **ArgMax** | `reduction_normalization/argmax/` | `argmax(x, axis)` | ⬜ |
| 9 | **ArgMin** | `reduction_normalization/argmin/` | `argmin(x, axis)` | ⬜ |
| 10 | **Mean** | `reduction_normalization/mean/` | `mean(x)` | ⬜ |
| 11 | **LogSumExp** | `reduction_normalization/logsumexp/` | `log(sum(exp(x)))` | ⬜ |
| 12 | **LogSoftmax** | `reduction_normalization/log_softmax/` | `x - max - log(sum exp(x-max))` | ⬜ |
| 13 | **Softmax** | `reduction_normalization/softmax/` | `exp(x_i-max)/sum exp(x_j-max)` | ⬜ |
| 14 | **LayerNorm** | `reduction_normalization/layernorm/` | `(x-mean)/sqrt(var+eps)*g+b` | ⬜ |
| 15 | **RMSNorm** | `reduction_normalization/rmsnorm/` | `x/sqrt(mean(x^2)+eps)*g` | ⬜ |
| 16 | **GroupNorm** | `reduction_normalization/group_norm/` | `分组内归一化` | ⬜ |
| 17 | **BatchNorm** | `reduction_normalization/batch_norm/` | `按 batch 统计归一化` | ⬜ |
| 18 | **InstanceNorm** | `reduction_normalization/instance_norm/` | `按实例归一化` | ⬜ |
| 19 | **DeepNorm** | `reduction_normalization/deep_norm/` | `残差+归一化复合` | ⬜ |
| 20 | **Normalize** | `reduction_normalization/normalize/` | `x/max(||x||, eps)` | ⬜ |
| 21 | **L2Norm** | `reduction_normalization/l2_norm/` | `x/||x||_2` | ⬜ |
| 22 | **WelfordUpdate** | `reduction_normalization/welford_update/` | `在线均值/方差更新` | ⬜ |
| 23 | **WelfordFinalize** | `reduction_normalization/welford_finalize/` | `由 running 出最终统计` | ⬜ |

### 扫描/前缀算法（`scan_prefix/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 24 | **CumSum** | `scan_prefix/cumsum/` | `y_i = sum_(j<=i) x_j` | ⬜ |
| 25 | **CumProd** | `scan_prefix/cumprod/` | `y_i = prod_(j<=i) x_j` | ⬜ |
| 26 | **CumMax** | `scan_prefix/cummax/` | `y_i = max_(j<=i) x_j` | ⬜ |
| 27 | **CumMin** | `scan_prefix/cummin/` | `y_i = min_(j<=i) x_j` | ⬜ |
| 28 | **InclusiveScan** | `scan_prefix/inclusive_scan/` | `含当前元素的前缀扫描` | ⬜ |
| 29 | **ExclusiveScan** | `scan_prefix/exclusive_scan/` | `不含当前元素的前缀扫描` | ⬜ |
| 30 | **SegmentedScan** | `scan_prefix/segmented_scan/` | `按段边界分段扫描` | ⬜ |
| 31 | **CumLogSumExp** | `scan_prefix/cumlogsumexp/` | `前缀 logsumexp` | ⬜ |

### 数据重排与索引（`data_rearrange_index/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 32 | **Transpose** | `data_rearrange_index/transpose/` | `y = x^T (permute)` | ⬜ |
| 33 | **LayoutConvert** | `data_rearrange_index/layout_convert/` | `布局转换 (NCHW/NHWC, AoS/SoA)` | ⬜ |
| 34 | **Broadcast** | `data_rearrange_index/broadcast/` | `y = broadcast(x, shape)` | ⬜ |
| 35 | **Pad** | `data_rearrange_index/pad/` | `y = pad(x, paddings, value)` | ⬜ |
| 36 | **Fill** | `data_rearrange_index/fill/` | `y = value` | ⬜ |
| 37 | **Arange** | `data_rearrange_index/arange/` | `y_i = start + i*step` | ⬜ |
| 38 | **Concat** | `data_rearrange_index/concat/` | `y = concat(x1, x2, axis)` | ⬜ |
| 39 | **Split** | `data_rearrange_index/split/` | `y1, y2 = split(x, axis)` | ⬜ |
| 40 | **Slice** | `data_rearrange_index/slice/` | `y = x[begin:end]` | ⬜ |
| 41 | **Gather** | `data_rearrange_index/gather/` | `y[i] = x[idx[i]]` | ⬜ |
| 42 | **Scatter** | `data_rearrange_index/scatter/` | `y[idx[i]] = x[i]` | ⬜ |
| 43 | **Embedding** | `data_rearrange_index/embedding/` | `y[i] = weight[token[i]]` | ⬜ |
| 44 | **OneHot** | `data_rearrange_index/one_hot/` | `y[i][token[i]] = 1` | ⬜ |
| 45 | **Interleave** | `data_rearrange_index/interleave/` | `按奇偶交错合并` | ⬜ |
| 46 | **DeInterleave** | `data_rearrange_index/deinterleave/` | `按奇偶拆分` | ⬜ |

### 排序与搜索（`sort_search/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 47 | **TopK** | `sort_search/topk/` | `top-k values / indices` | ⬜ |
| 48 | **Sort** | `sort_search/sort/` | `y = sorted(x)` | ⬜ |
| 49 | **MrgSort** | `sort_search/mrg_sort/` | `多路归并排序` | ⬜ |
| 50 | **ArgSort** | `sort_search/argsort/` | `y = argsort(x)` | ⬜ |
| 51 | **SearchSorted** | `sort_search/searchsorted/` | `有序表二分查找插入位` | ⬜ |
| 52 | **KthValue** | `sort_search/kth_value/` | `第 k 小/大值` | ⬜ |
| 53 | **Unique** | `sort_search/unique/` | `去重` | ⬜ |
| 54 | **NonZero** | `sort_search/nonzero/` | `非零元素下标` | ⬜ |
| 55 | **RadixSort** | `sort_search/radix_sort/` | `基数排序` | ⬜ |
| 56 | **BeamSearch** | `sort_search/beam_search/` | `束搜索` | ⬜ |

### 位置编码（`position_encoding/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 57 | **RoPE** | `position_encoding/rope/` | `y = x*cos + rotate_half(x)*sin` | ⬜ |
| 58 | **PartialRoPE** | `position_encoding/rope_partial/` | `仅部分维度旋转` | ⬜ |
| 59 | **NTKRope** | `position_encoding/rope_ntk/` | `NTK-aware 缩放 RoPE` | ⬜ |
| 60 | **ALiBi** | `position_encoding/alibi/` | `线性偏置注意力掩码` | ⬜ |
| 61 | **RelativePE** | `position_encoding/relative_position_encoding/` | `相对位置编码` | ⬜ |
| 62 | **SinusoidalPE** | `position_encoding/sinusoidal_encoding/` | `正弦位置编码` | ⬜ |
| 63 | **LearnedPE** | `position_encoding/learned_position_embedding/` | `可学习位置嵌入` | ⬜ |

### 随机（`random/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 64 | **PhiloxRandom** | `random/philox_random/` | `伪随机数生成` | ⬜ |
| 65 | **DropOut** | `random/dropout/` | `y = x*mask/(1-p)` | ⬜ |

## 建议书写顺序

按上表序号顺序推进；同类内先做代表算子打通结构，再做变体练复用。
初版工程 [`vector_add/`](vector_add/README.md) 是全库样板（easy 档）。
