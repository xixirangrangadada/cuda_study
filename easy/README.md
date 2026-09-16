# easy — T1 元素基础（28 题）

> 入门档。全部为 T1 元素映射算子。目标是打通「自包含工程 + CPU golden + sanitizer + bench」全链路，并把合并访存、grid-stride、float4 向量化、尾块处理练成肌肉记忆。

**算子总数：28**

## 分类总览

| 类别 | 目录 | 算子数 | 代表算子 |
|------|------|:---:|------|
| 逐元素算术 | `elementwise_arithmetic/` | 12 | Add / Log / Max |
| 激活函数 | `activation/` | 8 | Relu / Gelu / Erf |
| 类型转换/比较选择 | `cast_compare/` | 4 | Cast / Select / Where |
| 复合/融合原语 | `composite_fused/` | 4 | Axpy / MulAddDst / AddRelu |

## 各类算子清单

### 逐元素算术（`elementwise_arithmetic/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 1 | **Add** | `elementwise_arithmetic/add/` | `x1 + x2` | ⬜ |
| 2 | **Mul** | `elementwise_arithmetic/mul/` | `x1 * x2` | ⬜ |
| 3 | **Sub** | `elementwise_arithmetic/sub/` | `x1 - x2` | ⬜ |
| 4 | **Div** | `elementwise_arithmetic/div/` | `x1 / x2` | ⬜ |
| 5 | **Abs** | `elementwise_arithmetic/abs/` | `|x|` | ⬜ |
| 6 | **Exp** | `elementwise_arithmetic/exp/` | `e^x` | ⬜ |
| 7 | **Log** | `elementwise_arithmetic/log/` | `ln(x)` | ⬜ |
| 8 | **Sqrt** | `elementwise_arithmetic/sqrt/` | `sqrt(x)` | ⬜ |
| 9 | **Pow** | `elementwise_arithmetic/pow/` | `x^p` | ⬜ |
| 10 | **Clamp** | `elementwise_arithmetic/clamp/` | `min(max(x, lo), hi)` | ⬜ |
| 11 | **Min** | `elementwise_arithmetic/min/` | `min(x1, x2)` | ⬜ |
| 12 | **Max** | `elementwise_arithmetic/max/` | `max(x1, x2)` | ⬜ |

### 激活函数（`activation/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 13 | **Relu** | `activation/relu/` | `max(0, x)` | ⬜ |
| 14 | **LeakyRelu** | `activation/leaky_relu/` | `x>0 ? x : alpha*x` | ⬜ |
| 15 | **Sigmoid** | `activation/sigmoid/` | `1/(1+exp(-x))` | ⬜ |
| 16 | **Tanh** | `activation/tanh/` | `(exp(x)-exp(-x))/(exp(x)+exp(-x))` | ⬜ |
| 17 | **Gelu** | `activation/gelu/` | `0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))` | ⬜ |
| 18 | **Silu** | `activation/silu/` | `x * sigmoid(x)` | ⬜ |
| 19 | **HardSigmoid** | `activation/hard_sigmoid/` | `clamp(x/6 + 0.5, 0, 1)` | ⬜ |
| 20 | **Erf** | `activation/erf/` | `erf(x)` | ⬜ |

### 类型转换/比较选择（`cast_compare/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 21 | **Cast** | `cast_compare/cast/` | `(DTYPE_OUT) x` | ⬜ |
| 22 | **Compare** | `cast_compare/compare/` | `x1 > x2 ? 1 : 0` | ⬜ |
| 23 | **Select** | `cast_compare/select/` | `mask ? x1 : x2` | ⬜ |
| 24 | **Where** | `cast_compare/where/` | `cond ? x1 : x2` | ⬜ |

### 复合/融合原语（`composite_fused/`）

| 序 | 算子 | 目录 | 数学定义 | 状态 |
|---|------|------|----------|------|
| 25 | **Axpy** | `composite_fused/axpy/` | `y = a*x + y` | ⬜ |
| 26 | **FusedMulAdd** | `composite_fused/fused_mul_add/` | `x1*x2 + x3` | ⬜ |
| 27 | **MulAddDst** | `composite_fused/mul_add_dst/` | `dst = src1*src2 + dst` | ⬜ |
| 28 | **AddRelu** | `composite_fused/add_relu/` | `max(0, x1 + x2)` | ⬜ |

## 建议书写顺序

按上表序号顺序推进；同类内先做代表算子打通结构，再做变体练复用。
初版工程 [`vector_add/`](vector_add/README.md) 是全库样板（easy 档）。
