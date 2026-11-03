# C++ API and IR API Differences

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T06:22:30.004Z pushedAt=2026-11-02T03:00:52.258Z nl=2 -->

The PTO virtual instruction set provides two API surfaces: "PTO Virtual Instruction Set (C++)" describes the host-side C++ built-in APIs; "PTO Virtual Instruction Set (IR)" describes the operation form of the same batch of instructions at the IR layer. This document explains the positioning difference and mutual mapping rules between the two, for readers to cross-reference between the two chapters.

---

## Contents

- [1. Positioning Difference](#1-positioning-difference)
- [2. Two Forms of the Same Instruction](#2-two-forms-of-the-same-instruction)
- [3. Naming Mapping Rules](#3-naming-mapping-rules)
- [4. Type System Comparison](#4-type-system-comparison)
- [5. Parameters, Attributes, and Optional Operands](#5-parameters-attributes-and-optional-operands)
- [6. Result Passing and Synchronization](#6-result-passing-and-synchronization)

---

## 1. Positioning Difference

| | PTO Virtual Instruction Set (C++) | PTO Virtual Instruction Set (IR) |
| --- | --- | --- |
| API Form | C++ template functions, unified entry header file `<pto/pto-inst.hpp>` | Textual assembly form of `pto.*` operations |
| Target Readers | Developers writing kernels/operators | Engineers writing or reading compiler frontends and viewing instruction assembly forms |
| Result Passing | Output parameter reference `dst`, returns `RecordEvent` | Destination-passing style (DPS), result written to `outs()` |
| Operand Type | C++ types (`Tile<>`, `GlobalTensor<>`, scalar values) | IR types (`!pto.tile_buf<...>`, `!pto.partition_tensor_view<...>`, `!pto.ptr<T>`) |
| Variant Expression | Overloads, template parameters, runtime enum parameters | Dot-suffix operation names, attributes `{...}`, selected operands |
| Documentation Granularity | One page per instruction | One page per group of operations, segmented by `pto.xxx` within the page |

The two chapters describe the same set of architectural semantics: the behavior definition for each `(i, j)` in the valid region is completely identical, and the difference lies only in the API form and the carrying method. The IR (assembly) form of an instruction is uniformly governed by the IR chapter.

---

## 2. Two Forms of the Same Instruction

Take element-wise multiplication as an example.

C++ API (Auto mode):

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT src0, src1, dst;
    TMUL(dst, src0, src1);
}
```

IR operation:

```mlir
pto.tmul ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>,
                  !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
```

Both sides have the same semantics: for each `(i, j)` in the valid region, `dst[i,j] = src0[i,j] * src1[i,j]`.

---

## 3. Naming Mapping Rules

### 3.1 Basic Rules

C++ uses PascalCase, while IR uses lowercase with the `pto.` prefix:

```
TMUL  <->  pto.tmul
TLOAD <->  pto.tload
```

When looking up the same instruction in the other chapter, simply change the case and add or remove the prefix.

### 3.2 Variant Suffixes

C++ uses underscore suffixes, while IR uses dot suffixes:

| C++ | IR |
| --- | --- |
| `TMATMUL` | `pto.tmatmul` |
| `TMATMUL_ACC` | `pto.tmatmul.acc` |
| `TMATMUL_BIAS` | `pto.tmatmul.bias` |
| `TMATMUL_MX` | `pto.tmatmul.mx` |

### 3.3 One-to-Many: Direction Specialization

C++ uses a polymorphic API with a direction template parameter, while IR bakes the direction into the operation name:

| C++ | IR |
| --- | --- |
| `TPUSH` | `pto.tpush_to_aiv` / `pto.tpush_to_aic` |
| `TPOP` | `pto.tpop_from_aiv` / `pto.tpop_from_aic` |
| `TALLOC` | `pto.talloc_to_aiv` / `pto.talloc_to_aic` |
| `TFREE` | `pto.tfree_from_aiv` / `pto.tfree_from_aic` |

---

## 4. Type System Comparison

### 4.1 Tile Type

| C++ `Tile<>` Template Parameter | IR `!pto.tile_buf<>` Parameter | Meaning |
| --- | --- | --- |
| `TileType::Vec` / `Acc` / `Mat` … | `loc=vec` / `acc` / `mat` / `left` / `right` / `scaling` / `ctrl` | Location space |
| `float` / `half` / `bfloat16_t` / `int32_t` … | `f32` / `f16` / `bf16` / `i32` … | Element type |
| `Rows` / `Cols` | `rows` / `cols` | Physical shape |
| `ValidRow` / `ValidCol` | `v_row` / `v_col` | Valid region |
| `BLayout::RowMajor` / `ColMajor` | `blayout=row_major` / `col_major` | Block layout |
| `SLayout::NoneBox` / `RowMajor` … | `slayout=none_box` / `row_major` … | Storage layout |
| `SFractalSize` | `fractal=512` / `1024` | Fractal granularity |
| `PadVal` | `pad` | Padding value |

On the IR side, the element types also include low-precision forms such as `f8E4M3`, `f8E5M2`, `!pto.hif8`, `!pto.f4E1M2x2`, and `!pto.f4E2M1x2`.

### 4.2 Global Memory and Pointer

| C++ | IR |
| --- | --- |
| `GlobalTensor<>` | `!pto.partition_tensor_view<MxNxdtype>` |
| Address capability encapsulated by a C++ object | `!pto.ptr<T, space>`, memref |

### 4.3 Dynamic Valid Shape

- C++: Pass `-1` as the template parameter and specify it through the constructor parameter at runtime.
- IR: Write `v_row=?` / `v_col=?` in the type, provided by the `valid_row` / `valid_col` of `pto.alloc_tile`
  operand at the creation site:

```mlir
%t1 = pto.alloc_tile valid_row = %c16 valid_col = %c256
    : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=256,
                    v_row=?, v_col=?, blayout=col_major, ...>
```

---

## 5. Parameters, Attributes, and Optional Operands

The compile-time template parameters and runtime enum parameters in C++ are unified as operation attributes on the IR side:

| C++ | IR |
| --- | --- |
| `TMATMUL_MX<AccPhase::Final>(...)` | `pto.tmatmul.mx ... {accPhase = #pto<acc_phase final>}` |
| `TCVT(dst, src, RoundMode, SaturationMode)` | `pto.tcvt ... {rmode = #pto<round_mode RINT>, satmode = ...}` |

The overload families in C++ are commonly expressed as optional operands on the IR side:

| C++ Overload | IR |
| --- | --- |
| `TCVT(dst, src, tmp, mode, ...)` | `pto.tcvt ins(<src>[, <tmp>] ...)`, where `tmp` is optional |

---

## 6. Result Passing and Synchronization

- The C++ API uniformly returns `RecordEvent`, writes the result to the output parameter `dst`, and establishes dependencies on previous operations through the `WaitEvents&&...`
  variadic trailing parameter.
- IR computation and data movement operations are all DPS with no SSA return value; operations that produce handles or perform queries return
  SSA values (such as `pto.alloc_tile`, `pto.load`, `pto.tpop_from_*`, `pto.get_block_idx`).
- On the IR side, synchronization tokens related to buffer lifetime are managed by `pto.get_buf` / `pto.rls_buf`
  (see "Pointer and View Operations").
