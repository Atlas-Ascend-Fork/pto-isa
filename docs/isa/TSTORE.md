# TSTORE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T06:11:16.169Z pushedAt=2026-11-02T03:00:52.239Z nl=2 -->

## Instruction Diagram

![TSTORE tile operation](../figures/isa/TSTORE.svg)

## Introduction

Stores data in a tile to a GlobalTensor (GM), optionally using atomic write or quantization parameters.

## Mathematical Semantics

The symbolic representation depends on the shape/stride of the `GlobalTensor` and the layout of the `Tile`. Conceptually (two-dimensional view, with base offset):

$$ \mathrm{dst}_{r_0 + i,\; c_0 + j} = \mathrm{src}_{i,j} $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/constants.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events);

template <typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
          typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, uint64_t preQuantScalar, WaitEvents&... events);

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events);

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
          ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE_FP(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events);
```

`TSTORE_FP(...)` is retained as a source-compatibility entry for the legacy fp quantization form and maps directly to `TSTORE_IMPL(dst, src, fp)`. The canonical same-name `TSTORE(..., fp, ...)` overload participates in matching only when `FpTileData::Loc == TileType::Scaling`; the backend implementation may still perform additional validity checks.

The value rule for `STPhase` on the store-out side is asymmetric with the accumulate side: the `TMATMUL` that produces this L0C result must already be `AccPhase::Final`, meaning the data is ready; `STPhase::Final` is used for the last store-out and releases the unit flag; `STPhase::Partial` is used only for the non-final store-outs when the same L0C block is stored out in multiple passes, and it does not release the unit flag. Pairing `STPhase::Partial` with `AccPhase::Partial` causes the fixpipe to wait for a flag that will never arrive and hang.

This rule has been confirmed by simulation tests on Ascend 950PR across the three store-out paths L0C→L1, L0C→UB, and L0C→GM, where `TSTORE` uses the L0C→GM `copy_matrix_cc_to_gm` path (the unit flag is written in the Xt register rather than as a positional parameter), covered by `case_vector_quant_uf_multi_drain` of `tstore_acc2gm`.

On the Ascend 950PR board with CANN 9.2.0, all three targeted test cases of `tstore_acc2gm` (`case38`, `case_vector_quant_uf_final`, `case_vector_quant_uf_multi_drain`) pass with max diff of 0, covering the per-channel quantization baseline, `Final`, and `Partial` followed by `Final`. Together with TEXTRACT, TINSERT, and TMOV, all 24 targeted store-out cases in this round pass on the board; this result covers only the selected cases and does not represent the complete A5 ST test set.

The vector quantization `STPhase` form is exposed only on targets that have a corresponding backend implementation (A2A3, A5, kirin9030, kirin9050, and the CPU simulator).

## L2 Cache Hint

Optional first template parameter `TStoreL2Hint l2Control` (default `NormalFirstVictim`):

```cpp
TSTORE(dst, src);
TSTORE<TStoreL2Hint::NotAllocClean>(dst, src);
TSTORE<TStoreL2Hint::NotAllocClean, AtomicType::AtomicAdd>(dst, src);
```

When combined with `AtomicType` / `STPhase` / `ReluPreMode`, `TStoreL2Hint` is placed first. Do not insert a default hint before `TileData` in the legacy overload set.

Supported `TStoreL2Hint`:

| Enum | Value | A2/A3 | A5 |
| --- | --- | --- | --- |
| NormalFirstVictim | 0 | No effect | Supported |
| NormalLastVictim | 1 | No effect | Supported |
| NormalPersistent | 2 | No effect | Supported |
| NotAllocClean | 4 | No effect | Supported |

On A2/A3, the L2 hint has **no effect** (all values are no-ops; there is no store L2 control). On A5, the values listed in the table are passed through to the DMA. CPU / costmodel ignore it.

## Constraints

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - The source tile position must be one of the following: `TileType::Vec`, `TileType::Mat`, `TileType::Acc`.
    - Runtime: all `dst.GetShape(dim)` values and `src.GetValidRow()/GetValidCol()` must be `> 0`.
    - For source tile position `TileType::Vec` / `TileType::Mat`:
        - `TileData::DType` must be one of the following: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
        - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
        - The layout must match ND/DN/NZ (or special case: `TileData::Rows == 1` or `TileData::Cols == 1`).
      In this special case, the traversal is taken from the Tile layout rather than the `GlobalTensor` layout: the vector is written in a single contiguous transfer, without using the stride of the other dimension.
      Therefore, a ColMajor `[N, 1]` Tile written to memory through an ND `GlobalTensor` occupies `N` contiguous elements, rather than one element per row stride.
        - For `int64_t/uint64_t`, only ND->ND or DN->DN is supported.
    - For the source tile position being `TileType::Acc` (including call forms with quantization parameters and atomic write variants):
        - Supported layout conversions: NZ2ND, NZ2NZ, NZ2NC1HWC0, NZ2NDC1HWC0. NZ2DN is not supported.
        - The destination layout must be ND, NZ, NC1HWC0, or NDC1HWC0.
        - The source data type must be `int32_t` or `float`.
        - When quantization is not used, the destination data type must be `int32_t/float/half/bfloat16_t`.
        - The data type support for ACC-to-GM depends on the call form:

          | Call form | Source Data Type | Supported Target Data Types |
          | --- | --- | --- |
          | `TSTORE(dst, acc)` | `float` | `float`, `half`, `bfloat16_t` |
          | `TSTORE(dst, acc)` | `int32_t` | `int32_t` |
          | `TSTORE(dst, acc, preQuantScalar)` / `TSTORE(dst, acc, fp)` / `TSTORE_FP(dst, acc, fp)` | `float` | `int8_t`, `uint8_t` |
          | `TSTORE(dst, acc, preQuantScalar)` / `TSTORE(dst, acc, fp)` / `TSTORE_FP(dst, acc, fp)` | `int32_t` | `int8_t`, `uint8_t`, `half` |

          Other cross-type combinations not listed are not within the supported scope.

        - Static shape constraints: `1 <= TileData::Cols <= 4095`; if ND, `1 <= TileData::Rows <= 8192`; if NZ, NC1HWC0, or NDC1HWC0, `1 <= TileData::Rows <= 65535` and `TileData::Cols % 16 == 0`.
        - Runtime: `1 <= src.GetValidCol() <= 4095`.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - The source tile position must be `TileType::Vec` or `TileType::Acc` (`Mat` storage is not supported on this target).
    - For source tile position `TileType::Vec`:
        - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
        - `TileData::DType` must be one of the following: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`, `float8_e4m3_t`, `float8_e5m2_t`, `hifloat8_t`, `float8_e8m0_t`, `float4_e1m2x2_t`, `float4_e2m1x2_t`.
        - The layout must match ND/DN/NZ (or special case: `TileData::Rows == 1` or `TileData::Cols == 1`).
      In this special case, the traversal method is taken from the Tile layout rather than the `GlobalTensor` layout: the vector is written in one contiguous transfer without using the stride of the other dimension.
      Therefore, a ColMajor `[N, 1]` Tile written to GM through an ND `GlobalTensor` occupies `N` consecutive elements instead of one element per row stride.
        - Additional alignment constraints are enforced (for example, for ND, the row-major width in bytes must be a multiple of 32; for DN, the column-major height in bytes must be a multiple of 32, with exceptions in special cases).
    - For the source tile position being `TileType::Acc` (including call forms with quantization parameters and atomic write variants):
        - Supported layout conversions: NZ2ND, NZ2NZ, NZ2NHWC, NZ2NCHW, NZ2NCDHW. NZ2DN is not supported.
        - The target layout must be ND, NZ, NHWC, NCHW, or NCDHW; the source data type must be `int32_t` or `float`.
        - When quantization is not used, the destination data type must be `int32_t/float/half/bfloat16_t`.
        - The data type support for ACC-to-GM depends on the call form:

          | Call Form | Source Data Type | Supported Target Data Types |
          | --- | --- | --- |
          | `TSTORE(dst, acc)` | `float` | `float`, `half`, `bfloat16_t` |
          | `TSTORE(dst, acc)` | `int32_t` | `int32_t` |
          | `TSTORE(dst, acc, preQuantScalar)` / `TSTORE(dst, acc, fp)` / `TSTORE_FP(dst, acc, fp)` | `float` | `int8_t`, `uint8_t`, `half`, `bfloat16_t`, `hifloat8_t`, `float8_e4m3_t`, `float` |
          | `TSTORE(dst, acc, preQuantScalar)` / `TSTORE(dst, acc, fp)` / `TSTORE_FP(dst, acc, fp)` | `int32_t` | `int8_t`, `uint8_t`, `half`, `bfloat16_t` |

          Other cross-type combinations not listed are not within the supported scope.

        - The static shape constraints are the same as the row/column constraints of Atlas A2/A3 training products/Atlas A2/A3 inference products; `AtomicAdd` additionally restricts the destination data type to supported atomic types.
- **Valid region**:
    - The implementation uses `src.GetValidRow()`/`src.GetValidCol()` as the transfer size.

## Examples

### Automatic

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_auto(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TSTORE(gout, t);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_manual(__gm__ T* out) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gout(out);
  TileT t;
  TASSIGN(t, 0x1000);
  TSTORE<TileT, GTensor, AtomicType::AtomicAdd>(gout, t);
}
```
