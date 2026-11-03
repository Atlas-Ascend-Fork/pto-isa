# TINSERT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T05:07:56.643Z pushedAt=2026-11-02T03:00:52.191Z nl=2 -->

## Instruction Diagram

![TINSERT tile operation](../figures/isa/TINSERT.svg)

## Introduction

Inserts a source sub-tile into a destination tile at the `(indexRow, indexCol)` offset. Conceptually, it is the inverse operation of `TEXTRACT`.

`TINSERT` is used for:

- Acc → Mat insertion (with optional relu, scalar quantization, or vector quantization)
- Acc → Vec insertion (with optional `AccToVecMode`, relu, scalar quantization, or vector quantization) *(Ascend 950PR/Ascend 950DT)*
- Vec → Mat insertion (ND and NZ layouts) *(Ascend 950PR/Ascend 950DT)*
- Vec → Vec insertion (ND and NZ layouts) *(Ascend 950PR/Ascend 950DT)*
- NZ split insertion (`SPLIT2`, `SPLIT4`) *(Ascend 950PR/Ascend 950DT)*

## Mathematical Semantics

Assume `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{\mathrm{indexRow}+i,\;\mathrm{indexCol}+j} = \mathrm{src}_{i,j}
$$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint64_t preQuantScalar,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint64_t preQuantScalar,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT_FP(DstTileData &dst, SrcTileData &src,
                                FpTileData &fp,
                                uint16_t indexRow, uint16_t indexCol,
                                WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             FpTileData &fp,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData,
          AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             FpTileData &fp,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
template <TInsertMode mode, typename DstTileData, typename SrcTileData,
          typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow = 0, uint16_t indexCol = 0,
                             WaitEvents &... events);
#endif

template <STPhase Phase, typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src,
                             uint64_t preQuantScalar,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(DstTileData &dst, SrcTileData &src, FpTileData &fp,
                             uint16_t indexRow, uint16_t indexCol,
                             WaitEvents &... events);
```

The `STPhase` overload writes the unit flag into the L0C move-out instruction, which is used to pair with `TMATMUL<AccPhase>` to complete the Cube-to-Fixpipe hardware synchronization, thereby eliminating the explicit `set_flag`/`wait_flag`. It is exposed only on targets that have a corresponding backend implementation (Atlas A2/A3 training products/Atlas A2/A3 inference products, Ascend 950PR/Ascend 950DT, and the CPU simulator). For the applicable scope and pairing rules, see the implementation checks below.

`TINSERT_FP(...)` retains a source-compatibility entry for the historical fp quantization form and directly maps to the `TINSERT_IMPL(dst, src, fp, indexRow, indexCol)` path without `mode`. The canonical same-name `TINSERT(..., fp, ...)` overload participates in matching only when `FpTileData::Loc == TileType::Scaling`. `TINSERT_FP` also provides an `STPhase` version, aligned with `TMOV_FP` / `TSTORE_FP`, and its semantics are consistent with the canonical interface.

## Constraints

### General Constraints/Checks

- The `STPhase` overload (unit flag) supports only the `TileType::Acc -> TileType::Mat` (L0C→L1) path;
  A compile-time error is reported when the destination is `TileType::Vec`. This overload is exposed on Atlas A2/A3 training products/Atlas A2/A3 inference products, Ascend 950PR/Ascend 950DT, and the CPU simulator.
- The value selection rule on the move-out side differs from that on the accumulation side and is not a simple correspondence:
  The `TMATMUL` that produces this L0C result must already be `AccPhase::Final`, meaning the data is ready; `STPhase::Final` is used for the last move-out and releases the unit flag; `STPhase::Partial` is used only for the non-final instructions when the same L0C block is moved out in multiple parts, and it does not release the unit flag. Pairing `STPhase::Partial` with `AccPhase::Partial` causes the fixpipe to wait for a flag that will never arrive and hang, a phenomenon that has been reproduced on the Ascend 950PR simulator.
- The Acc→Mat move-out has passed on-board testing on A3, and `tmov_acc2mat` covers `STPhase::Final` and
  `STPhase::Partial` followed by `STPhase::Final`.
  Ascend 950PR simulation tests have passed; on the Ascend 950PR board with CANN 9.2.0,
  all 4 directional use cases of `tinsert` (`TInsertTest.case_acc2mat_*`) passed, with max diff of 0,
  covering baseline, `Final`, and `Partial` followed by `Final`.
- `TINSERT` has the following overload families:
    - Normal insertion: `TINSERT(dst, src, indexRow, indexCol)`
    - relu form: `TINSERT<..., reluMode>(dst, src, indexRow, indexCol)`
    - Accumulator-to-vector form: `TINSERT<..., mode, reluMode>(dst, src, indexRow, indexCol)`
    - Scalar quantization form: `TINSERT<..., reluMode>(dst, src, preQuantScalar, indexRow, indexCol)` and `TINSERT<..., mode, reluMode>(dst, src, preQuantScalar, indexRow, indexCol)`
    - Vector quantization form: `TINSERT<..., FpTileData, reluMode>(dst, src, fp, indexRow, indexCol)`,
      `TINSERT_FP<..., reluMode>(dst, src, fp, indexRow, indexCol)`, and the target-supported
      `TINSERT<..., FpTileData, mode, reluMode>(dst, src, fp, indexRow, indexCol)` Acc-to-Vec route
    - NZ split form *(Ascend 950PR/Ascend 950DT/Kirin9030/KirinX90 only)*: `TINSERT<TInsertMode::SPLIT2>(dst, src, indexRow, indexCol)` or `TINSERT<TInsertMode::SPLIT4>(...)`
- `reluMode` takes the value of `ReluPreMode::{NoRelu, NormalRelu}`.
- `mode` takes the value of `AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`.
  The vector quantization `fp + mode` overload is exposed only on targets that have a corresponding backend implementation
  (A5, kirin9030, kirinX90, and the CPU simulator).
- Runtime boundary: `indexRow + src.ValidRow <= dst.Rows` and `indexCol + src.ValidCol <= dst.Cols`.

### Implementation Check for Atlas A2/A3 Training Products/Atlas A2/A3 Inference Products

- Supported tile type pairs: only `TileType::Acc → TileType::Mat`.
- The source layout must be `(BFractal: ColMajor, SFractal: RowMajor)`.
- The target layout must be `(BFractal: ColMajor, SFractal: RowMajor)`, and `SFractalSize == 512`.
- `Dst.Cols * sizeof(DstDType)` must be a non-zero multiple of `32` bytes.
- Supported dtype pairs for **normal/relu** (non-quantized):
    - `float` Acc → `half`, `bfloat16_t`
- Supported dtype pairs for **scalar quantization**:
    - `float` Acc → `int8_t`
    - `int32_t` Acc → `int8_t`, `uint8_t`, `half`, `int16_t`
- **Vector quantization** (`TINSERT` with `FpTileData`; historical `TINSERT_FP` alias) supported dtype pairs:
    - `float` Acc → `int8_t`
    - `int32_t` Acc → `int8_t`, `uint8_t`, `half`, `int16_t`
- The normal vector quantization overload requires the `FpTileData` Scaling operand (`TileType::Scaling`);
  `TINSERT_FP(...)` maintains history source compatibility, and the selected backend implementation continues to check its validity.

### Ascend 950PR/Ascend 950DT Implementation Check

In addition to the Acc → Mat path, Ascend 950PR/Ascend 950DT also supports the Acc → Vec, Vec → Vec, Vec → Mat, and NZ split paths.

- **Acc → Mat** (`TileType::Acc → TileType::Mat`):
    - The source Acc type must be `float` or `int32_t`; the source layout must be `(BFractal: ColMajor, SFractal: RowMajor)`.
    - The target layout must be `(!isRowMajor, SFractal: RowMajor)` (NZ format).
    - **Non-quantized** (normal/relu) target types:
        - `float` Acc → `half`, `bfloat16_t`, `float`
        - `int32_t` Acc → `int32_t`
    - **Scalar quantization** target types:
        - `float` Acc → `int8_t`, `uint8_t`, `hifloat8_t`, `half`, `bfloat16_t`, `float8_e4m3_t`
        - `int32_t` Acc → `int8_t`, `uint8_t`, `half`, `bfloat16_t`
    - **Vector quantization** (`TINSERT` with `FpTileData`; historical `TINSERT_FP` alias) target type: same as scalar quantization.

- **Acc → Vec** (`TileType::Acc → TileType::Vec`):
    - The source Acc type must be `float` or `int32_t`; the source layout must be `(BFractal: ColMajor, SFractal: RowMajor)`.
    - **Non-quantized** (normal/relu) target types:
        - `float` Acc → `half`, `bfloat16_t`, `float`
        - `int32_t` Acc → `int32_t`
    - **Scalar quantization** target types:
        - `float` Acc → `int8_t`, `uint8_t`, `hifloat8_t`, `half`, `bfloat16_t`, `float8_e4m3_t`
        - `int32_t` Acc → `int8_t`, `uint8_t`, `half`, `bfloat16_t`
    - **Vector quantization** (`TINSERT` with `FpTileData`; historical `TINSERT_FP` alias) target types: same as scalar quantization.
    - The target layout must be one of the following: NZ-to-NZ (`!isRowMajor, SFractal: RowMajor`), NZ-to-ND (`isRowMajor, SFractal: NoneBox`), or NZ-to-DN (`!isRowMajor, SFractal: NoneBox`).
    - `AccToVecMode` selects `SingleModeVec0`, `SingleModeVec1`, `DualModeSplitM`, or `DualModeSplitN`.
    - Dual-target modes (`DualModeSplitM`, `DualModeSplitN`) require `QuantMode_t::NoQuant` and do not support the NZ-to-DN path.
    - For 32-bit target types (`float`/`int32_t`), when using `DualModeSplitN`, the `ValidCol` before splitting must be an integer multiple of `32`.
    - The target stride must be non-zero, and `dstStride * sizeof(dstType)` must be a multiple of `32` bytes.

- **Vec → Vec** (`TileType::Vec → TileType::Vec`):
    - `DstTileData::DType` must equal `SrcTileData::DType`.
    - Supported element types: `half`, `bfloat16_t`, `float`, `int32_t`, `int8_t`, `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`, `float8_e8m0_t`, `float4_e2m1x2_t`, `float4_e1m2x2_t`.
    - The source and target layouts must match (both ND or both NZ).
    - ND path: The source valid region must be within the target boundary. The dispatch selects `copy_ubuf_to_ubuf` (aligned), `vlds`/`vsts` (stride-aligned, unaligned validCol), `vlds`/`vstus` (unaligned stride or indexCol), or scalar copy (1×1 element).
    - NZ path: The number of source columns must not exceed the number of target columns. Use `ComputeNZBlockParams` for fractal block `copy_ubuf_to_ubuf`.

- **Vec → Mat** (`TileType::Vec → TileType::Mat`, UB → L1):
    - `DstTileData::DType` must equal `SrcTileData::DType`.
    - Supported element types: `half`, `bfloat16_t`, `float`, `int32_t`, `int8_t`, `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`, `float8_e8m0_t`, `float4_e2m1x2_t`, `float4_e1m2x2_t`.
    - ND path: The source must be `isRowMajor`; use `copy_ubuf_to_cbuf`. The number of bytes per row must be aligned with `BLOCK_BYTE_SIZE` (32 bytes).
    - NZ path: The source must be `(!isRowMajor, SFractal: RowMajor)`; use `ComputeNZBlockParams` for fractal block `copy_ubuf_to_cbuf`.

- **NZ Split** (`TInsertMode::SPLIT2` / `TInsertMode::SPLIT4`, only Ascend 950PR/Ascend 950DT/Kirin9030/KirinX90):
    - The target must be `TileType::Mat`; the source must be `TileType::Vec`.
    - `DstTileData::DType` must equal `SrcTileData::DType`.
    - The source must be in NZ format: `(!isRowMajor, SFractal: RowMajor)`.
    - Supported element types: `half`, `bfloat16_t`, `float`, `int32_t`, `int8_t`, `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`, `float8_e8m0_t`, `float4_e2m1x2_t`, `float4_e1m2x2_t`.
    - `validRow` is aligned to `FRACTAL_NZ_ROW` (16) for burst calculation.
    - Split the total burst of `copy_ubuf_to_cbuf` into 2 or 4 sub-transfers, each handling `totalBurstNum / SplitCount` column blocks.

## Examples

### Automatic

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// Vec -> Mat insert (NZ layout).
void example_auto() {
  using SrcT = Tile<TileType::Vec, half, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor>;
  using DstT = Tile<TileType::Mat, half, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
  SrcT src;
  DstT dst(16, 32);
  TINSERT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// Vec -> Mat insert (NZ layout, manual buffer allocation).
void example_manual() {
  using SrcT = Tile<TileType::Vec, half, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor>;
  using DstT = Tile<TileType::Mat, half, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
  SrcT src;
  DstT dst(16, 32);
  TASSIGN(src, 0x0);
  TASSIGN(dst, 0x0);
  TINSERT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

L0C→L1 move-out with unit flag (Atlas A2/A3 training products/Atlas A2/A3 inference products, Ascend 950PR/Ascend 950DT):

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_unit_flag() {
  TileLeft<half, 32, 32> a;
  TileRight<half, 32, 32> b;
  TileAcc<float, 32, 32> c;
  Tile<TileType::Mat, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor> l1;
  TASSIGN(a, 0x0);
  TASSIGN(b, 0x0);
  TASSIGN(c, 0x0);
  TASSIGN(l1, 0x2000);
  // Move out after data is ready: no explicit set_flag/wait_flag is required between the two instructions.
  TMATMUL<AccPhase::Final>(c, a, b);
  TINSERT<STPhase::Final>(l1, c, /*indexRow=*/0, /*indexCol=*/0);

  // When the same L0C block is moved out in multiple passes, use Partial for non-final passes without releasing the unit flag, and use Final for the final pass to release it.
  // TINSERT<STPhase::Partial>(l1, c, /*indexRow=*/0, /*indexCol=*/0);
  // TINSERT<STPhase::Final>(l1, c, /*indexRow=*/0, /*indexCol=*/0);
}
```
