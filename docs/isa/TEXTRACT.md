# TEXTRACT

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:57:16.080Z pushedAt=2026-11-02T03:00:52.183Z nl=2 -->

## Instruction Diagram

![TEXTRACT tile operation](../figures/isa/TEXTRACT.svg)

## Introduction

Extracts a smaller sub-tile from a larger source tile.

## Mathematical Semantics

Conceptually, copies a smaller window from the larger `src` tile into `dst`, starting at `(indexRow, indexCol)`. The exact mapping depends on the tile layout.

Assume `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{\mathrm{indexRow}+i,\; \mathrm{indexCol}+j} $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, FpTileData &fp,
                              uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
          typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT_FP(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
          ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow, uint16_t indexCol, WaitEvents &... events);
```

The `STPhase` overload writes the unit flag into the L0C move-out instruction, and is used to pair with `TMATMUL<AccPhase>` to complete the Cube-to-Fixpipe hardware synchronization, thereby eliminating the need for explicit `set_flag`/`wait_flag`. It is exposed only on targets that have the corresponding backend implementation (Atlas A2/A3 training products/Atlas A2/A3 inference products, Ascend 950PR/Ascend 950DT, and the CPU simulator). For the applicable scope and pairing rules, see the implementation checks below.

`TEXTRACT_FP(...)` retains a source-compatibility entry for the historical fp quantization form and directly maps to the `TEXTRACT_IMPL(dst, src, fp, indexRow, indexCol)` path without `mode`. The canonical `TEXTRACT(..., fp, ...)` overload of the same name participates in matching only when `FpTileData::Loc == TileType::Scaling`. The canonical interface also provides an explicit `AccToVecMode` form for the Acc-to-Vec routing supported by the target. `TEXTRACT_FP` also provides an `STPhase` version, aligned with `TMOV_FP` / `TSTORE_FP`, with semantics consistent with the canonical interface.

## Constraints

### General Constraints or Checks

- The `STPhase` overload (unit flag) supports only the `TileType::Acc -> TileType::Mat` (L0C→L1) path;
  A compile-time error is reported when the target is `TileType::Vec`. This overload is exposed on
  Atlas A2/A3 training products/Atlas A2/A3 inference products, Ascend 950PR/Ascend 950DT, and the CPU simulator.
- The value-taking rule on the move-out side differs from that on the accumulation side and the two are not in a simple correspondence:
  The `TMATMUL` that produces this L0C result must already be `AccPhase::Final`, meaning the data is ready;
  `STPhase::Final` is used for the last move-out and releases the unit flag;
  `STPhase::Partial` is used only for the non-final move-outs when the same L0C block is moved out multiple times, and it does not release the unit flag.
  Pairing `STPhase::Partial` with `AccPhase::Partial` causes the fixpipe to wait for a flag that will never arrive and hang,
  a phenomenon that has been reproduced on the Ascend 950PR simulator.
- The Acc→Mat move-out has passed the A3 on-board test, and `tmov_acc2mat` covers `STPhase::Final` and
  The scenario where `STPhase::Partial` is followed by `STPhase::Final`.
  The Ascend 950PR simulation test has passed; in the Ascend 950PR board machine, CANN 9.2.0 environment,
  all 7 directed test cases of `textract` (`case1`, `case21`–`case26`) passed, with max diff of 0,
  covering NZ512/NZ1024, `Final`, `Partial` followed by `Final`, `Unspecified`, and K-split accumulation.
- For the same dtype extraction/layout path, `DstTileData::DType` must equal `SrcTileData::DType`.
  The Acc conversion and quantization paths use the following backend-specific dtype combinations.
- The original A2A3 Mat/Acc extraction paths check the following physical rectangle boundaries (the small-M Mat→Left path described below instead checks the valid window and the full fractal read range separately):
    - `indexRow + DstTileData::Rows <= SrcTileData::Rows`
    - `indexCol + DstTileData::Cols <= SrcTileData::Cols`

### Implementation Check for Atlas A2/A3 Training Products/Atlas A2/A3 Inference Products

For Mat→Left/Right layout extraction:

- Supported element types: `int8_t`, `half`, `bfloat16_t`, `float`.
- The source layout must satisfy one of the following checked extraction layouts of Atlas A2/A3 training products/Atlas A2/A3 inference products:
    - `(SFractal == ColMajor && isRowMajor)`, or
    - `(SFractal == RowMajor && !isRowMajor)`.
- In the GEMV scenario targeting `TileType::Left`, the checked source layout also allows `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`.
- The destination must be `TileType::Left` or `TileType::Right`, with a layout configuration supported by the destination.

### Ascend 950PR/Ascend 950DT Implementation Check

- Supported element types: `int8_t`, `hifloat8_t`, `float8_e5m2_t`, `float8_e4m3_t`, `half`, `bfloat16_t`, `float`, `float4_e2m1x2_t`, `float4_e1m2x2_t`, `float8_e8m0_t`.
- The normal Acc→Mat extraction also supports `int32_t -> int32_t`.
- The source layout must satisfy one of the following checked Ascend 950PR/Ascend 950DT extraction layouts:
    - For `Left` / `Right`: `(SFractal == ColMajor && isRowMajor)` or `(SFractal == RowMajor && !isRowMajor)`
    - For `ScaleLeft`: `(SFractal == RowMajor && isRowMajor)`
    - For `ScaleRight`: `(SFractal == ColMajor && !isRowMajor)`
- In the GEMV scenario targeting `Left`, the checked source layout also allows `(SrcTileData::Rows == 1 && SrcTileData::isRowMajor)`.
- The destination supports `TileType::Mat -> TileType::Left/Right/Scale`, `TileType::Acc -> TileType::Mat` (the ReLU and quantization forms are subject to dtype and layout restrictions; see the Acc→Mat NZ constraints below), `TileType::Acc -> TileType::Vec`, and specific `TileType::Vec -> TileType::Mat` extraction paths.
- The canonical vector quantization `TEXTRACT(..., fp, ...)` form additionally requires the `FpTileData` Scaling operand.
  `TEXTRACT_FP(...)` is still retained as a source-compatible historical alias, and the selected backend implementation continues to check its validity.
- The vector quantization Acc-to-Vec form is exposed only on targets that have a corresponding backend implementation
  (A5, kirin9030, kirinX90, and the CPU simulator), accepting
  `mode = AccToVecMode::{SingleModeVec0, SingleModeVec1, DualModeSplitM, DualModeSplitN}`.
- For `TileType::Acc -> TileType::Vec`, when the destination is a 32-bit type (`float`/`int32_t`) and `DualModeSplitN` is used, the `ValidCol` before splitting must be an integer multiple of `32`.

### A5 Acc→Mat NZ Layout Conversion

The following paths target the NZ layout (`BLayout::ColMajor`, `SLayout::RowMajor`):

| Source to Target Type | Target Layout | Supported Operations |
| --- | --- | --- |
| `float → half/bfloat16_t` | NZ1024, 32 columns per group | Normal conversion, optionally `NoRelu` or `NormalRelu`; scalar/vector quantization is rejected at compile time |
| `int32_t → int32_t` | NZ512, 8 columns per group | Bit-pattern-preserving move under `NoQuant` and `NoRelu`; ReLU is rejected at compile time |

Both the normal `TEXTRACT(dst, src, row, col)` form and the explicit `STPhase` form can enter the above paths.
The normal form uses `STPhase::Unspecified` and requires explicit synchronization between Cube and Fixpipe.
ReLU requires selecting the overload that explicitly specifies the `ReluPreMode` template argument.

The half/bfloat16 path performs Fixpipe NZ→ND with at most 32 columns per group. Within a group, the destination row stride is 32 elements,
and the starting offset of each group is `group * DstTileData::Rows * 32` elements; when the last group has only 16 valid columns, only these 16 columns are written.
The int32 path generates 8-column groups through the channel split overloaded by float instructions, without performing numeric type conversion or floating-point operations.

The caller must satisfy the following storage requirements:

- The source uses the regular 16-column L0C layout, with the physical stride taken from `SrcTileData::Rows`;
  This path does not derive a compact stride from the source valid row count.
- The L0C source window start address must be 64-byte aligned (for a regular L0C tile whose base address is already aligned, `indexCol` is a multiple of 16);
  The L1 destination address must be 32-byte aligned.
- The valid row and column counts must be positive and must not exceed the target physical shape, and the complete source window must fall within the allocated storage range of the source.
- For half/bfloat16 NZ1024, the valid column count must be a multiple of 16. The helper function uses `PTO_ASSERT` to check this condition and
  the source storage boundary, which takes effect only when `_DEBUG` is defined; this is not an unconditional runtime check, nor does it check the source valid shape.
- The int32 NZ512 instruction rounds the valid column count up to a multiple of 8. The rounded window must fall within the source and target storage ranges,
  and extra columns within the last block may be written. This branch adds no runtime boundary check;
  tests that require column padding to remain unchanged use a multiple of 8 as the valid column count.

For each half/bfloat16 NZ1024 instruction, the flag is set according to the outer phase:

| Outer Phase | Non-Last Group | Last Group |
| --- | --- | --- |
| `Final` | `Partial` | `Final` |
| `Partial` | `Partial` | `Partial` |
| `Unspecified` | `Unspecified` | `Unspecified` |

Therefore, when the outer phase is `Partial`, the caller must still perform a subsequent `Final` extraction on the same accumulator.
The int32 path issues only one instruction and passes the outer phase through unchanged.

`textract_acc2mat_layout` covers the native NZ layout comparison, phase pairing, K-split accumulation, offset,
partial valid shape, padding, output protected region, and representative int32 special bit patterns.
case22–23 cover half/bfloat16 ReLU; case24–26 cover the normal overloads of int32/half/bfloat16.

### Small M Mat→Left Extraction (A2A3 and A5)

On A2A3 and A5, the normal `TEXTRACT(dst, src, indexRow, indexCol, events...)` overload automatically selects `load_cbuf_to_ca` (L1→L0A transfer, also called load2d) under the following conditions. No new mode or public overload is added, and the existing event interface remains unchanged. Use the overload without the `ReluPreMode` template argument. Even if `ReluPreMode::NoRelu` is explicitly specified, the Acc→Mat overload is selected on A2A3, which does not support this Mat→Left path.

| Attribute | Requirement |
| --- | --- |
| Element Type | Both source and destination are `half`, or both are `bfloat16_t` |
| Source | `TileType::Mat`, NZ512: `BLayout::ColMajor`, `SLayout::RowMajor`, fractal size 512 |
| Source Physical Shape | Both rows and columns are multiples of 16 |
| A2A3 Target | `TileType::Left`, ZZ512: `BLayout::RowMajor`, `SLayout::RowMajor`, fractal size 512 |
| A5 Target | `TileType::Left`, NZ512: `BLayout::ColMajor`, `SLayout::RowMajor`, fractal size 512 |
| Target Physical Shape | Exactly 16 rows, columns are multiples of 16 |
| Target Compact Mode | `CompactMode::Null` (normal `TileLeft`) or `CompactMode::Normal` (`TileLeftCompact`) |
| Valid Shape | Supports static or dynamic shapes; this load2d path handles 1 to 15 valid rows |

Let `M = dst.GetValidRow()`, `K = dst.GetValidCol()`, `r = indexRow`, `c = indexCol`. Both indices are of unsigned type `uint16_t`. Dynamic valid row/column counts must be set before the call; `TEXTRACT` uses the existing valid shape and does not automatically derive or update the valid shape based on the indices. The small M path requires:

```text
1 <= M <= 15
1 <= K <= DstTileData::Cols
r + M <= src.GetValidRow()
c + K <= src.GetValidCol()
c % 16 == 0
```

`r` does not need to be aligned to 4 or 16. As long as these boundaries and the storage boundaries below are satisfied, the valid window can cross the 16-row fractal boundary inside the source. The source valid row/column counts can also be dynamic values. When the dynamic destination has `M = 16` at runtime, the original normal/Compact implementation is used, which still requires the row and column indices to be aligned to 16. A2A3 also retains the physical rectangle boundary described above; A5 retains the original rule that Compact transfers by the aligned valid width. Static full M, transpose, Right, GEMV, other data types, and architecture paths other than A2A3/A5 retain their original behavior. This extension does not add support for empty valid shapes, valid shapes exceeding physical capacity, or ReLU, quantization, or `STPhase` variants for Mat→Left.

**Physical read/write range.** Each repeat still transfers a complete 512B fractal, and reducing the valid M does not reduce the number of bytes transferred. Define `ceil16(K) = ((K + 15) / 16) * 16` using integer division; the actual transfer width is:

```text
copyCols = DstTileData::Cols         // Normal TileLeft
copyCols = ceil16(K)                // TileLeftCompact
```

The following boundaries must also be satisfied, where `srcEndBytes` is the read end relative to the source tile base address (in bytes, right-open interval):

```text
c + copyCols <= SrcTileData::Cols
copyCols <= DstTileData::Cols
srcEndBytes = (c + copyCols - 16) * SrcTileData::Rows * 2 + r * 32 + 512
srcEndBytes <= SrcTileData::Numel * 2
```

Therefore, the physical column count of a Compact destination can be larger than the source, as long as the aligned valid transfer width satisfies the boundary; a normal destination must accommodate the full physical width transfer. On A2A3, this relaxation applies only to small M: when dynamic `M = 16`, even for a Compact tile, `c + DstTileData::Cols <= SrcTileData::Cols` is still required. A5 dynamic `M = 16` retains the rule that Compact transfers by the aligned valid width and does not add the A2A3 restriction above.

Both backends offset the source pointer by `(c / 16) * SrcTileData::Rows * 16 + r * 16` elements, but the load2d parameter formats differ:

| Backend | Small M Move Parameters |
| --- | --- |
| A2A3 `pto_load_cbuf_to_ca` | `baseIdx = 0`, `repeat = copyCols / 16`, `srcStride = SrcTileData::Rows / 16`, destination gap is 0 |
| A5 `load_cbuf_to_ca` | `mStart = 0`, `kStart = 0`, `mStep = 1`, `kStep = copyCols / 16`, `srcStride = SrcTileData::Rows / 16`, `dstStride = 1`, transpose is 0 |

Adjacent source fractal start points are `srcStride * 512` bytes apart, and destination fractals are arranged contiguously.
The actual amount of data read and written is `copyCols * 32` bytes; when `srcStride > 1`, this amount can be smaller than the source address span.
When repeat/kStep exceeds 255, it is segmented and the source and destination pointers are advanced accordingly;
this does not relax the platform's original L1/L0A capacity limits.

**Source padding and synchronization.** When `c = 0`, reusing a 16×K source whose K is a multiple of 16 and extracting four rows from rows 4/8/12 up to the last K fractal
will exceed the end of the compactly allocated 16×K tile by 128/256/384B, respectively. The physical column count can be declared as `K + 16` with the valid column count kept as `K`,
explicitly owning an extra 512B of storage. For the tail K, taking the source physical column count as `ceil16(K) + 16` and the destination physical column count as `ceil16(K)`
also yields an equally safe arrangement. ND→NZ `TLOAD` can still load the 16×K GlobalTensor only once, and the NZ row stride remains unchanged.
When the column start is non-zero, the source valid column count is at least `c + K`, and `c` must also be counted at the end of the physical read.
Other shapes should be calculated using the read-end formula: the existing capacity may already be sufficient, and adding 16 columns occupies `SrcTileData::Rows * 32` bytes,
which is 512B only when the physical row count is 16.

The entire source allocation (including padding) must remain readable and must not be overwritten or reused before the MTE1 read completes.
Padding does not need to be initialized to zero. Elements outside the destination valid window are undefined and must not be used as valid results.
In manual mode, MTE2→MTE1 synchronization is required after `TLOAD`, MTE1→M synchronization is required before `TMATMUL`,
and M→MTE1 synchronization is required before overwriting a Left tile still in use by the Cube. Automatic mode tracks these dependencies through tile operands.
`STPhase` is used for Acc→Mat move-out and cannot replace these dependencies. Allocation and synchronization must cover the entire physical tile;
the valid shape is used to define the result and cannot replace the physical allocation size. Runtime checks use `PTO_ASSERT`, enabled by `_DEBUG`;
callers must satisfy the constraints in all build modes.

The following snippets use `<pto/pto-inst.hpp>` and
`using namespace pto;` in the A2A3 or A5 device compilation context (`__CCE_AICORE__`); the Left alias selects the layout of the corresponding architecture. It is assumed that the full tile has been allocated, the 16×64 valid window of the source has been loaded, and the caller or automatic mode handles the necessary synchronization:

```cpp
using Mat = Tile<TileType::Mat, half, 16, 80, BLayout::ColMajor, 16, 64, SLayout::RowMajor, 512>;
using Left = TileLeft<half, 16, 64, 4, 64>;
AICORE void ExtractFourRows(Left &left, Mat &mat)
{
    TEXTRACT(left, mat, /*indexRow=*/12, /*indexCol=*/0);
}
```

A Compact target with dynamic M/K can also be used instead; when `m = 4, k = 63` is passed, the physical column count is 128 but only 64 columns are read.
The helper function explicitly sets the valid shape. For this source and row start, the legal parameters are `1 <= m <= 4` and `1 <= k <= 64`;
the caller must allocate the destination first:

```cpp
using CompactLeft = TileLeftCompact<half, 16, 128, DYNAMIC, DYNAMIC>;
AICORE void ExtractCompactRows(CompactLeft &left, Mat &mat, uint32_t m, uint32_t k)
{
    left.SetValidShape(m, k);
    TEXTRACT(left, mat, /*indexRow=*/12, /*indexCol=*/0);
}
```

See [A2A3 Implementation](../../include/pto/npu/a2a3/TExtract.hpp) and
[A5 Implementation](../../include/pto/npu/a5/TExtract.hpp),
[A2A3 Numerical ST](../../tests/npu/a2a3/src/st/testcase/textract_small_m/textract_small_m_kernel.cpp) and
[A5 Numerical ST](../../tests/npu/a5/src/st/testcase/textract_small_m/textract_small_m_kernel.cpp), as well as
[A2A3 Instruction and Boundary Tests](../../tests/costmodel/st/testcase/textract_small_m/main.cpp) and
[A5 Instruction and Boundary Tests](../../tests/cpu/st/testcase/textract_small_m_a5/main.cpp).
The A5 host test simulates the underlying move instruction and executes the real A5 backend dispatch entry and helper; the NPU ST performs numerical verification of the complete instruction sequence.

### Vec → Vec Extraction Path

In addition to the `Mat/Acc -> ...` paths above, `TEXTRACT` also supports the `TileType::Vec -> TileType::Vec` extraction path (ND and NZ layouts). A2A3 uses `CheckTExtractVecToVecCommon`, and A5 checks separately in `TEXTRACT_IMPL`:

- `DstTileData::DType` must equal `SrcTileData::DType`.
- A2A3 element types: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`.
- A5 element types: `int8_t`, `int32_t`, `half`, `bfloat16_t`, `float`, `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`, `float8_e8m0_t`, `float4_e2m1x2_t`, `float4_e1m2x2_t`. A5 does not support `uint8_t`, `int16_t`, `uint16_t`, `uint32_t`, or 64-bit integers for this path.
- For the A5 ND fp4 path, elements are counted as packed elements (each 1 byte contains two fp4 values); the row stride, static valid column byte count, and column offset byte count must be 32-byte aligned.
- ND path: the source/destination row stride must be 32-byte aligned; the `Dst` rows/columns must not exceed `Src`.
- The A5 ND Vec→Vec path first checks `indexRow + dst.GetValidRow() <= SrcTileData::Rows` and `indexCol + dst.GetValidCol() <= SrcTileData::Cols`; after the checks pass, if the destination valid row count or column count is 0, it returns directly without reading the source or writing the destination. This rule applies to both aligned and unaligned column offsets; it does not extend the type support of this path, and int64 is still not supported.

## Examples

### Automatic

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT src;
  DstT dst;
  TEXTRACT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::ColMajor>;
  using DstT = TileLeft<float, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TEXTRACT(dst, src, /*indexRow=*/0, /*indexCol=*/0);
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
  // Move out after the data is ready: no explicit set_flag/wait_flag is required between the two instructions.
  TMATMUL<AccPhase::Final>(c, a, b);
  TEXTRACT<STPhase::Final>(l1, c, /*indexRow=*/0, /*indexCol=*/0);

  // When the same L0C block is moved out in multiple parts, use Partial for non-final parts without releasing the unit flag, and use Final for the last part to release it.
  // TEXTRACT<STPhase::Partial>(l1, c, /*indexRow=*/0, /*indexCol=*/0);
  // TEXTRACT<STPhase::Final>(l1, c, /*indexRow=*/0, /*indexCol=*/0);
}
```
