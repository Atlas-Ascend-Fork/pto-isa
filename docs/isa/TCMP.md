# TCMP

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:36:46.055Z pushedAt=2026-11-02T03:00:52.160Z nl=2 -->

## Instruction Diagram

![TCMP tile operation](../figures/isa/TCMP.svg)

## Introduction

Compares two tiles and writes a packed predicate mask.

## Mathematical Semantics

Conceptually, for each element `(i, j)` in the valid region, a predicate is defined:

$$ p_{i,j} = \left(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{src1}_{i,j}\right) $$

The predicate mask is stored in `dst` using an implementation-defined packed layout.

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/type.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCMP(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, CmpMode cmpMode, WaitEvents &... events);
```

## Constraints

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - The input type must be one of the following: `int32_t`, `half`, `float`.
    - The output type must be `uint8_t`.
    - The `src0/src1/dst` tile position must be `TileType::Vec`.
    - Static valid bounds: `TileDataSrc::ValidRow <= TileDataSrc::Rows` and `TileDataSrc::ValidCol <= TileDataSrc::Cols`.
    - Runtime: the valid row and column counts of `src0` and `src1` are respectively equal, and `src0.GetValidRow() == dst.GetValidRow()`.
    - The destination valid column count represents the packed capacity and is not required to equal the source valid column count.
    - For `int32_t` input, `EQ` and `NE` are supported; `NE` negates the equality comparison result, and other modes use the `EQ` path.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - The input type must be one of the following: `uint32_t`, `int32_t`, `int64_t`, `uint64_t`, `uint16_t`, `int16_t`, `uint8_t`, `int8_t`, `float`, `half`, `bfloat16_t`.
    - The output is packed predicate bytes, which can be used as a RowMajor `uint8_t` mask tile.
    - Implemented (see `include/pto/npu/a5/TCmp.hpp`).
    - The iteration domain is `src0.GetValidRow()` / `src0.GetValidCol()`; `src1` must provide the corresponding valid elements. The destination valid column count represents the packed capacity and does not determine the number of comparisons.
- **Mask encoding**:
    - 64-bit input (Ascend 950PR/Ascend 950DT): the comparison result of column `j` is stored in bit `j % 8` of byte `j / 8` in that row, least significant bit first.
    - For a `uint8_t` mask, the valid shape can be set to `[R, ceil(C / 8)]`, with the physical `Cols` aligned to 32 bytes, where `[R,C]` is the source valid shape. Row addresses are calculated based on the destination physical stride; the padding values after the last valid bit are unspecified.
    - The mask tile is interpreted as packed predicate bits in the destination-defined layout.

## Examples

### Automatic

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src0, src1;
  MaskT mask(16, 2);
  TCMP(mask, src0, src1, CmpMode::GT);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src0, src1;
  MaskT mask(16, 16);
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(mask, 0x3000);
  TCMP(mask, src0, src1, CmpMode::GT);
}
```
