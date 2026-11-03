# TANDS

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:33:32.840Z pushedAt=2026-11-02T03:00:52.157Z nl=2 -->

## Instruction Diagram

![TANDS tile operation](../figures/isa/TANDS.svg)

## Introduction

Element-wise bitwise AND between a tile and a scalar.

## Mathematical Semantics

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \;\&\; \mathrm{scalar} $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TANDS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar, WaitEvents &... events);
```

## Constraints

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - Applicable to integer element types.
    - `dst` and `src` must use the same element type.
    - `dst` and `src` must be vector tiles.
    - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`.
    - In manual mode, setting the source tile and destination tile to the same memory is not supported.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - Applicable to the integer element types supported by `TANDS`.
    - `dst` and `src` must use the same element type.
    - `dst` and `src` must be vector tiles.
    - Runtime: `src0.GetValidRow() == dst.GetValidRow()` and `src0.GetValidCol() == dst.GetValidCol()`.
    - In manual mode, setting the source tile and destination tile to the same memory is not supported.
- **Valid region**:
    - The operation uses `dst.GetValidRow()`/`dst.GetValidCol()` as the iteration domain.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileDst = Tile<TileType::Vec, uint16_t, 16, 16>;
  using TileSrc = Tile<TileType::Vec, uint16_t, 16, 16>;
  TileDst dst;
  TileSrc src;
  TANDS(dst, src, 0xffu);
}
```
