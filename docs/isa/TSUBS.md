# TSUBS

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T06:13:37.011Z pushedAt=2026-11-02T03:00:52.240Z nl=2 -->

## Instruction Diagram

![TSUBS tile operation](../figures/isa/TSUBS.svg)

## Introduction

Subtracts a scalar from each element in the tile.

## Mathematical Semantics

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} - \mathrm{scalar} $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSUBS(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType scalar, WaitEvents &... events);
```

## Constraints

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - `TileData::DType` must be one of the following: `int32_t`, `int`, `int16_t`, `half`, `float16_t`, `float`, `float32_t`.
    - The tile position must be a vector (`TileData::Loc == TileType::Vec`).
    - Runtime: `src0.GetValidRow() == dst.GetValidRow()` and `src0.GetValidCol() == dst.GetValidCol()`.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - `TileData::DType` must be one of the following: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `int64_t`, `uint64_t`, `half`, `float`, `bfloat16_t`.
    - The tile position must be a vector (`TileDataDst::Loc == TileType::Vec` and `TileDataSrc::Loc == TileType::Vec`).
    - Static valid boundary: `TileDataDst::ValidRow <= TileDataDst::Rows`, `TileDataDst::ValidCol <= TileDataDst::Cols`, `TileDataSrc::ValidRow <= TileDataSrc::Rows`, and `TileDataSrc::ValidCol <= TileDataSrc::Cols`.
    - Runtime: `src0.GetValidRow() == dst.GetValidRow()` and `src0.GetValidCol() == dst.GetValidCol()`.
- **General constraints**:
    - `dst` and `src0` must use the same element type.
    - The scalar type must match `TileDataSrc::DType`.
- **Valid region**:
    - The operation uses `dst.GetValidRow()`/`dst.GetValidCol()` as the iteration domain.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TSUBS(out, x, 1.0f);
}
```
