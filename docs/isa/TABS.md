# TABS

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:24:34.586Z pushedAt=2026-11-02T03:00:52.153Z nl=2 -->

## Instruction Diagram

![TABS tile operation](../figures/isa/TABS.svg)

## Introduction

Element-wise absolute value of a tile.

## Mathematical Semantics

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \left|\mathrm{src}_{i,j}\right| $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TABS(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## Constraints

- **Implementation check (CPU sim)**:
    - The data types of `dst` and `src` must be the same, and must be one of the following: `int32_t`, `int`, `int16_t`, `half`, `float`.
    - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`. If they are inconsistent, an assertion is triggered.
    - The implementation iterates over `dst.GetValidRow()`/`dst.GetValidCol()`.
- **Implementation check (Costmodel)**:
    - `TileData::DType` must be one of the following: `int32_t`, `int16_t`, `int8_t`, `uint8_t`, `half`, `float`.
- **Implementation check (NPU)**:
    - For Atlas A3 training products/Atlas A3 inference products, `TileData::DType` must be one of the following: `float` or `half`.
    - For Ascend 950PR/Ascend 950DT, `TileData::DType` must be one of the following: `int32_t`, `int16_t`, `int8_t`, `int64_t`, `half`, `float`.
    - The tile position must be a vector (`TileData::Loc == TileType::Vec`).
    - Static valid boundary: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
    - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`.
    - The tile layout must be row-major (`TileData::isRowMajor`).
- **Valid region**:
    - The operation uses `dst.GetValidRow()`/`dst.GetValidCol()` as the iteration domain.
    - Provided that the element types are the same, `dst` and `src` can use different C++ Tile types, including mixing static and dynamic `ValidRow`/`ValidCol` template parameters. CPU_SIM computes addresses based on each operand's own Tile layout and physical shape.

## Examples

### Automatic

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TABS(dst, src);
}
```

### Automatic (Mixed Static and Dynamic Valid Shapes)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_mixed_valid_shape() {
  using DynamicTile = Tile<TileType::Vec, int32_t, 16, 16, BLayout::RowMajor, -1, -1>;
  using StaticTile = Tile<TileType::Vec, int32_t, 16, 16, BLayout::RowMajor, 16, 16>;
  DynamicTile dst(16, 16);
  StaticTile src;
  TABS(dst, src);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TABS(dst, src);
}
```
