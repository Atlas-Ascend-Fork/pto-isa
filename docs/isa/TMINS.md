# TMINS

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T05:19:22.511Z pushedAt=2026-11-02T03:00:52.199Z nl=2 -->

## Instruction Diagram

![TMINS tile operation](../figures/isa/TMINS.svg)

## Introduction

Computes the element-wise minimum value of a tile and a scalar.

## Mathematical Semantics

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \min(\mathrm{src}_{i,j}, \mathrm{scalar}) $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TMINS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar, WaitEvents &... events);
```

## Constraints

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - `TileData::DType` must be one of the following: `int32_t`, `int`, `int16_t`, `half`, `float16_t`, `float`, `float32_t`.
    - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - `TileData::DType` must be one of the following: `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `int64_t`, `uint64_t`, `half`, `float`, `bfloat16_t`.
    - Runtime: `src.GetValidCol() == dst.GetValidCol()`.
- **General constraints**:
    - `dst` and `src` must use the same element type.
    - The scalar type must be consistent with the tile data type.
    - The tile position must be a vector (`TileData::Loc == TileType::Vec`).
- **Valid region**:
    - The operation uses `dst.GetValidRow()`/`dst.GetValidCol()` as the iteration domain.

## Examples

### Automatic

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TMINS(dst, src, 0.0f);
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
  TMINS(dst, src, 0.0f);
}
```
