# TSHLS

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T06:06:41.848Z pushedAt=2026-11-02T03:00:52.235Z nl=2 -->

## Instruction Diagram

![TSHLS tile operation](../figures/isa/TSHLS.svg)

## Introduction

Performs element-wise left shift on tiles by scalar.

## Mathematical Semantics

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,j} \ll \mathrm{scalar} $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSHLS(TileDataDst &dst, TileDataSrc &src, typename TileDataDst::DType scalar, WaitEvents &... events);
```

## Constraints

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - The supported element types are `int32_t`, `int`, `int16_t`, `uint32_t`, `unsigned int`, and `uint16_t`.
    - `dst` and `src` must use the same element type.
    - `dst` and `src` must be vector tiles.
    - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`.
    - The scalar supports only zero and positive values.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - The supported element types are `int32_t`, `int16_t`, `int8_t`, `uint32_t`, `int64_t`, `uint64_t`, `uint16_t`, and `uint8_t`.
    - `dst` and `src` must use the same element type.
    - `dst` and `src` must be vector tiles.
    - The static valid boundaries of both tiles must satisfy `ValidRow <= Rows` and `ValidCol <= Cols`.
    - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`.
    - The scalar supports only zero and positive values.
    - For `int64_t` and `uint64_t`, the actual shift amount is `scalar & 63`.
- **Implementation check (A6)**:
    - The supported element types are `int32_t`, `int16_t`, `int8_t`, `uint32_t`, `int64_t`, `uint64_t`, `uint16_t`, and `uint8_t`.
    - `dst` and `src` must use the same element type.
    - `dst` and `src` must be vector tiles.
    - The static valid boundaries of both tiles must satisfy `ValidRow <= Rows` and `ValidCol <= Cols`.
    - Runtime: `src.GetValidRow() == dst.GetValidRow()` and `src.GetValidCol() == dst.GetValidCol()`.
    - The scalar is interpreted as a signed number: a negative value reverses the shift direction (shifts right by `|scalar|`).
    - For `int64_t` and `uint64_t`, the actual shift amount is `scalar & 63`.
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
  TSHLS(dst, src, 0x2);
}
```
