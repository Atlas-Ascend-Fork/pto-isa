# TSHR

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T06:07:51.367Z pushedAt=2026-11-02T03:00:52.235Z nl=2 -->

## Instruction Diagram

![TSHR tile operation](../figures/isa/TSHR.svg)

## Introduction

Performs element-wise right shift on two tiles.

## Mathematical Semantics

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \gg \mathrm{src1}_{i,j} $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSHR(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - The supported element types are `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, and `int32_t`.
    - `dst`, `src0`, and `src1` must use the same element type.
    - `dst`, `src0`, and `src1` must be row-major.
    - Runtime: `src0.GetValidRow()/GetValidCol()` and `src1.GetValidRow()/GetValidCol()` must be consistent with `dst`.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - The supported element types are `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `int64_t`, and `uint64_t` (consistent with Atlas A2/A3 training products/Atlas A2/A3 inference products).
    - `dst`, `src0`, and `src1` must use the same element type.
    - `dst`, `src0`, and `src1` must be row-major.
    - Runtime: `src0.GetValidRow()/GetValidCol()` and `src1.GetValidRow()/GetValidCol()` must be consistent with `dst`.
- **Implementation check (A6)**:
    - The supported element types are `uint8_t`, `int8_t`, `uint16_t`, `int16_t`, `uint32_t`, `int32_t`, `int64_t`, and `uint64_t`.
    - `dst`, `src0`, and `src1` must use the same element type.
    - `dst`, `src0`, and `src1` must be row-major.
    - Runtime: `src0.GetValidRow()/GetValidCol()` and `src1.GetValidRow()/GetValidCol()` must be consistent with `dst`.
    - The shift amount (`src1`) is interpreted as a signed number: a negative value reverses the shift direction (shifts left by `|src1|`).
    - For `int64_t` and `uint64_t`, the actual shift amount is `src1 & 63`.
- **Valid region**:
    - The operation uses `dst.GetValidRow()`/`dst.GetValidCol()` as the iteration domain.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileT x, sh, out;
  TSHR(out, x, sh);
}
```
