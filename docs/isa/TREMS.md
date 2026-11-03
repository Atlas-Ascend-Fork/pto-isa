# TREMS

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T05:41:04.343Z pushedAt=2026-11-02T03:00:52.219Z nl=2 -->

## Instruction Diagram

![TREMS tile operation](../figures/isa/TREMS.svg)

## Introduction

Takes the element-wise remainder with a scalar: `remainder(src, scalar)`. When the scalar divisor is non-zero, the sign of a non-zero result is the same as that of the divisor.

## Mathematical Semantics

For each element `(i, j)` in the valid region whose divisor is non-zero:

$$\mathrm{dst}_{i,j} = \mathrm{src}_{i,j} - \mathrm{floor}(\frac{\mathrm{src}_{i,j}}{\mathrm{scalar}}) \times \mathrm{scalar}$$

When `scalar` is non-zero, the sign of a non-zero result is the same as that of `scalar`; when the division is exact, the result is zero.
For example, `remainder(-7, 3) = 2`, `remainder(7, -3) = -2`, and `remainder(-6, 3) = 0`.
On Ascend 950PR/Ascend 950DT, `int64_t` follows the floor-based definition above, and when the divisor is non-zero, its remainder semantics are consistent with those of `int32_t`.

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <auto PrecisionType = RemSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename TileDataTmp,
          typename... WaitEvents>
PTO_INST RecordEvent TREMS(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar, TileDataTmp &tmp,
                           WaitEvents &...events);
```

`PrecisionType` can specify the following values:

* `RemSAlgorithm::DEFAULT`: normal algorithm that is fast but less accurate.
* `RemSAlgorithm::HIGH_PRECISION`: high-precision algorithm that is slower and supports only the `float` type.

## Constraints

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - `dst` and `src` must use the same element type.
    - Supported element types: `float` and `int32_t`.
    - `dst` and `src` must be vector tiles.
    - `dst` and `src` must be row-major.
    - Runtime: `dst.GetValidRow() == src.GetValidRow() > 0` and `dst.GetValidCol() == src.GetValidCol() > 0`.
    - **tmp buffer requirements**:
      - `tmp.GetValidCol() >= dst.GetValidCol()` (at least the same number of columns as dst)
      - `tmp.GetValidRow() >= 1` (at least 1 row)
      - The data type must match `TileDataDst::DType`.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - `dst` and `src` must use the same element type.
    - Supported element types: `float`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `int16_t`, and `uint16_t`.
    - `dst` and `src` must be vector tiles.
    - The static valid boundaries of both tiles must satisfy `ValidRow <= Rows` and `ValidCol <= Cols`.
    - Runtime: `dst.GetValidRow() == src.GetValidRow()` and `dst.GetValidCol() == src.GetValidCol()`.
    - Note: The tmp parameter is accepted but not validated or used on Ascend 950PR/Ascend 950DT.
- **Division by zero**:
    - On Ascend 950PR/Ascend 950DT, when `scalar` is zero, `int64_t` and `uint64_t` return `-1` and `UINT64_MAX` (all 64 bits set to 1), respectively.
    - The behavior is target-defined; the CPU simulator asserts in debug builds.
- **Valid region**:
    - The operation uses `dst.GetValidRow()`/`dst.GetValidCol()` as the iteration domain.
- **For `int32_t` inputs (Atlas A2/A3 training products/Atlas A2/A3 inference products only)**: The elements of `src` and `scalar` must be within the range `[-2^24, 2^24]` (that is, `[-16777216, 16777216]`) to ensure exact conversion to float32 during computation.
- **High-precision algorithm**:
    - Valid only on Ascend 950PR/Ascend 950DT; the `PrecisionType` option is ignored on Atlas A3 training products/Atlas A3 inference products.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  Tile<TileType::Vec, float, 16, 16> tmp;
  TREMS(out, x, 3.0f, tmp);
}
```
