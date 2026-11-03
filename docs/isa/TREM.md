# TREM

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T05:40:13.850Z pushedAt=2026-11-02T03:00:52.218Z nl=2 -->

## Instruction Diagram

![TREM tile operation](../figures/isa/TREM.svg)

## Introduction

Performs element-wise remainder operation on two tiles. When the divisor is non-zero, the sign of a nonzero result is the same as that of the divisor.

## Mathematical Semantics

For each element `(i, j)` in the valid region where the divisor is nonzero:

$$\mathrm{dst}_{i,j} = \mathrm{remainder}(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) = \mathrm{src0}_{i,j} - \mathrm{floor}(\frac{\mathrm{src0}_{i,j}}{\mathrm{src1}_{i,j}}) \times \mathrm{src1}_{i,j}$$

When the divisor (`src1`) is nonzero, the sign of a nonzero result is the same as that of the divisor; when the division is exact, the result is zero.
For example, `remainder(-7, 3) = 2`, `remainder(7, -3) = -2`, `remainder(-6, 3) = 0`.
On Ascend 950PR/Ascend 950DT, `int64_t` follows the floor definition above, and when the divisor is nonzero, the remainder semantics are consistent with `int32_t`.

**Note**: This differs from `TFMOD`, and the result sign of `TFMOD` is the same as that of the dividend (`src0`).

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <auto PrecisionType = RemAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0,
          typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TREM(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataTmp &tmp, WaitEvents &... events);
```

## Constraints

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - `TileData::DType` must be one of the following: `float`, `float32_t`, `int32_t`.
    - The tile layout must be row-major (`TileData::isRowMajor`).
    - The tile position must be a vector (`TileData::Loc == TileType::Vec`).
    - Runtime: the `src0`, `src1`, and `dst` tiles should have the same `validRow/validCol`.
    - `tmp` must have at least 1 valid row and `TileDataDst::RowStride + 8 * ceil(TileDataDst::RowStride / 256) + 8` valid columns, used in sequence for the intermediate result, the 32-byte-aligned compressed comparison mask, and the 32-byte address buffer.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - `TileData::DType` must be one of the following: `half`, `float`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`.
    - The tile layout must be row-major (`TileData::isRowMajor`).
    - The tile position must be a vector (`TileData::Loc == TileType::Vec`).
    - Runtime: the `src0`, `src1`, and `dst` tiles should have the same `validRow/validCol`.
    - Note: The `tmp` parameter is accepted but not used on Ascend 950PR/Ascend 950DT.
- **Valid region**:
    - The operation uses `dst.GetValidRow()`/`dst.GetValidCol()` as the iteration domain.
- **Division by zero**:
    - On Ascend 950PR/Ascend 950DT, `int64_t` and `uint64_t` return `-1` and `UINT64_MAX` (all 64 bits set to 1), respectively, when the divisor is zero.
    - The behavior is target-defined; the CPU emulation asserts in debug builds.
- **High-precision arithmetic**:
    - It is valid only for the `float` type on Ascend 950PR/Ascend 950DT; the `PrecisionType` option is ignored on Atlas A2/A3 training products/Atlas A2/A3 inference products.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using TmpT = Tile<TileType::Vec, float, 1, 32>;
  TileT dst, src0, src1;
  TmpT tmp;
  TREM(dst, src0, src1, tmp);
}
```
