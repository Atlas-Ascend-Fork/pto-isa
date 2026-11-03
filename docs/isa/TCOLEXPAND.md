# TCOLEXPAND

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:39:58.557Z pushedAt=2026-11-02T03:00:52.163Z nl=2 -->

## Instruction Diagram

![TCOLEXPAND tile operation](../figures/isa/TCOLEXPAND.svg)

## Introduction

Broadcasts the first element of each source column into the destination column.

## Mathematical Semantics

Assume `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{0,j} $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPAND(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## Constraints

- The operation iterates over `dst.GetValidRow()`/`dst.GetValidCol()`.
- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**: `TileData::DType` must be a 1-, 2-, or 4-byte type (b8/b16/b32): `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`.
- **Implementation check (Ascend 950PR/Ascend 950DT)**: `TileData::DType` must be a 1-, 2-, 4-, or 8-byte type (b8/b16/b32/b64): `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.

- **Shape and layout (Ascend 950PR/Ascend 950DT)**:
    - The source and destination are both non-fractal RowMajor Vec Tiles of the same type.
    - `src.GetValidCol() == dst.GetValidCol()`, and the valid row and column counts of the source must be non-zero.
    - Only row 0 of the source is broadcast, and the valid region is written according to the destination physical row stride.
    - For 64-bit types, the physical `Cols % 4 == 0` is required, while the valid column count does not need to be aligned.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src, dst;
  TCOLEXPAND(dst, src);
}
```
