# TAND

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:33:22.863Z pushedAt=2026-11-02T03:00:52.157Z nl=2 -->

## Instruction Diagram

![TAND tile operation](../figures/isa/TAND.svg)

## Introduction

Element-wise bitwise AND on two tiles.

## Mathematical Semantics

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} \;\&\; \mathrm{src1}_{i,j} $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TAND(TileData &dst, TileData &src0, TileData &src1, WaitEvents &... events);
```

## Constraints

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - The supported element types are 1-byte, 2-byte, or 4-byte integer types.
    - `dst`, `src0`, and `src1` must use the same element type.
    - `dst`, `src0`, and `src1` must be row-major.
    - Runtime: `src0.GetValidRow()/GetValidCol()` and `src1.GetValidRow()/GetValidCol()` must be consistent with `dst`.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - The supported element types are 1-byte, 2-byte, 4-byte, or 8-byte integer types.
    - `dst`, `src0`, and `src1` must use the same element type.
    - `dst`, `src0`, and `src1` must be row-major.
    - Runtime: `src0.GetValidRow()/GetValidCol()` and `src1.GetValidRow()/GetValidCol()` must be consistent with `dst`.
- **Valid region**:
    - The operation uses `dst.GetValidRow()`/`dst.GetValidCol()` as the iteration domain.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, int32_t, 16, 16>;
  TileT a, b, out;
  TAND(out, a, b);
}
```
