# TNEG

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T05:25:17.850Z pushedAt=2026-11-02T03:00:52.203Z nl=2 -->

## Instruction Diagram

![TNEG tile operation](../figures/isa/TNEG.svg)

## Introduction

Element-wise negation of a tile.

## Mathematical Semantics

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = -\mathrm{src}_{i,j} $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TNEG(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## Constraints

- The operation iterates over `dst.GetValidRow()`/`dst.GetValidCol()`.
- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**: `TileData::DType` must be one of the following: `int32_t`, `int16_t`, `half`, `float`.
- **Implementation check (Ascend 950PR/Ascend 950DT)**: `TileData::DType` must be one of the following: `int32_t`, `int16_t`, `uint32_t`, `uint16_t`, `half`, `float`, `bfloat16_t`.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TNEG(out, x);
}
```
