# TADD

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:30:35.573Z pushedAt=2026-11-02T03:00:52.154Z nl=2 -->

## Instruction Diagram

![TADD tile operation](../figures/isa/TADD.svg)

## Introduction

Element-wise addition of two tiles.

## Mathematical Semantics

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + \mathrm{src1}_{i,j} $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - `dst`, `src0`, and `src1` must have the same data type, which must be one of the following: `int32_t`, `int16_t`, `half`, `float`.
    - The Tile layouts of `dst`, `src0`, and `src1` must all be row-major (`TileData::isRowMajor`).
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - `dst`, `src0`, and `src1` must have the same data type, which must be one of the following: `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `float`, `int16_t`, `uint16_t`, `half`, `bfloat16_t`, `uint8_t`, `int8_t`.
    - The tile layouts of `dst`, `src0`, and `src1` must all be row-major (`TileData::isRowMajor`).
- **Implementation check (CPU_SIM)**:
    - The three operands must have the same data type. CPU_SIM does not additionally restrict to row-major; row-major, column-major, and other supported Tile layouts each use the corresponding operand's own offset for calculation.
- **Valid region**:
    - The operation uses `dst.GetValidRow()`/`dst.GetValidCol()` as the iteration domain.
    - `dst`, `src0`, and `src1` may use different C++ Tile types, including different static or dynamic `ValidRow`/`ValidCol` template parameters, provided that the three operands have the same element type.
    - A2A3, A5, and CPU_SIM all require that the runtime valid row and column counts of `src0`, `src1`, and `dst` be exactly identical; a mismatch triggers an assertion failure. CPU_SIM computes addresses separately based on each operand's own Tile layout and physical shape.

## Examples

### Automatic

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TADD(dst, src0, src1);
}
```

### Automatic (Mixed Static and Dynamic Valid Shapes)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_mixed_valid_shape() {
  using DynamicTile = Tile<TileType::Vec, int32_t, 16, 16, BLayout::RowMajor, -1, -1>;
  using StaticTile = Tile<TileType::Vec, int32_t, 16, 16, BLayout::RowMajor, 16, 16>;
  DynamicTile dst(16, 16), src1(16, 16);
  StaticTile src0;
  TADD(dst, src0, src1);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TADD(dst, src0, src1);
}
```
