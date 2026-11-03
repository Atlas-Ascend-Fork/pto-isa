# TPARTMIN

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T05:29:52.800Z pushedAt=2026-11-02T03:00:52.209Z nl=2 -->

## Instruction Diagram

![TPARTMIN tile operation](../figures/isa/TPARTMIN.svg)

## Introduction

Performs element-wise minimum value selection within the destination valid region. If both `src0` and `src1` are valid at a position, the result is `min(src0, src1)`; if only one input is valid at that position, the result takes the value of that input directly. Other cases where the valid regions do not match are implementation-defined.

## Mathematical Semantics

For each element `(i, j)` in the destination valid region:

$
\mathrm{dst}_{i,j} =
\begin{cases}
\min(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) & \text{if both inputs are defined at } (i,j) \text{ } \\\\
\mathrm{src0}_{i,j} & \text{if only src0 is defined at } (i,j) \text{ } \\\\
\mathrm{src1}_{i,j} & \text{if only src1 is defined at } (i,j) \text{ }
\end{cases}
$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMIN(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

### General Constraints or Checks

- The element types of `dst`, `src0`, and `src1` must be consistent.
- The destination valid region defines the computation range of the result.
- For each element within the destination valid region:
    - If both inputs are valid, the element-wise minimum value operation is performed;
    - If only one input is valid, the result directly takes the value of that input.
- If the valid region of `dst` is zero, the instruction returns directly.
- The supported partial valid region modes require that the valid region of at least one source tile is exactly identical to `dst`, and the valid region of the other source tile cannot exceed `dst` in either dimension.
- For valid region combinations outside the above range, the behavior is defined by the specific implementation.

### Implementation Check for Atlas A2/A3 Training Products/Atlas A2/A3 Inference Products

- Supported element types: `int32_t`, `int`, `int16_t`, `half`, `float16_t`, `float`, `float32_t`.
- `dst`, `src0`, and `src1` must all be row-major (`isRowMajor`).

### Ascend 950PR/Ascend 950DT Implementation Check

- Supported element types: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.

## Examples

### Automatic

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TPARTMIN(dst, src0, src1);
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
  TPARTMIN(dst, src0, src1);
}
```
