# TROWEXPAND

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T05:44:00.743Z pushedAt=2026-11-02T03:00:52.221Z nl=2 -->

## Instruction Diagram

![TROWEXPAND tile operation](../figures/isa/TROWEXPAND.svg)

## Introduction

Broadcasts the first element of each source row to the destination row.

## Mathematical Semantics

Assume `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{i,0} $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPAND(TileDataDst &dst, TileDataSrc &src, WaitEvents &... events);
```

## Constraints

Implementation check (NPU):

- Tile type: `dst` and `src` must be `TileType::Vec`.
- Tile layout: `dst` is non-fractal RowMajor; `src` can be non-fractal RowMajor or ColMajor (`SLayout::NoneBox`). The input and output element types must be the same.
- The source static `ValidCol` of A5 must be 1 or the dynamic value -1; a single-scalar row broadcast should use a valid column count of 1.
- Data type (Atlas A2/A3 training products/Atlas A2/A3 inference products): The element type must be one of the following: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`.
- Data type (Ascend 950PR/Ascend 950DT): The element type must be one of the following: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
- Runtime valid region check:
    - Atlas A2/A3 training products/Atlas A2/A3 inference products: if any of `dstValidRow`, `dstValidCol`, `srcValidRow`, or `srcValidCol` is zero, return early.
    - Ascend 950PR/Ascend 950DT: assert `srcValidRow == dstValidRow`, and assert `srcValidRow != 0 && srcValidCol != 0`.

- 64-bit input (Ascend 950PR/Ascend 950DT): The ND source can use physical `[64,4]` and valid `[64,1]`, and the DN source can use compact `[64,1]`. Each row broadcasts `src[i,0]`, and the address uses the physical `RowStride`.
- Unlike binary row broadcasts such as [TROWEXPANDADD](TROWEXPANDADD.md), the ND source of this instruction still provides a single scalar per row and does not repeatedly broadcast 32-byte blocks.

## Examples

### Automatic

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 1>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TROWEXPAND(dst, src);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 1>;
  using DstT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TROWEXPAND(dst, src);
}
```
