# TROWMAX

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T05:53:53.612Z pushedAt=2026-11-02T03:00:52.228Z nl=2 -->

## Instruction Diagram

![TROWMAX tile operation](../figures/isa/TROWMAX.svg)

## Introduction

Reduces each row by taking the maximum value across columns.

## Mathematical Semantics

Assume `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R`:

$$ \mathrm{dst}_{i,0} = \max_{0 \le j < C} \mathrm{src}_{i,j} $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMAX(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## Constraints

### General Constraints or Checks

- `dst` and `src` must both be `TileType::Vec`.
- `src` must use the standard ND layout: row-major and non-fractal (`BLayout::RowMajor`, `SLayout::NoneBox`).
- `dst` must use one of the following two non-fractal layouts:
    - ND layout (`BLayout::RowMajor`, `SLayout::NoneBox`), or
    - DN layout with the number of columns strictly equal to 1 (`BLayout::ColMajor`, `SLayout::NoneBox`, `Cols == 1`).
- The element types of `dst` and `src` must be consistent.
- Runtime valid region checks:
    - `src.GetValidRow() != 0`
    - `src.GetValidCol() != 0`
    - `src.GetValidRow() == dst.GetValidRow()`
- The built-in API signature requires explicitly passing the `tmp` operand.

### Implementation Check for Atlas A2/A3 Training Products/Atlas A2/A3 Inference Products

- Supported element types: `half`, `float`, `int32_t`, `int16_t`.

### Ascend 950PR/Ascend 950DT Implementation Check

- Supported element types: `half`, `float`, `int32_t`, `int64_t`, `uint64_t`, `int16_t`, `int8_t`, `uint8_t`.
- For `int64_t` / `uint64_t`:
    - The output valid column count should be 1; only column 0 of each valid row is written, and the remaining physical padding is preserved.
    - ND output requires physical `Cols % 4 == 0`; DN output requires physical `Cols == 1` and `Rows % 4 == 0`. The valid row count does not need to be a multiple of 4.
    - The row stride is determined by the physical shape. For details, see [Shape and Layout Conventions](conventions.md).

## Temporary Space

### Atlas A2/A3 Training Products/Atlas A2/A3 Inference Products

`tmp` **is used** as the scratch storage for row maximum value reduction.

- For **integer** types (`int32_t`, `int16_t`): `tmp` is used as a per-row accumulator buffer (1 block). For each row, `tmp` is initialized to the minimum representable value, and then the blocks of `src` are accumulated through `vmax`. The final maximum value is read from `tmp` in scalar mode.
  - `tmp` size: at least 1 row and `BLOCK_BYTE_SIZE / sizeof(T)` columns (8 for `int32_t`, 16 for `int16_t`).
- For **floating-point** types (`float`, `half`): `tmp` is used for binary tree reduction through `vcmax`/`vcgmax`.
  - Safe default settings: set `tmp` to the same shape as `src`.

### Ascend 950PR/Ascend 950DT

`tmp` is accepted by the API but not used. 64-bit integers use exact integer reduction without floating-point conversion.

## Examples

### Automatic

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWMAX(dst, src, tmp);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(tmp, 0x3000);
  TROWMAX(dst, src, tmp);
}
```

### 64-Bit ND Output (Ascend 950PR/Ascend 950DT)

The output physical shape is `[64,4]` and the valid shape is `[64,1]`; it writes one 8-byte result every 32 bytes.

```cpp
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

void example_int64() {
  using SrcT = Tile<TileType::Vec, int64_t, 64, 16>;
  using DstT = Tile<TileType::Vec, int64_t, 64, 4, BLayout::RowMajor, 64, 1>;
  SrcT src, tmp;
  DstT dst;
  TROWMAX(dst, src, tmp);
}
```
