# TROWSUM

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T05:58:57.274Z pushedAt=2026-11-02T03:00:52.231Z nl=2 -->

## Instruction Diagram

![TROWSUM tile operation](../figures/isa/TROWSUM.svg)

## Introduction

Reduces each row by summing its columns.

## Mathematical Semantics

Assume `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= i < R`:

$$ \mathrm{dst}_{i,0} = \sum_{j=0}^{C-1} \mathrm{src}_{i,j} $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWSUM(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp, WaitEvents &... events);
```

## Constraints

### General Constraints or Checks

The following constraints describe the NPU backend; see below for CPU_SIM checks and compatibility exceptions.

- `dst` and `src` must both be `TileType::Vec`.
- `src` must use the standard ND layout: row-major and non-fractal (`BLayout::RowMajor`, `SLayout::NoneBox`).
- `dst` must be a non-fractal layout (`!isBoxedLayout`) and use one of the following two layouts:
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

- Supported element types: `half`, `float`, `int32_t`, `int64_t`, `uint64_t`, `int16_t`.
- For `int64_t` / `uint64_t`:
    - The output valid column count should be 1; only column 0 of each valid row is written, and the remaining physical padding is preserved.
    - ND output requires physical `Cols % 4 == 0`; DN output requires physical `Cols == 1` and `Rows % 4 == 0`. The number of valid rows does not need to be a multiple of 4.
    - The row stride is determined by the physical shape. See [Shape and Layout Conventions](conventions.md).

### CPU_SIM Implementation Check

The target architecture is taken from the memory model of the calling thread and does not depend on the host hardware. For how to set it, see
[Selecting the Simulated Target Architecture](../coding/cpu_sim.md#selecting-the-simulated-architecture).

- **A5**: The input and output types must be identical, supporting `half`, `float`, `int16_t`, `int32_t`,
  `int64_t`, and `uint64_t`. Native BF16 or mixed input/output types are not supported.
  The constraints described above for Vec, ND/DN layouts, non-empty input, and equal valid row counts are all checked at runtime;
  the physical row counts of the input and output may differ.
- **A2A3 compatibility path**: The supported input/output type combinations are `half`/`half`, `half`/`float`,
  `bfloat16_t`/`bfloat16_t`, `bfloat16_t`/`float`, `float`/`float`,
  `int16_t`/`int16_t`, `int16_t`/`int32_t`, and `int32_t`/`int32_t`.
  The physical row counts must be equal, but the A5-specific layout and valid-region assertions are not performed.
  These compatibility behaviors do not extend the interface constraints of the A2A3 NPU.
- Type combinations unsupported by both CPU paths are rejected at compile time; type combinations unsupported only by the selected architecture are rejected at runtime.
  An architecture-related assertion failure terminates the process, and defining `NDEBUG` does not disable these checks.
- The caller must ensure that the dynamic valid shape lies within the physical Tile and provide destination storage and a valid region that cover all output rows and column zero.
  The current implementation does not check `dst.GetValidCol()` and does not provide general dynamic-shape bounds checking.
  Both paths write only column zero of the processed rows, leaving the remaining destination elements and valid-shape metadata unchanged.
- **Numerical differences**: The A5 floating-point path uses grouped binary tree reduction, converting back to the element type after each addition;
  The integer path wraps around according to the output bit width without saturation. A2A3 retains the original accumulation loop, where `half`/`bfloat16_t`
  outputs use `float` accumulation followed by conversion, and other outputs use their own types, without adopting A5's unsigned modulo overflow handling.
  The vectorization hints in the A2A3 row-major loop may reorder floating-point additions even when `-ffast-math` is not enabled;
  strict left-to-right accumulation is not guaranteed, nor are bitwise-identical results across different compilers.
- Full bitwise equivalence with real hardware is not promised: A5's floating-point details such as NaN payload and denormal/flush-to-zero have not been proven equivalent,
  and the A2A3 path is provided only for compatibility. When relying on the A5 reduction order, options that allow floating-point reassociation, such as `-ffast-math`, should not be enabled.
  For the grouping order, rounding, and BF16 placeholder type, see [TROWSUM Implementation Notes](../coding/cpu_sim.md#trowsum-implementation-notes).

## Temporary Space

### Atlas A2/A3 Training Products/Atlas A2/A3 Inference Products

`tmp` **is used** as scratch storage for row reduction.

- For **integer** types (`int32_t`, `int16_t`): `tmp` is used as a row-wise accumulator buffer (1 block). For each row, `tmp` is initialized to 0, and then the blocks of `src` are accumulated through `vadd`. The final sum result is read from `tmp` in scalar mode.
  - `tmp` size: at least 1 row and `BLOCK_BYTE_SIZE / sizeof(T)` columns (8 for `int32_t`, 16 for `int16_t`).
- For **floating-point** types (`float`, `half`): `tmp` is used for binary tree reduction through `vcadd`/`vcgadd`.
  - Safe default settings: set `tmp` to the same shape as `src`.

### Ascend 950PR/Ascend 950DT

`tmp` is accepted by the interface but not used. 64-bit integers use exact integer reduction without floating-point conversion.

### CPU_SIM

Both architecture paths accept `tmp` for interface compatibility but do not access its storage. Cross-backend kernels must still reserve the temporary space required by the target NPU backend.

## Examples

The following snippets only show how the call and storage binding are performed; they are not complete executable programs. The source data must be initialized before the reduction.
When compiling under CPU_SIM, define `__CPU_SIM`; the automatic mode example also requires defining `__PTO_AUTO__`.
To verify the A5 path, select A5 during application startup as described in the CPU_SIM guide. These two functions do not select an architecture themselves;
otherwise, the configured default architecture (initially A2A3) is used.

### Automatic

```cpp
#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 1, BLayout::ColMajor>;
  using TmpT = Tile<TileType::Vec, float, 16, 16>;
  SrcT src;
  DstT dst;
  TmpT tmp;
  TROWSUM(dst, src, tmp);
}
```

### Manual

```cpp
#include <cstddef>
#include <cstdint>
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
  TROWSUM(dst, src, tmp);
}
```

### 64-Bit ND Output (Ascend 950PR/Ascend 950DT)

The output physical shape is `[64,4]` and the valid shape is `[64,1]`; an 8-byte result is written every 32 bytes.

Before using `TROWSUM` with 64-bit integers in CPU_SIM, select the A5 architecture; see the CPU_SIM implementation check above for details.

```cpp
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

void example_int64() {
  using SrcT = Tile<TileType::Vec, int64_t, 64, 16>;
  using DstT = Tile<TileType::Vec, int64_t, 64, 4, BLayout::RowMajor, 64, 1>;
  SrcT src, tmp;
  DstT dst;
  TROWSUM(dst, src, tmp);
}
```
