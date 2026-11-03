# TCMPS

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:37:52.662Z pushedAt=2026-11-02T03:00:52.161Z nl=2 -->

## Instruction Diagram

![TCMPS tile operation](../figures/isa/TCMPS.svg)

## Introduction

Compares a Tile with a **scalar** or another Tile, and writes the element-wise comparison result. The Tile form has the following backend differences in behavior.

Provides two overload forms:

- **Scalar form**: Compares each element of `src0` with a scalar value.
- **Tile form**: A2/A3 compares each element of `src0` with the scalar broadcast from the first element of the `src1` Tile; A5 and CPU_SIM compare the corresponding `src0[i,j]` and `src1[i,j]` element by element.

## Mathematical Semantics

**Scalar form**: for each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \left(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{scalar}\right) $$

**Tile form on A2/A3**: for each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \left(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{src1}_{0,0}\right) $$

**Tile form on A5 and CPU_SIM**: for each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = \left(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{src1}_{i,j}\right) $$

The encoding/type of `dst` is implementation-defined (a bit-packed mask tile, where each bit represents a comparison result).

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/type.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

**Scalar form**: compares a tile with a scalar:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents,
          std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TCMPS(TileDataDst& dst, TileDataSrc& src0,
                           typename TileDataSrc::DType scalar, CmpMode mode,
                           WaitEvents&... events);
```

**Tile form**: compares a Tile with another Tile (see the backend behavior above):

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
          typename... WaitEvents,
          std::enable_if_t<is_tile_data_v<TileDataSrc1> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TCMPS(TileDataDst& dst, TileDataSrc0& src0,
                           TileDataSrc1& src1, CmpMode mode,
                           WaitEvents&... events);
```

## Constraints

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - `TileData::DType` must be one of the following: `int32_t`, `float`, `half`, `uint16_t`, `int16_t`.
    - The tile layout must be row-major (`TileData::isRowMajor`).
    - When the input type is `int32_t`, only `CmpMode::EQ` is supported; other comparison modes fall back to `EQ`.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - `TileData::DType` must be one of the following: `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `float`, `int16_t`, `uint16_t`, `half`, `uint8_t`, `int8_t`, `bfloat16_t`.
    - The tile layout must be row-major (`TileData::isRowMajor`).
- **General constraints**:
    - The tile positions of both `src` and `dst` must be vectors (`TileData::Loc == TileType::Vec`).
    - Static valid boundary: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
    - Data types: `src0` and `src1` must have the same data type.
    - The tile form of A5 and CPU_SIM compares `src0` with `src1` element by element; A2/A3 broadcasts `src1[0,0]`.
- **Backend shape check**:
    - A2/A3 and A5 assert that `src0` and `dst` have the same number of valid rows, but do not explicitly check the number of valid columns.
    - CPU_SIM does not assert that the valid shapes are the same. It computes over the valid region of `src0`, limits the bit-packed result writes by the valid shape of `dst`, and requires the Tile form of `src1` to provide readable elements at the corresponding positions.
- **Valid region**:
    - This operation uses `src0.GetValidRow()`/`src0.GetValidCol()` as the iteration domain.
- **Mask encoding**:
    - 64-bit inputs (Ascend 950PR/Ascend 950DT): the comparison result of column `j` is stored in bit `j % 8` of byte `j / 8` of that row, least significant bit first.
    - For a `uint8_t` mask, the valid shape can be set to `[R, ceil(C / 8)]`, with the physical `Cols` aligned to 32 bytes, where `[R,C]` is the source valid shape. Row addresses are computed by the destination physical stride; the padding values after the last valid bit are unspecified.

- **Comparison mode**:
    - Supports `CmpMode::EQ`, `CmpMode::NE`, `CmpMode::LT`, `CmpMode::GT`, `CmpMode::LE`, and `CmpMode::GE` (note: on Atlas A2/A3 training products/Atlas A2/A3 inference products, when the input type is `int32_t`, only `CmpMode::EQ` is supported, and other modes fall back to `EQ`; Ascend 950PR/Ascend 950DT support all modes).

## Examples

### Automatic

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src;
  DstT dst(16, 2);
  TCMPS(dst, src, 0.0f, CmpMode::GT);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src;
  DstT dst(16, 2);
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TCMPS(dst, src, 0.0f, CmpMode::GT);
}
```

### Tile Form (Compared with Another Tile)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_tile() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
  SrcT src0, src1;
  DstT dst(16, 2);
  // A2/A3: broadcast src1[0,0]; A5 and CPU_SIM: compare elements at corresponding positions.
  TCMPS(dst, src0, src1, CmpMode::GE);
}
```
