# TSCATTER

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T06:01:12.220Z pushedAt=2026-11-02T03:00:52.232Z nl=2 -->

## Instruction Diagram

![TSCATTER tile operation](../figures/isa/TSCATTER.svg)

## Introduction

TSCATTER provides two operation modes:

1. **Index-based scatter**: Uses per-element flat element offsets to scatter the elements of the source tile into the destination Tile.
2. **Mask scatter**: Scatters source elements to destination positions according to a mask pattern, interleaving zero values between elements. It supports two modes: row-wise scatter (`SCATTER_ROW`) and column-wise scatter (`SCATTER_COL`).

## Mathematical Semantics

### Index-based Scatter

Assume `R = idx.GetValidRow()` and `C = idx.GetValidCol()`. For `0 <= i < R` and `0 <= j < C`, write:

$$ \mathrm{dst.data()}[\mathrm{idx}_{i,j}] = \mathrm{src}_{i,j} $$

`idx[i,j]` is an element offset relative to `dst.data()`, not a byte offset or a row number. In a RowMajor destination, the index of coordinate `(r,c)` is `r * DstTile::Cols + c`; in ColMajor, it is `c * DstTile::Rows + r`. The source valid region must cover the index valid region. The caller must ensure that the index is non-negative and smaller than the number of physical elements in the destination; the implementation does not perform boundary checks.

If multiple elements map to the same destination position, the final value is implementation-defined, and the write order is not guaranteed.

### Mask Scatter

For mask mode `P`, the source elements are scattered and interleaved with zero values. The scatter direction is controlled by `ScatterAxis`:

Assume the expansion multiple is `f`, the selected position is `pos_P`, and `q` traverses the positions other than `pos_P` in `0 <= q < f`. For `P0101/P1010`, `pos_P` is 0/1 respectively; for `P0001/P0010/P0100/P1000`, it is 0/1/2/3 respectively; for `P1111`, it is 0.

#### SCATTER_ROW (Default)

Scatters along the column direction and expands the column dimension:

$$ \mathrm{dst}_{i, f \cdot j + \mathrm{pos}_P} = \mathrm{src}_{i,j} $$

$$ \mathrm{dst}_{i, f \cdot j + q} = 0 $$

Where:

- `DstTileData::ValidCol` = `SrcTileData::ValidCol` × expansion multiple
- `DstTileData::ValidRow` = `SrcTileData::ValidRow`

#### SCATTER_COL

Scatters along the row direction and expands the row dimension:

$$ \mathrm{dst}_{f \cdot i + \mathrm{pos}_P, j} = \mathrm{src}_{i,j} $$

$$ \mathrm{dst}_{f \cdot i + q, j} = 0 $$

Where:

- `DstTileData::ValidRow` = `SrcTileData::ValidRow` × expansion multiple
- `DstTileData::ValidCol` = `SrcTileData::ValidCol`

#### Expansion Multiple

- `P1010` or `P0101`: expansion multiple = 2
- `P0001`, `P0010`, `P0100`, `P1000`: expansion multiple = 4
- `P1111`: expansion multiple = 1 (equivalent to `TMOV`)

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

### Index-based Scatter

```cpp
template <typename TileDataD, typename TileDataS, typename TileDataI, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(TileDataD& dst, TileDataS& src, TileDataI& indexes, WaitEvents&... events);
```

### Mask Scatter

```cpp
template <MaskPattern maskPattern = MaskPattern::P1111, auto ScatterType = ScatterAxis::SCATTER_ROW,
          typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(DstTileData& dst, SrcTileData& src, WaitEvents&... events);
```

### MaskPattern Enum

Defined in `include/pto/common/type.hpp`:

| Value | Mode | Description | Expansion Multiple |
|---|------|------|---------|
| `P0101` | 01010101... | Takes the first of every two elements | ×2 |
| `P1010` | 10101010... | Takes the second of every two elements | ×2 |
| `P0001` | 00010001... | Takes the first of every four elements | ×4 |
| `P0010` | 00100010... | Takes the second of every four elements | ×4 |
| `P0100` | 01000100... | Takes the third of every four elements | ×4 |
| `P1000` | 10001000... | Takes the fourth of every four elements | ×4 |
| `P1111` | 11111111... | Takes all elements (equivalent to TMOV) | ×1 |

### ScatterAxis Enum

Defined in `include/pto/common/type.hpp`:

| Value | Description |
|---|------|
| `SCATTER_ROW` | Scatters along the column direction and expands the column dimension (default) |
| `SCATTER_COL` | Scatters along the row direction and expands the row dimension |

## Constraints

### Index-based Scatter

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - `TileDataD::Loc`, `TileDataS::Loc`, and `TileDataI::Loc` must be `TileType::Vec`.
    - `TileDataD::DType` and `TileDataS::DType` must be one of the following: `int32_t`, `int16_t`, `int8_t`, `half`, `float16_t`, `float32_t`, `uint32_t`, `uint16_t`, `uint8_t`, `bfloat16_t`.
    - `TileDataI::DType` must be one of the following: `int16_t`, `int32_t`, `uint16_t`, or `uint32_t`.
    - No boundary check is performed on the `indexes` values.
    - Statically valid boundary: `TileDataD::ValidRow <= TileDataD::Rows`, `TileDataD::ValidCol <= TileDataD::Cols`, `TileDataS::ValidRow <= TileDataS::Rows`, `TileDataS::ValidCol <= TileDataS::Cols`, `TileDataI::ValidRow <= TileDataI::Rows`, `TileDataI::ValidCol <= TileDataI::Cols`.
    - `TileDataD::DType` and `TileDataS::DType` must be the same.
    - When the size of `TileDataD::DType` is 4 bytes, the size of `TileDataI::DType` must be 4 bytes.
    - When the size of `TileDataD::DType` is 2 bytes, the size of `TileDataI::DType` must be 2 bytes.
    - When the size of `TileDataD::DType` is 1 byte, the size of `TileDataI::DType` must be 2 bytes.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - `TileDataD::Loc`, `TileDataS::Loc`, and `TileDataI::Loc` must be `TileType::Vec`.
    - `TileDataD::DType` and `TileDataS::DType` must be one of the following: `int32_t`, `int16_t`, `int8_t`, `half`, `float16_t`, `float32_t`, `uint32_t`, `int64_t`, `uint64_t`, `uint16_t`, `uint8_t`, `bfloat16_t`.
    - `TileDataI::DType` must be one of the following: `int16_t`, `int32_t`, `uint16_t`, or `uint32_t`.
    - No boundary check is performed on the `indexes` values.
    - Statically valid boundary: `TileDataD::ValidRow <= TileDataD::Rows`, `TileDataD::ValidCol <= TileDataD::Cols`, `TileDataS::ValidRow <= TileDataS::Rows`, `TileDataS::ValidCol <= TileDataS::Cols`, `TileDataI::ValidRow <= TileDataI::Rows`, `TileDataI::ValidCol <= TileDataI::Cols`.
    - `TileDataD::DType` and `TileDataS::DType` must be the same.
    - When the size of `TileDataD::DType` is 4 bytes, the size of `TileDataI::DType` must be 4 bytes.
    - When the size of `TileDataD::DType` is 2 bytes, the size of `TileDataI::DType` must be 2 bytes.
    - When the size of `TileDataD::DType` is 1 byte, the size of `TileDataI::DType` must be 2 bytes.

    - 64-bit data requires `int32_t` / `uint32_t` indexes; the source and index tiles support RowMajor/ColMajor, using physical row and column strides respectively.

### Mask Scatter

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - `DstTileData::Loc` and `SrcTileData::Loc` must be `TileType::Vec`.
    - `DstTileData::DType` and `SrcTileData::DType` must be one of the following: `int32_t`, `int16_t`, `int8_t`, `half`, `float16_t`, `float32_t`, `uint32_t`, `uint16_t`, `uint8_t`, `bfloat16_t`.
    - `DstTileData::DType` and `SrcTileData::DType` must be the same.
    - `maskPattern` must be within the range from `P0101` to `P1111`.
    - Statically valid boundary: `DstTileData::ValidCol <= DstTileData::Cols`, `SrcTileData::ValidCol <= SrcTileData::Cols`, `DstTileData::ValidRow <= DstTileData::Rows`, `SrcTileData::ValidRow <= SrcTileData::Rows`.
    - The `P1111` mode is equivalent to `TMOV`: it requires `validRow` and `validCol` to match respectively, and is implemented internally through `TMOV_IMPL`.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - `DstTileData::Loc` and `SrcTileData::Loc` must be `TileType::Vec`.
    - `DstTileData::DType` and `SrcTileData::DType` must be one of the following: `int32_t`, `int16_t`, `int8_t`, `half`, `float16_t`, `float32_t`, `uint32_t`, `int64_t`, `uint64_t`, `uint16_t`, `uint8_t`, `bfloat16_t`.
    - `DstTileData::DType` and `SrcTileData::DType` must be the same.
    - `maskPattern` must be within the range from `P0101` to `P1111`.
    - Statically valid boundary: `DstTileData::ValidRow <= DstTileData::Rows`, `DstTileData::ValidCol <= DstTileData::Cols`, `SrcTileData::ValidRow <= SrcTileData::Rows`, `SrcTileData::ValidCol <= SrcTileData::Cols`.
    - Runtime assertions for `SCATTER_ROW` mode:
        - `src.GetValidRow()` must be equal to `dst.GetValidRow()`.
        - `dst.GetValidCol()` must be equal to `src.GetValidCol() × expansion multiple`, where the expansion multiple depends on the mask pattern (1 for P1111, 2 for P1010/P0101, and 4 for P0001/P0010/P0100/P1000).
    - Runtime assertions for `SCATTER_COL` mode:
        - `src.GetValidCol()` must be equal to `dst.GetValidCol()`.
        - `dst.GetValidRow()` must be equal to `src.GetValidRow() × expansion multiple`, where the expansion multiple depends on the mask pattern (1 for P1111, 2 for P1010/P0101, and 4 for P0001/P0010/P0100/P1000).

    - 64-bit mask spreading requires row-major source and destination, and supports six expansion patterns and two `ScatterAxis` values.
    - `P1111` delegates to [TMOV](TMOV.md) and does not perform whole-block zeroing; the current Ascend 950PR/Ascend 950DT does not support `int64_t`/`uint64_t` Vec→Vec moves in this mode.

## Important Notes

For A5, index spreading and non-`P1111` mask spreading first zero the entire destination physical Tile (`Rows * Cols` elements) and then write to the selected positions. Therefore, unselected positions (including physical padding) are zero; padding positions explicitly selected by the index are still written with the source value. The active storage of the source, index, and destination must not overlap.

For the CPU simulator, index spreading updates only the positions specified by the index and preserves other positions; an empty index valid region is also not written. Cross-backend code that requires consistent zero padding should explicitly initialize the destination. `P1111` follows the write semantics of `TMOV`, and the whole-block zeroing rule above does not apply.

## Examples

### Index Scatter (Automatic)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TSCATTER(dst, src, idx);
}
```

### Index Scatter (Manual)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  using IdxT = Tile<TileType::Vec, uint32_t, 16, 16>;
  TileT src, dst;
  IdxT idx;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TASSIGN(idx, 0x3000);
  TSCATTER(dst, src, idx);
}
```

### Mask Scatter (Automatic)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_mask_auto() {
  // P1010: destination size = source size × 2.
  using SrcTileT = Tile<TileType::Vec, half, 16, 64>;
  using DstTileT = Tile<TileType::Vec, half, 16, 128>;
  SrcTileT src;
  DstTileT dst;
  TSCATTER<MaskPattern::P1010>(dst, src);
}

void example_mask_p1000() {
  // P1000: destination size = source size × 4.
  using SrcTileT = Tile<TileType::Vec, float, 16, 64>;
  using DstTileT = Tile<TileType::Vec, float, 16, 256>;
  SrcTileT src;
  DstTileT dst;
  TSCATTER<MaskPattern::P1000>(dst, src);
}

void example_mask_scatter_col() {
  // SCATTER_COL: spread along the row direction, expanding the row dimension.
  // Destination row count = source row count × 2.
  using SrcTileT = Tile<TileType::Vec, half, 64, 16>;
  using DstTileT = Tile<TileType::Vec, half, 128, 16>;
  SrcTileT src;
  DstTileT dst;
  TSCATTER<MaskPattern::P1010, ScatterAxis::SCATTER_COL>(dst, src);
}
```

### Mask Scatter (Manual)

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_mask_manual() {
  using SrcTileT = Tile<TileType::Vec, half, 16, 64>;
  using DstTileT = Tile<TileType::Vec, half, 16, 128>;
  SrcTileT src;
  DstTileT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TSCATTER<MaskPattern::P1010>(dst, src);
}

void example_mask_manual_scatter_col() {
  // SCATTER_COL manual binding mode.
  using SrcTileT = Tile<TileType::Vec, half, 64, 16>;
  using DstTileT = Tile<TileType::Vec, half, 128, 16>;
  SrcTileT src;
  DstTileT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TSCATTER<MaskPattern::P1010, ScatterAxis::SCATTER_COL>(dst, src);
}
```
