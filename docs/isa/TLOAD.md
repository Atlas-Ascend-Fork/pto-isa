# TLOAD

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T05:10:12.307Z pushedAt=2026-11-02T03:00:52.193Z nl=1 -->

## Instruction Diagram

![TLOAD tile operation](../figures/isa/TLOAD.svg)

## Introduction

Loads data from GlobalTensor (GM) to a tile.

## Mathematical Semantics

The symbolic representation depends on the shape/stride of `GlobalTensor` and the layout of `Tile`. Conceptually (two-dimensional view, with base offsets `r_0` and `c_0`):

$$ \mathrm{dst}_{i,j} = \mathrm{src}_{r_0 + i,\; c_0 + j} $$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);

template <TLoadL2Hint l2Control, typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData &dst, GlobalData &src, WaitEvents &... events);
```

The original `TLOAD(dst, src)` and `TLOAD<TileT, GTensor>(dst, src)` remain available. The form with `TLoadL2Hint` as the first template parameter is an additional overload (`l2Control` has no default value).

## L2 Cache Hint

Optional first template parameter overload (the original `TLOAD(dst, src)` remains unchanged):

```cpp
TLOAD<TLoadL2Hint::NotAllocKeep>(dst, src);
```

Supported `TLoadL2Hint`:

| Enum | Value | A2/A3 | A5 |
| --- | --- | --- | --- |
| NormalFirstVictim | 0 | Default Allocation (No Effect) | Supported |
| NormalLastVictim | 1 | Default Allocation (No Effect) | Supported |
| NormalPersistent | 2 | Default Allocation (No Effect) | Supported |
| NotAllocKeep | 4 | Non-Allocation (GM address plus runtime `l2Cacheoffset`) | Supported |
| NotAllocClean | 5 | Non-Allocation (same as Keep) | Supported |
| NotAllocDrop | 6 | Non-Allocation (same as Keep) | Supported |

On A2/A3, there are actually only **two behaviors**:

1. **Default allocation**: `NormalFirstVictim` (0), `NormalLastVictim` (1), `NormalPersistent` (2): On A2/A3, these have no effect (no impact on different hints such as VLU / first/last/persist). They are not meaningful independent modes and all follow the default allocation path.
2. **Non-allocation**: `NotAllocKeep` (4), `NotAllocClean` (5), `NotAllocDrop` (6): These take effect by adding the runtime `l2Cacheoffset` to the GM address (on A2/A3, Keep/Clean/Drop behave the same).

On A5, all values in the table above are passed through to the DMA. The CPU / costmodel accepts this template and ignores it.

## Constraints

- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**:
    - `TileData::DType` must be one of the following: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`, `half`, `bfloat16_t`, `float`.
    - The destination tile position must be `TileType::Vec` or `TileType::Mat`.
    - `sizeof(TileData::DType) == sizeof(GlobalData::DType)`.
    - Runtime: all `src.GetShape(dim)` values and `dst.GetValidRow()/GetValidCol()` must be `> 0`.
    - `TileType::Vec` loads support only matching layouts: ND->ND, DN->DN, NZ->NZ.
    - `TileType::Mat` loads support: ND->ND, DN->DN, NZ->NZ, as well as ND->NZ and DN->ZN.
    - For ND->NZ: `GlobalData::staticShape[0..1] == 1` and `TileData::SFractalSize == 512`.
    - For DN->ZN: `GlobalData::staticShape[0..2] == 1` and `TileData::SFractalSize == 512`.
    - ND->NZ also supports `GlobalData::staticShape[2] != 1`. In this case, `Shape2` is the number of ND small matrices moved by a single instruction
      (The `ndNum` operand carried by the instruction), each small matrix has the shape `[Shape3, Shape4]`, and multiple small matrices are stacked along the tile row direction,
      so `dst.GetValidRow() == Shape2 * Shape3`. On this platform, `srcNdMatrixStride` and `dstNzMatrixStride`
      are 16-bit element counts, so in addition to `Shape2 * Shape3 <= TileData::Rows` and `1 <= Shape2 <= 65535`,
      it also requires `Stride2 <= 65535` and `Shape3 * (32 / sizeof(DType)) <= 65535`.
    - For `int64_t/uint64_t`, only ND->ND or DN->DN is supported.
    - Vec tile (UB path): `1 <= TileData::Rows <= 4095`.
    - Mat tile (L1 path): `1 <= TileData::Rows <= 16384`.
- **Implementation check (Ascend 950PR/Ascend 950DT)**:
    - `sizeof(TileData::DType)` must be `1`, `2`, `4`, or `8` bytes, and must match `sizeof(GlobalData::DType)`.
    - For `int64_t/uint64_t`, `TileData::PadVal` must be `PadValue::Null` or `PadValue::Zero`.
    - `TileType::Vec` loading requires one of the following layout pairs:
    - ND uses row-major + `SLayout::NoneBox` (ND->ND),
    - DN uses column-major + `SLayout::NoneBox` (DN->DN),
    - NZ uses `SLayout::RowMajor` (NZ->NZ).
    - For row-major ND->ND with compile-time known shapes, `TileData::ValidCol` must equal `GlobalData::staticShape[4]`, and `TileData::ValidRow` must equal the product of `GlobalData::staticShape[0..3]`.
    - `TileType::Mat` loads are also subject to `TLoadCubeCheck` constraints (for example, only specific ND/DN/NZ conversions and L1 size limits).
    - For ND->NZ and DN->NZ of `TileType::Mat`: `TileData::SFractalSize == 512`, `sizeof(TileData::DType) != 8`,
      and `GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1`.
    - ND->NZ additionally supports `GlobalData::staticShape[2] != 1` (including dynamic Shape2). `Shape2` indicates the number of ND small matrices available in the source,
      each small matrix has the shape `[Shape3, Shape4]`, and multiple small matrices are stacked along the tile row direction.
      At runtime: `1 <= Shape2 <= 65535`, `1 <= Shape3 <= 16384`, and
      `1 <= dst.GetValidRow() <= min(TileData::Rows, Shape2 * Shape3)`. The data type cannot be fp4.
    - DN->NZ requires `GlobalData::staticShape[2] == 1`.
    - For DN->ZN of `TileType::Mat`, the ZN tile uses `BLayout::RowMajor` and `SLayout::ColMajor`.
      When `GlobalData::staticShape[2] != 1` (including dynamic Shape2), the multiple-matrix path is selected.
      `Shape2` indicates the number of DN matrices available in the source, each matrix has the shape `[Shape3, Shape4]`, and they are merged along the tile column direction;
      `Stride2` indicates the distance between the start addresses of adjacent matrices, in elements.
      The rows of each matrix must be contiguous (`Stride3 == 1`); `Stride4` indicates the distance between the start addresses of adjacent columns, in elements.
    - The multiple-matrix DN->ZN path requires `GlobalData::staticShape[0..1] == 1`, `TileData::SFractalSize == 512`,
      `sizeof(TileData::DType)` of `1`, `2`, or `4` bytes (fp4/hif4 are not supported), and `TileData::Cols <= 65535`.
      At runtime: `1 <= Shape2 <= 65535`, `1 <= Shape4 <= 16384`,
      `1 <= dst.GetValidCol() <= min(TileData::Cols, Shape2 * Shape4)`, and
      `1 <= dst.GetValidRow() <= min(TileData::Rows, Shape3)`.
    - On A5, the source strides of the `TileType::Mat` multiple-matrix ND->NZ and DN->ZN paths retain 64-bit precision, which applies to both static and dynamic `Stride`.
      `Stride2` and the intra-matrix strides (`Stride3` for ND, `Stride4` for DN) are element counts of type `int64_t`.
      For the supported b8/b16/b32 types, the byte distance is `stride * sizeof(DType)`, computed with 64-bit arithmetic;
      element strides exceeding `INT32_MAX` and byte distances of 4 GiB or more are not truncated to 32 bits.
      Before passing `Stride`, use 64-bit operands to compute large stride expressions and ensure that the GM storage covers all accessed addresses.
    - `TileType::Mat` loads also handle mx-format loads, including `MX_A_ZZ/MX_A_ND/MX_A_DN` to ZZ (for scalarA) and `MX_B_NN/MX_B_ND/MX_B_DN` to NN (for scalarB).
    - For `MX_A_ZZ/MX_B_NN`: `(GlobalData::staticShape[3] == 16 || GlobalData::staticShape[3] == -1)` and `(GlobalData::staticShape[4] == 2 || GlobalData::staticShape[4] == -1)`.
    - For `MX_A_ND/MX_A_DN/MX_B_ND/MX_B_DN`: `(GlobalData::staticShape[0] == 1 || GlobalData::staticShape[0] == -1)` and `(GlobalData::staticShape[1] == 1 || GlobalData::staticShape[1] == -1)` and `(GlobalData::staticShape[4] == 2 || GlobalData::staticShape[4] == -1)`.
    - For scaleA, `dst.GetValidCol() % 2 == 0`.
    - For scaleB, `dst.GetValidRow() % 2 == 0`.

- **Valid region**:
    - The implementation uses `dst.GetValidRow()`/`dst.GetValidCol()` as the transfer size.
    - On A2/A3, a same-layout, block-aligned `TileType::Mat` load writes only the valid region. ND-to-NZ and DN-to-ZN loads (as well as the single-row/single-column Mat special paths) also zero-fill the tail of the last incomplete C0 block. Other data remains unchanged, including the data corresponding to other tile views that share the same underlying storage.
    - On A5, a same-layout `TileType::Mat` ND/DN load fills only the last incomplete 32B block according to `PadVal`; ND/DN-to-fractal-layout loads zero-fill the tail of the last incomplete C0 block. Complete 32B interval blocks and rows or columns that do not participate in the transfer remain unchanged.
    - On A5, an ND->NZ load with `GlobalData::staticShape[2] != 1` merges the first `dst.GetValidRow()` rows.
      First, one instruction moves `dst.GetValidRow() / Shape3` complete matrices; if `dst.GetValidRow() % Shape3` is non-zero,
      these tail rows are then moved separately; when there is no complete matrix, only the tail block is moved. Both moves maintain the NZ column-block stride of the entire tile.
      This also applies when Shape2 is a dynamic dimension whose runtime value is 1. For example, when `Shape3 = 3` and `dst.GetValidRow() = 17`,
      it moves 5 complete matrices and the first 2 rows of the 6th matrix, which requires `Shape2 >= 6`.
    - On A5, a DN->ZN load with `GlobalData::staticShape[2] != 1` merges the first `dst.GetValidCol()` columns.
      A single instruction first moves `dst.GetValidCol() / Shape4` complete matrices; if `dst.GetValidCol() % Shape4` is non-zero,
      these tail columns of the next matrix are then moved separately. When there is no complete matrix, only the tail block is moved.
      This also applies when Shape2 is a dynamic dimension whose runtime value is 1.
      When tail columns exist, let `n = dst.GetValidCol() / Shape4`; the tail matrix starts at `src.data() + n * Stride2`
      (the offset is in elements); the source offset retains 64-bit precision even if the corresponding byte distance reaches or exceeds 4 GiB.
      Both moves maintain the C0 block stride corresponding to `TileData::Cols`, in units of 32 bytes.
      Only the portion at the end of the valid rows that is less than one C0 block is zero-filled; columns that do not participate in the move and complete C0 blocks outside the valid rows retain their original values.
      When the compile-time Shape2 is 1, the single-matrix path is used: it moves `Shape4` columns and `dst.GetValidRow()` rows,
      so the source shape must describe the column range to be moved.
    - On A2/A3 and A5, when `PadVal` is non-null, the ND/DN load of `TileType::Vec` only fills the tail of each burst transfer that is less than 32B; complete 32B interval blocks and rows or columns that do not participate in the transfer remain unchanged. NZ load and `PadValue::Null` do not add padding.

## Examples

### Automatic

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_auto(__gm__ T* in) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gin(in);
  TileT t;
  TLOAD(t, gin);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
void example_manual(__gm__ T* in) {
  using TileT = Tile<TileType::Vec, T, 16, 16>;
  using GShape = Shape<1, 1, 1, 16, 16>;
  using GStride = BaseShape2D<T, 16, 16, Layout::ND>;
  using GTensor = GlobalTensor<T, GShape, GStride, Layout::ND>;

  GTensor gin(in);
  TileT t;
  TASSIGN(t, 0x1000);
  TLOAD(t, gin);
}
```

### A5 Multiple Matrices DN to ZN Load (Manual Mode)

The following Cube function loads the first 17 columns after merging, that is, 5 complete 3-column matrices and the first 2 columns of the 6th matrix.
The GM physical storage is `[8, 9, 64]` (matrix, column, row), and the DN view selects 3 columns and 35 rows of each matrix.
The following strides are all in elements.

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

AICORE void example_dn_to_zn(__gm__ int16_t* in) {
  using SrcShape = Shape<1, 1, 8, 35, 3>;
  using SrcStride = pto::Stride<4608, 4608, 576, 1, 64>;
  using SrcGlobal = GlobalTensor<int16_t, SrcShape, SrcStride, Layout::DN>;
  using MatTile = Tile<TileType::Mat, int16_t, 64, 32, BLayout::RowMajor,
                       35, 17, SLayout::ColMajor, 512>;

  SrcGlobal src(in);
  MatTile dst;
  TASSIGN(dst, 0x1000);
  TLOAD(dst, src);
}
```

For `0 <= r < 35` and `0 <= c < 17`, the result is
`dst[r, c] = in[(c / 3) * 576 + (c % 3) * 64 + r]`.

### A5 Dynamic Large Stride DN to ZN Load (Manual Mode)

The following Cube function loads the first 3 columns after coalescing, that is, all 2 columns of the first matrix and column 1 of the second matrix. `matrixStride` is passed in at runtime and indicates the distance between the start addresses of adjacent matrices, in `uint16_t` elements. For example, `(int64_t{1} << 31) + 128` elements correspond to `4294967552` bytes (4 GiB + 256 bytes). The GM allocation must cover all accessed elements, including the 32 rows at the start of the second matrix. When the example stride above is used, the required storage span starting from `in` is `(matrixStride + 32) * sizeof(uint16_t)` bytes. The `-1` in `SrcStride` only sets `Stride2` to a dynamic value; `Shape2` is still the compile-time matrix count 2.

```cpp
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

AICORE void example_dn_to_zn_large_stride(__gm__ uint16_t* in, int64_t matrixStride) {
  using SrcShape = Shape<1, 1, 2, 32, 2>;
  using SrcStride = pto::Stride<1, 1, -1, 1, 64>;
  using SrcGlobal = GlobalTensor<uint16_t, SrcShape, SrcStride, Layout::DN>;
  using MatTile = Tile<TileType::Mat, uint16_t, 64, 64, BLayout::RowMajor,
                       32, 3, SLayout::ColMajor, 512>;

  SrcGlobal src(in, SrcShape{}, SrcStride(1, 1, matrixStride, 1, 64));
  MatTile dst;
  TASSIGN(dst, 0x1000);
  TLOAD(dst, src);
}
```

For `0 <= r < 32` and `0 <= c < 3`, the result is `dst[r, c] = in[(c / 2) * matrixStride + (c % 2) * 64 + r]`. Therefore, the last loaded column comes from `in[matrixStride + r]`.
