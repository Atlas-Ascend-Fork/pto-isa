# TFILLPAD

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:57:36.791Z pushedAt=2026-11-02T03:00:52.181Z nl=2 -->

## Instruction Diagram

![TFILLPAD tile operation](../figures/isa/TFILLPAD.svg)

## Introduction

Copies a tile and fills the elements outside the valid region with a compile-time fill value.

Copies the source tile into the destination tile and fills the remaining (padding) elements with a compile-time fill value selected by `TileDataDst::PadVal` (such as `PadValue::Min`/`PadValue::Max`).

This operation is commonly used to deterministically materialize a specific value outside the runtime valid region, allowing subsequent operations to compute on the complete static tile shape.

## Mathematical Semantics

Assume `VR = src.GetValidRow()` and `VC = src.GetValidCol()`. For each destination element `(i, j)`:

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src}_{i,j} & \text{if } i < VR \text{ and } j < VC \\
\mathrm{pad}       & \text{otherwise}
\end{cases}
$$

`pad` is determined by `TileDataDst::PadVal` and the element type (for example, `+inf/-inf` when supported by the floating-point type, otherwise `std::numeric_limits<T>::max()/min()`).

## C++ Built-in APIs

Implemented in the backend header file introduced by `include/pto/common/pto_instr_impl.hpp`:

```cpp
template <typename TileData, PadValue PadVal = PadValue::Zero, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(TileData &dst, TileData &src, WaitEvents &... events);

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(DstTileData &dst, SrcTileData &src, WaitEvents &... events);
```

## Constraints

- `TileDataDst::PadVal != PadValue::Null` (Vec type overload).
- `sizeof(TileDataDst::DType) == sizeof(TileDataSrc::DType)` and the element size must be `1`, `2`, or `4` bytes.
- `TFILLPAD`: `TileDataDst::Rows/Cols` must match `TileDataSrc::Rows/Cols`.
- `TFILLPAD_EXPAND`: `TileDataDst::Rows >= TileDataSrc::Rows` and `TileDataDst::Cols >= TileDataSrc::Cols`.
- `TFILLPAD(TileData &dst, TileData &src)` (Mat type overload): when `TileData::TileType` is `Mat`, the layout must satisfy `!TileData::isRowMajor && TileData::SLayout::RowMajor`, and `PadVal` must be `PadValue::Zero` or `PadValue::Null`. This Mat overload and the first Vec overload (`PadVal != PadValue::Null`) belong to different SFINAE overloads and do not conflict.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example1() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox, TileConfig::fractalABSize, PadValue::Min>;

  SrcT src;
  DstT dst;
  TFILLPAD(dst, src);
}

void example2() {
  using TileMatData = Tile<TileType::Mat, float, 16, 256, BLayout::ColMajor, 1, 224, SLayout::RowMajor, 512>;

  TileMatData matTile;
  TFILLPAD(matTile, matTile);
}
```
