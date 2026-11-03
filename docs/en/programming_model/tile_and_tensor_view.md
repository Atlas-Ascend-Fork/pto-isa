# Tile and Tensor View

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T06:28:25.599Z pushedAt=2026-11-02T03:00:52.271Z nl=2 -->

## Overview

PTO explicitly separates the "global tensor object" and the "on-chip tile object" into two models. The former describes the logical view of global data, while the latter describes the actual working object for local computation and data movement. Many compile-time checks and automatic analyses are built on this boundary.

## Four-Layer Data Objects

From the perspective of the upper-level programming model, when writing a PTO kernel, users typically encounter the following four layers of objects in sequence:

1. Global address entry
2. Global tensor view
3. Tensor local partition
4. Local tile object

These four layers of objects represent different expressions of the same data at different stages, rather than four completely independent sets of data.

### Global Address Entry

In the upper-level model, it represents a global address entry obtained by the kernel.

- C++ representation: `__gm__` pointer (kernel entry parameter, such as `__gm__ float* in`)
- IR representation: `!pto.ptr<T>`

It only expresses "the global address of a certain type of element" and does not carry:

- Shape
- Stride
- Slice
- Tile layout

Therefore, it is more like the raw input or output handle of the kernel. A single `ptr` alone is not enough to express "what kind of tensor this block is."

### Global Tensor View

In the upper-level model, it means "interpreting a segment of global address in a tensor manner."

- C++ representation: `pto::GlobalTensor<Element, Shape, Stride, Layout>`
- IR representation: `!pto.tensor_view<...>`

`tensor_view` is used to describe:

- The logical shape of the global tensor
- The stride/layout interpretation corresponding to the shape

From the user's perspective, this step is saying:

- What tensor shape this segment of global memory should be interpreted as
- How each dimension should be addressed

It is still a global object, not yet a local computation object.

### Tensor Local Partition

In the upper-level model, it represents "taking out the block to be processed this time from the entire tensor."

- C++ representation: also `GlobalTensor`, constructed through pointer offset and sub-shape (there is no separate partition type on the C++ side; layers ② and ③ share the same type.)
- IR representation: `!pto.partition_tensor_view<...>`

`partition_tensor_view` further narrows a global tensor view to a logical partition. It expresses:

- A certain local sub-region
- The shape of this sub-region
- The logical relationship between it and the upstream global tensor

From the perspective of the programming model, this step typically corresponds to the following:

- The current block processes only one block of the entire tensor.
- The current loop iteration processes only a certain sub-block.
- A single `TLOAD` / `TSTORE` corresponds to only one of the local regions.

In other words, it answers the question "which block is to be processed this time."

### Local Tile Object

In the upper-level model, it represents a local data block that has entered on-chip memory and can directly participate in computation.

- C++ representation: `pto::Tile<TileType::Loc, Element, Rows, Cols, ...>`
- IR representation: `!pto.tile_buf<loc=..., dtype=..., rows=..., cols=..., ...>`

`tile_buf` is a local tile object that explicitly carries:

- Address space
- Element type
- Physical size
- Valid size
- Layout/pad/fractal and other configurations

From the user's perspective, this is no longer "a view of the global tensor". Instead:

- It has already entered on-chip memory.
- It has already been placed at a specific local location.
- It can directly participate in data movement, element-wise, matrix, or reduction computations.

Therefore, it is the most direct working object in a PTO kernel.

## Typical Usage Path

The most common path is as follows:

| Step | C++ | IR |
| --- | --- | --- |
| ① Global address | `__gm__` pointer (kernel parameter) | `ptr` |
| ② Construct global view | Construct `GlobalTensor<...>` | `make_tensor_view` |
| ③ Slice out partition | Construct `GlobalTensor<...>` with offset | `partition_view` |
| ④ Move to on-chip | `TLOAD` | `tload` / `tprefetch` |
| ⑤ Local computation | On `Tile<...>` | On `tile_buf` |
| ⑥ Write back | `TSTORE` | `tstore` |

C++ form:

```cpp
using GT = pto::GlobalTensor<float,
                             pto::Shape<1,1,1,32,32>,
                             pto::Stride<1,1,1,32,1>>;
using TileT = pto::Tile<pto::TileType::Vec, float, 32, 32>;

GT gin(in);        // ①→② Construct the global view from the __gm__ pointer.
TileT t;           // ④ Local tile object.
TLOAD(t, gin);     // ②→④ Move from global to on-chip.
// ... Perform local computation on t ...
TSTORE(gout, t);   // ④→② Write back.
```

IR form:

```mlir
%tv = pto.make_tensor_view %arg0,
  shape = [%c32, %c32],
  strides = [%c32, %c1]
  : !pto.tensor_view<?x?xf32>

%pv = pto.partition_view %tv,
  offsets = [%c0, %c0],
  sizes = [%c32, %c32]
  : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
```

This path can be intuitively understood as follows:

- First, specify "what tensor the entire global memory represents".
- Then, specify "which block of it to take this time".
- Finally, move this block on-chip for use as a tile.

## Why Distinguish View and Tile

Without distinguishing view from tile, the upper-level model becomes ambiguous:

- Whether the data is still in GM or has already been moved on-chip
- Whether the current operation is "selecting a block of data" or "actually moving data"
- Whether the current object describes a logical tensor region or a local compute buffer
- Whether subsequent operations should take a global view or a local tile

PTO preserves these stage distinctions stably through these two object models, which can be roughly understood as:

- View model: focuses on "how global data is understood and partitioned"
- Tile model: focuses on "how on-chip data is placed and used"

For users writing kernels, this distinction is usually more important than the type details.

## Programming Implications

### First Determining Which Layer the Data Is In

When writing PTO ISA, the first thing to determine is:

- Whether what you currently have is a global address
- A global tensor view
- A tensor partition
- Or a local tile

The legal operations differ across these layers.

### Expressing Global Tiling and Local Computation Separately

Global tiling is usually accomplished by constructing a global tensor view (optionally with an offset); local computation usually occurs on local tile objects. Writing these two steps separately makes the program clearer.

### Keeping the Data Flow Path Intuitive

For most kernels, the most stable approach is as follows:

1. Construct `tensor_view` from `ptr`.
2. Slice `partition_tensor_view` from `tensor_view`.
3. Use `TLOAD` to move data to the local tile.
4. Perform local computation on `tile_buf`.
5. Write back with `TSTORE`.

The clearer this path is kept, the more straightforward subsequent reading, debugging, and maintenance will be.
