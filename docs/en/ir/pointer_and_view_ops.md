# Pointer and View Operations

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:46:48.740Z pushedAt=2026-11-02T03:00:52.136Z nl=2 -->

This chapter describes the pointer and view operations in the PTO ISA. These operations cover pointer arithmetic, type conversion, tensor view construction, tiling subview creation, tile type reinterpretation, effective dimension read/write, and buffer synchronization token management. Most of them are pure metadata operations that involve no data movement and occupy no hardware pipe. These operations are critical in dynamic shape processing, memory remapping, and multi-level storage access.

---

## Contents

- [`pto.addptr`: Pointer Addition](#ptoaddptr-pointer-addition)
- [`pto.castptr`: Pointer Field Type Conversion](#ptocastptr-pointer-field-type-conversion)
- [`pto.make_tensor_view`: Constructing a Tensor View](#ptomake_tensor_view-constructing-a-tensor-view)
- [`pto.get_tensor_view_dim`: Obtaining Tensor View Dimensions](#ptoget_tensor_view_dim-obtaining-tensor-view-dimensions)
- [`pto.partition_view`: Splitting a View](#ptopartition_view-splitting-a-view)
- [`pto.subview`: Creating a Tiling Subview](#ptosubview-creating-a-tiling-subview)
- [`pto.bitcast`: Tile Element Type Reinterpretation](#ptobitcast-tile-element-type-reinterpretation)
- [`pto.get_validshape`: Reading Runtime Effective Dimensions](#ptoget_validshape-reading-runtime-effective-dimensions)
- [`pto.set_validshape`: Setting Runtime Effective Dimensions](#ptoset_validshape-setting-runtime-effective-dimensions)
- [`pto.get_buf`: Obtaining Buffer Synchronization Tokens](#ptoget_buf-obtaining-buffer-synchronization-tokens)
- [`pto.rls_buf`: Releasing Buffer Synchronization Tokens](#ptorls_buf-releasing-buffer-synchronization-tokens)
- [`pto.tget_scale_addr`: Binding a Scaling Tile View](#ptotget_scale_addr-binding-a-scaling-tile-view)

---

## Operation Details

### `pto.addptr`: Pointer Addition

```mlir
pto.addptr <ptr>, <offset> : !pto.ptr<elementType> -> !pto.ptr<elementType>
```

**Semantics:**

```text
result = ptr + offset  // offset is in elements, not Bytes
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `ptr` | `!pto.ptr<elementType>` | Base pointer |
| `offset` | `index` | Element offset (not byte offset) |

**Returns:** `!pto.ptr<elementType>`: a pointer of the same type as the input pointer.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The result type must match the input pointer type.
  - Pure operation (no side effects).

**Example:**

```mlir
%c2 = pto.constant 2 : index
%ptr_off = pto.addptr %base, %c2 : !pto.ptr<f32> -> !pto.ptr<f32>
```

---

### `pto.castptr`: Pointer Field Type Conversion

```mlir
%result = pto.castptr <input> : <input_type> -> <result_type>
```

**Semantics:**

```text
Pointer -> i64: Return the byte address of the pointer.
i64 -> Pointer: Construct a pointer with the specified element type and address space at the given byte address.
Pointer -> Pointer: Keep the address and address space, reinterpret the pointer element type, and do not convert the pointed-to data.
memref -> Pointer: Extract the aligned base address of the memref without allocation or copying.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `input` | `i64`, `!pto.ptr<T, space>`, or `memref` | Source byte address, pointer, or memory descriptor |

**Returns:** An `i64` byte address, or a `!pto.ptr<T, space>` pointer. When the pointer address space is omitted, it defaults to GM.

**Constraints:**

- Supports conversion between `i64` and pointer, reinterpretation of the pointer element type within the same address space, and conversion from `memref` to pointer.
- The integer address must be a signless `i64`; `index`, `i32`, or `ui64` cannot be used.
- Pointers cannot be converted across address spaces. For example, a GM pointer cannot be directly converted to a UB pointer.
- When extracting a pointer from a `memref` with a PTO address space, the result must preserve that address space; conversions from `memref` to integer or from integer to integer are not supported.
- The pointer element type must be a supported scalar element type; the conversion itself does not guarantee that the address is valid, aligned, or has sufficient capacity, as these conditions are determined by subsequent access requirements.

**Example:**

```mlir
// Obtain the address from a valid GM pointer, then restore a pointer of the same type.
%addr = pto.castptr %ptr : !pto.ptr<f32> -> i64
%restored = pto.castptr %addr : i64 -> !pto.ptr<f32>
// Keep the address and access the same storage in bytes.
%bytes = pto.castptr %restored : !pto.ptr<f32> -> !pto.ptr<i8>
```

---

### `pto.make_tensor_view`: Constructing a Tensor View

```mlir
pto.make_tensor_view <ptr>, shape = [<d0>, <d1>, ...], strides = [<s0>, <s1>, ...]
                     : !pto.tensor_view<...>
```

**Semantics:**

```text
result = tensor_view(ptr, shape, strides, layout)
// Construct a global tensor view from a pointer without allocation or data movement.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `ptr` | `!pto.ptr<elementType>` | Source pointer |
| `shape` | Mutable `index` | Dynamic shape dimensions |
| `strides` | Mutable `index` | Dynamic strides in elements |

**Returns:** `!pto.tensor_view<...>`: The constructed tensor view.

**Attributes:**

- `layout`: Optional layout attribute (ND, DN, or NZ).

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `ptr` must be `!pto.ptr<...>`, with its element type matching the result.
  - The operand counts of `shape` and `strides` must match the tensor view rank.

**Example:**

```mlir
%m = pto.constant 128 : index
%n = pto.constant 256 : index
%s0 = pto.constant 256 : index
%s1 = pto.constant 1 : index
%tv = pto.make_tensor_view %ptr, shape = [%m, %n], strides = [%s0, %s1]
    : !pto.tensor_view<?x?xf32>
```

---

### `pto.get_tensor_view_dim`: Obtaining Tensor View Dimensions

```mlir
pto.get_tensor_view_dim <tensor_view>, <dim_index>
                        : !pto.tensor_view<...> -> index
```

**Semantics:**

```text
result = get_dim(tensor_view, dim_index)
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `tensor_view` | `!pto.tensor_view<...>` | Source tensor view |
| `dim_index` | `index` | 0-based dimension index |

**Returns:** `index`, size of the specified dimension.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - Pure operation (no side effects).

**Example:**

```mlir
%h = pto.get_tensor_view_dim %tv, %c0 : !pto.tensor_view<?x?xf32> -> index
%w = pto.get_tensor_view_dim %tv, %c1 : !pto.tensor_view<?x?xf32> -> index
```

---

### `pto.partition_view`: Splitting a View

```mlir
pto.partition_view <source>, offsets = [<o0>, <o1>, ...], sizes = [<s0>, <s1>, ...]
                   : !pto.tensor_view<...> -> !pto.partition_tensor_view<...>
```

**Semantics:**

```text
result = source[offsets : offsets + sizes]
// Create a logical window on the tensor view.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `source` | `!pto.tensor_view<...>` | Source tensor view |
| `offsets` | Mutable `index` | Window start offset |
| `sizes` | Mutable `index` | Window size |

**Returns:** `!pto.partition_tensor_view<...>`: the split tensor view.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The count of `offsets` and `sizes` must match `source`.
  - The operation is formally declared `Pure` in ODS; it only creates a logical view and does not read or write the underlying data.

**Example:**

```mlir
%pv = pto.partition_view %tv, offsets = [%off0, %off1], sizes = [%s0, %s1]
    : !pto.tensor_view<1024x512xf16> -> !pto.partition_tensor_view<16x16xf16>
```

---

### `pto.subview`: Creating a Tiling Subview

```mlir
pto.subview <source>[<i>, <j>] sizes [<rows>, <cols>] (valid [<vr>, <vc>])?
            : <source_type> -> <result_type>
```

**Semantics:**

```text
result = source[offsets]
result.shape = sizes
result.valid = clip(explicit_valid_or_sizes, sizes)
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `source` | `!pto.tile_buf<...>` | Source tiling buffer |
| `offsets` | Mutable `index` | Runtime dynamic offsets [i, j] |
| `valid_row` | `index` (optional) | Number of valid rows |
| `valid_col` | `index` (optional) | Number of valid columns |

**Returns:** `!pto.tile_buf<...>`: Subview tiling buffer.

**Attributes:**

- `sizes`: Static shape array (length 2).

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - For boxed layout (`slayout != none_box`): `sizes` must be a multiple of the inner boxed shape; `offsets`, when constant, must be a multiple of the inner boxed shape.
  - For boxed row-major: the subview retains the full source column range, and the column offset must be 0.
  - For boxed column-major: the subview retains the full source row range, and the row offset must be 0.
  - `valid_row`/`valid_col` must be present or absent together.

**Example:**

```mlir
%i = pto.constant 0 : index
%j = pto.constant 0 : index
%sub = pto.subview %src[%i, %j] sizes [32, 32]
    : !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=64,
                    v_row=64, v_col=64, blayout=row_major,
                    slayout=none_box, fractal=512, pad=0>
    -> !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32,
                     v_row=32, v_col=32, blayout=row_major,
                     slayout=none_box, fractal=512, pad=0>

// With effective dimensions.
%sub2 = pto.subview %src[%i, %j] sizes [32, 32] valid [%vr, %vc]
    : !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=64,
                    v_row=64, v_col=64, blayout=row_major,
                    slayout=none_box, fractal=512, pad=0>
    -> !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32,
                     v_row=?, v_col=?, blayout=row_major,
                     slayout=none_box, fractal=512, pad=0>
```

---

### `pto.bitcast`: Tile Element Type Reinterpretation

```mlir
pto.bitcast <src> : <src_type> -> <result_type>
```

**Semantics:**

```text
result = reinterpret_dtype(src)
// Return a tile buffer view that shares the same underlying storage as src but has a different element type.
// Only metadata/config is rewritten, with no data movement involved.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `!pto.tile_buf<...>` | Source tile buffer. |

**Returns:** `!pto.tile_buf<...>`: a tile buffer view with a different element type, sharing storage with the source.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The source and result must have different element types (use `pto.treshape` if only the shape/config needs to change).
  - The source and result must have the same shape, validShape, memory space, and tile config.
  - The shape must be statically known.
  - The total number of bytes required by the result must not exceed the total number of bytes of the source.
  - A pure operation (`Pure` + `ViewLikeOpInterface`) that does not allocate storage.

**Example:**

```mlir
%casted = pto.bitcast %src
    : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                    v_row=16, v_col=16, blayout=row_major,
                    slayout=none_box, fractal=512, pad=0>
    -> !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                     v_row=16, v_col=16, blayout=row_major,
                     slayout=none_box, fractal=512, pad=0>
```

---

### `pto.get_validshape`: Reading Runtime Effective Dimensions

```mlir
pto.get_validshape <source> : <source_type>
```

**Semantics:**

```text
(valid_row, valid_col) = read_valid_metadata(source)
// Read the current runtime valid_row/valid_col metadata on the tile_buf handle.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `source` | `!pto.tile_buf<...>` | Source tile handle |

**Returns:** `(index, index)`: the current number of valid rows and valid columns.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The source must be a rank-2 tile_buf.
  - The **validShape** of the source must be rank-2.

**Example:**

```mlir
%vr, %vc = pto.get_validshape %tile
    : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                    v_row=?, v_col=?, blayout=row_major,
                    slayout=none_box, fractal=512, pad=0>
```

---

### `pto.set_validshape`: Setting Runtime Effective Dimensions

```mlir
pto.set_validshape <source>, <valid_row>, <valid_col> : <source_type>
```

**Semantics:**

```text
source.valid_row = valid_row
source.valid_col = valid_col
// Update the runtime effective dimension metadata on the tile_buf handle in place.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `source` | `!pto.tile_buf<...>` | Target tile handle |
| `valid_row` | `index` | New valid row count |
| `valid_col` | `index` | New valid column count |

**Returns:** None. Modifies the metadata of `source` in place.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The source must be a rank-2 tile_buf.
  - Both valid_row and valid_col of the source tile_buf must be dynamic (`?`).
  - The source must be a locally bound tile handle (such as `alloc_tile` or a view derived from it); function parameters/return values are not supported.
  - `valid_row` must not be negative and must not exceed the static size of the shape dimension.
  - `valid_col` must not be negative and must not exceed the static size of the shape dimension.

**Example:**

```mlir
pto.set_validshape %tile, %new_vr, %new_vc
    : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                    v_row=?, v_col=?, blayout=row_major,
                    slayout=none_box, fractal=512, pad=0>
```

---

### `pto.get_buf`: Obtaining Buffer Synchronization Tokens

```mlir
pto.get_buf [<op_type>, <buf_id>]
```

**Semantics:**

```text
acquire_buffer_token(op_type, buf_id, mode)
// Obtain the synchronization token for the specified pipe operation type and buffer ID.
// Operations mapped to the same pipe and protected by the same buf_id execute in program order.
```

**Parameters:** No operands.

**Returns:** None.

**Attributes:**

- `op_type`: Pipe event type (`pipe_event_type` or `sync_op_type`), mapped to a specific hardware pipe.
- `buf_id`: Buffer ID (`i32`), in the range [0, 31].
- `mode`: Mode (`i32`), default value `0`.

**Constraints:**

- **Implementation check (Ascend 950PR&950DT products)**
  - `op_type` must map to a specific pipe (cannot be `PIPE_ALL` or `PIPE_UNASSIGNED`).
  - `buf_id` must be in the range [0, 31].
  - `mode` must be a non-negative integer.

**Example:**

```mlir
pto.get_buf[#pto.pipe_event_type<TLOAD>, 0]
pto.get_buf[#pto.pipe_event_type<TVEC>, 1]
```

---

### `pto.rls_buf`: Releasing Buffer Synchronization Tokens

```mlir
pto.rls_buf [<op_type>, <buf_id>]
```

**Semantics:**

```text
release_buffer_token(op_type, buf_id, mode)
// Release the buffer synchronization token previously obtained through pto.get_buf.
```

**Parameters:** No operands.

**Returns:** None.

**Attributes:**

- `op_type`: Pipe event type (`pipe_event_type` or `sync_op_type`), mapped to a specific hardware pipe.
- `buf_id`: Buffer ID (`i32`), in the range [0, 31].
- `mode`: Mode (`i32`), with a default value of `0`.

**Constraints:**

- **Implementation check (Ascend 950PR&950DT products)**
  - `op_type` must be mapped to a specific pipe (it cannot be `PIPE_ALL` or `PIPE_UNASSIGNED`).
  - `buf_id` must be in the range [0, 31].
  - `mode` must be a non-negative integer.

**Example:**

```mlir
pto.rls_buf[#pto.pipe_event_type<TLOAD>, 0]
pto.rls_buf[#pto.pipe_event_type<TVEC>, 1]
```

---

### `pto.tget_scale_addr`: Binding a Scaling Tile View

```mlir
pto.tget_scale_addr ins(<src> : <src_type>)
                    outs(<dst> : <dst_type>)
```

**Semantics:**

```text
dst = scale_view_of(src)
// Bind the scaling tile view to the scaling address of the source tile for MX (mixed precision) floating-point support.
// Pure address binding operation with no data movement.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `!pto.tile_buf<...>` | Source tile buffer |
| `dst` | `!pto.tile_buf<loc=scaling, ...>` | Destination scaling tile view |

**Returns:** None. Binds `dst` to the scaling address of `src` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - This operation is not supported.

- **Implementation check (Ascend 950PR&950DT products)**
  - `src` is located in `left` or `right`, and `dst` is located in `scaling`; both the physical and invalid shapes are rank-2.
  - For the left source, let K be the number of effective columns of the source: the destination physical shape is `[src.rows,ceil(K/32)]`, and the invalid shape is `[src.v_row,ceil(K/32)]`.
  - For the right source, let K be the source valid row count: the destination physical shape is `[ceil(K/32),src.cols]`, and the valid shape is `[ceil(K/32),src.v_col]`.
  - The destination only binds the scale address of the source and does not copy or generate scale data; subsequent MX operations further constrain the layout of this view.

**Example:**

```mlir
// Ascend 950PR&950DT products: bind the f8 source to its associated scale tile.
pto.tget_scale_addr
    ins(%src : !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=1, cols=128,
                             v_row=1, v_col=128, blayout=col_major,
                             slayout=row_major, fractal=512, pad=0>)
    outs(%scale : !pto.tile_buf<loc=scaling, dtype=f16, rows=1, cols=4,
                                v_row=1, v_col=4, blayout=row_major,
                                slayout=row_major, fractal=32, pad=0>)
```
