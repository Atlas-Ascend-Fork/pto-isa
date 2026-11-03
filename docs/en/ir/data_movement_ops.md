# Data Movement Operations

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:24:28.161Z pushedAt=2026-11-02T03:00:52.131Z nl=2 -->

This section describes data transfer between global memory and local buffers, data movement between local memory domains, and scalar reads and writes. The tile movement form adopts the Destination-Passing Style (DPS), writing to a pre-allocated destination; `pto.load` returns the read value, and `pto.tprefetch_async` returns an asynchronous event.

Data movement operations typically involve the following scenarios:

- **GM to local tile buffer transfer**: Loads a global memory partition view into a local tile buffer through `pto.tload` and `pto.tprefetch`.
- **Asynchronous prefetch**: Uses `pto.tprefetch_async` to initiate an SDMA-driven asynchronous prefetch and returns a synchronization event.
- **Local buffer store back to global memory**: Writes the tile buffer back to a global memory partition through `pto.tstore`, supporting atomic operations and quantization conversion.
- **Accumulator store with scale factor**: Uses the `fp` parameter of `pto.tstore` to transform accumulator data with a scale tile before storing it to global memory.
- **Gather/scatter operations**: Performs non-contiguous data movement between global memory and local tiles based on indices through `pto.mgather` and `pto.mscatter`; the basic form supports Atlas A3 products/Ascend 950PR&950DT products.
- **Inter-domain data movement in local memory**: Uses `pto.tmov` to transfer data between local storage domains such as `mat`, `vec`, `acc`, and `bias`.
- **Tile transpose**: Performs matrix transpose on the tile buffer through `pto.ttrans`.
- **Scalar read/write**: reads and writes elements in the specified pointer address space through `pto.load` and `pto.store`.

---

## Contents

- [`pto.tload`: Physical DMA Load](#ptotload-physical-dma-load)
- [`pto.tprefetch`: Prefetch Load](#ptotprefetch-prefetch-load)
- [`pto.tprefetch_async`: Asynchronous Prefetch](#ptotprefetch_async-asynchronous-prefetch)
- [`pto.tstore`: Tile Buffer Storage](#ptotstore-tile-buffer-storage)
- [`pto.tstore` `fp` Form](#ptotstore-fp-form)
- [`pto.mgather`: Gather Load](#ptomgather-gather-load)
- [`pto.mscatter`: Scatter Storage](#ptomscatter-scatter-storage)
- [`pto.load`: Scalar Load](#ptoload-scalar-load)
- [`pto.store`: Scalar Storage](#ptostore-scalar-storage)
- [`pto.tmov`: Local Memory Inter-Domain Data Movement](#ptotmov-local-memory-inter-domain-data-movement)
- [`pto.ttrans`: Tile Transpose](#ptottrans-tile-transpose)
- [`pto.tmov` `fp` Form](#ptotmov-fp-form)

---

## Operation Details

### `pto.tload`: Physical DMA Load

```mlir
pto.tload ins(<src> : <src_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j) in tile valid region:
    dst[i, j] = src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.partition_tensor_view<...>` | Source global memory partition view. |
| `dst` | `pto.tile_buf` | Destination local tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be one of the following: `i8`, `i16`, `i32`, `i64`, `f16`, `bf16`, `f32`.
  - `dst` must use `loc=vec` or `loc=mat`.
  - The element types of `dst` and `src` must have the same bit width.
  - Runtime constraints: the extent of all source partitions must be positive; the valid region of `dst` must be non-negative.

- **Implementation check (Ascend 950PR&950DT products)**
  - The `src` and `dst` element types must be one of the following: `i8`, `i16`, `i32`, `i64`, `f16`, `bf16`, `f32`, `f8E4M3*`, `f8E5M2*`, `!pto.hif8`, `!pto.f4E1M2x2`, `!pto.f4E2M1x2`.
  - The `dst` element size must be 1, 2, 4, or 8 bytes, and must match `src`.
  - For the `i64` type, the padding of `dst` must be null or zero.

**Hardware pipe:** PIPE_MTE2 (DMA transfer from GM to UB)

**Example:**

```mlir
pto.tload ins(%pv : !pto.partition_tensor_view<16x16xf16>)
          outs(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### `pto.tprefetch`: Prefetch Load

```mlir
pto.tprefetch ins(<src> : <src_type>)
              outs(<dst> : <dst_type>)
```

**Semantics:**

```text
TPREFETCH(dst, src)
For each element (i, j) in tile valid region:
    dst[i, j] = src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.partition_tensor_view<...>` | Source global memory view. |
| `dst` | `pto.tile_buf` | Destination local buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `dst` must use `loc=vec` or `loc=mat`.
  - The static source extent must be positive; the valid extent of `dst` must be non-negative.
  - The element types of `src` and `dst` must have the same element size.
  - Low-precision types (`f8*`, `!pto.hif8`, `!pto.f4*`) are supported only on Ascend 950PR&950DT products.

**Hardware pipe:** PIPE_MTE2 (GM-to-local-tile prefetch)

**Example:**

```mlir
pto.tprefetch ins(%pv : !pto.partition_tensor_view<16x16xf16>)
              outs(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                     v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tprefetch_async`: Asynchronous Prefetch

```mlir
%event = pto.tprefetch_async ins(<src>, <ctx> : <src_type>, !pto.prefetch_async_context)
                             -> !pto.async_event
```

**Semantics:**

```text
%event = pto.tprefetch_async(%src, %ctx)
Starts an SDMA-based asynchronous prefetch from global memory to the cache and returns an asynchronous event.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tensor_view` / `pto.partition_tensor_view` | Source global memory view, which must be a flat contiguous logical 1D view. |
| `ctx` | `pto.prefetch_async_context` | Prefetch asynchronous context, which must be valid. |

**Returns:** `!pto.async_event`, an event handle used to synchronize the asynchronous operation.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src` must be a flat contiguous logical 1D GM view.
  - `ctx` must be a valid `prefetch_async_context`.

**Execution mode:** This operation generates a `TPREFETCH_ASYNC` call, and the asynchronous transfer is completed by the CANN SDMA runtime. `SDMA` is the transfer engine here, not a PTO `PIPE_*` scheduling enum.

**Example:**

```mlir
%ctx = pto.make_prefetch_async_context(%workspace : !pto.ptr<i8>) -> !pto.prefetch_async_context
%event = pto.tprefetch_async(%src, %ctx : !pto.partition_tensor_view<128xf32>,
                             !pto.prefetch_async_context) -> !pto.async_event
```

---

### `pto.tstore`: Tile Buffer Storage

```mlir
pto.tstore ins(<src> : <src_type>)
           outs(<dst> : <dst_type>)
           {attributes}

// Supports the optional preQuantScalar parameter:
pto.tstore ins(<src> : <src_type>, <preQuantScalar> : i64)
           outs(<dst> : <dst_type>)
           {attributes}
```

**Semantics:**

```text
For each element (i, j) in tile valid region:
    dst[i, j] = src[i, j]

(Supports optional atomic operations, ReLU pre-processing, and quantization conversion)
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer, located at `vec`, `mat`, or `acc`. |
| `dst` | `pto.partition_tensor_view<...>` | Destination global memory partition view. |
| `preQuantScalar` | `i64` (optional) | Scalar value before quantization, mutually exclusive with fp. |
| `fp` | `!pto.tile_buf<loc=scaling, ...>` (optional) | Per-column scale/quantization parameter, see the fp form in this section. |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `stPhase`: `PTO_STPhaseAttr` storage phase attribute, default value is
  `#pto<st_phase unspecified>`.

  | Text Attribute Value | Meaning |
  | --- | --- |
  | `#pto<st_phase unspecified>` | Unspecified storage phase |
  | `#pto<st_phase partial>` | Partial storage (accumulate intermediate values) |
  | `#pto<st_phase final>` | Final storage (complete accumulation) |

- `atomicType`: Atomic operation type. The default value is `atomic_none`.
  - `#pto<atomic_type atomic_none>`: No atomic operation
  - `#pto<atomic_type atomic_add>`: Atomic addition (dst += src)

- `reluPreMode`: ReLU pre-processing mode. The default value is `no_relu`.
  - `#pto<relu_pre_mode no_relu>`: No ReLU processing
  - `#pto<relu_pre_mode normal_relu>`: Apply standard ReLU (max(0, x))

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src.loc` must be `vec`, `mat`, or `acc`.
  - For `loc=vec` or `loc=mat`: `preQuantScalar` is not allowed; the `src` element type must be `i8`, `i16`, `i32`, `i64`, `f16`, `bf16`, or `f32`; the bit width must match.
  - For `loc=acc`: `src` must be `i32` or `f32`.
  - Without `preQuantScalar`: `dst` is `i32`, `f32`, `f16`, or `bf16`.
  - With `preQuantScalar`: `src=i32` → `dst=i8` or `f16`; `src=f32` → `dst=i8`.
  - Static column count `1 <= cols <= 4095`.

- **Implementation check (Ascend 950PR&950DT products)**
  - `src.loc` must be `vec` or `acc`.
  - For `loc=vec`: `preQuantScalar` is not allowed; the `src` element type must be `i8`, `i16`, `i32`, `i64`, `f16`, `bf16`, `f32`, `f8*`, `!pto.hif8`, or `!pto.f4*`.
  - For `loc=acc`: `src` must be `i32` or `f32`.
  - Without `preQuantScalar`: `dst` is `i32`, `f32`, `f16`, or `bf16`.
  - With `preQuantScalar`: `src=i32` → `dst=i8`, `f16`, or `bf16`; `src=f32` → `dst=i8`, `f16`, `bf16`, `f32`, `!pto.hif8`, or `f8E4M3*`.

**Hardware pipe:**

- `loc=acc` uses PIPE_FIX (floating-point fix)
- `loc=vec` / `loc=mat` use PIPE_MTE3 (MTE3 store)

**Example:**

```mlir
// Basic storage.
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)

// Store with final-stage flag.
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
           {stPhase = #pto<st_phase final>}

// Store with atomic add.
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
           {atomicType = #pto<atomic_type atomic_add>}

// Store from accumulator to mat with atomic operation and ReLU.
pto.tstore ins(%acc : !pto.tile_buf<loc=acc, dtype=i32, rows=32, cols=32,
                   v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                   fractal=1024, pad=0>)
           outs(%pv2 : !pto.partition_tensor_view<32x32xf16>)
           {atomicType = #pto<atomic_type atomic_add>, reluPreMode = #pto<relu_pre_mode normal_relu>}
```

---

### `pto.tstore` `fp` Form

```mlir
pto.tstore ins(<src> : <src_type> fp <fp> : <fp_type>)
  outs(<dst> : <dst_type>) {attributes}
```

**Semantics:** Uses the per-column scale/quantization parameter fp to transform the valid region of the accumulator and stores it to the GM destination view. fp is an independent scale tile, not a new data source or scalar byte address.

**Parameters and constraints:** **src** resides in `acc`, with element type `f32` or 32-bit integer; fp resides in `scaling`; dst is a GM partition view. fp and preQuantScalar are mutually exclusive; the source layout uses `col_major/row_major`, with valid column count in `[1,4095]`; precision conversion, ReLU, atomic, and phase attributes follow the corresponding rules of `tstore`. There is no SSA return value.

**Example:**

```mlir
pto.tstore
  ins(%acc : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=32,
    v_row=16, v_col=32, blayout=col_major, slayout=row_major,
    fractal=1024, pad=0>
    fp %fp : !pto.tile_buf<loc=scaling, dtype=f16, rows=1, cols=32,
    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  outs(%dst : !pto.partition_tensor_view<16x32xf16>)
```

---

### `pto.mgather`: Gather Load

```mlir
pto.mgather ins(<mem>, <idx> : <mem_type>, <idx_type>)
  outs(<dst> : <dst_type>) {coalesce = #pto<coalesce row|elem>}

// The elem mode of GM -> L1 additionally requires a GM temporary view.
pto.mgather ins(<mem>, <idx>, <scratch> : <mem_type>, <idx_type>, <scratch_type>)
  outs(<dst> : <dst_type>) {coalesce = #pto<coalesce elem>}
```

**Semantics:** Reads data from the GM view by row index or element-by-element index and writes it to the destination tile.

```text
coalesce=row:
    dst[r,j] = mem[row_index[r],j]
coalesce=elem:
    dst[r,j] = flat_mem[indexes[r,j]]
```

In row mode, the index unit is the source row; in elem mode, the index unit is the source element, addressed by the consecutive element number in the GM table. `coalesce` must be specified explicitly and cannot be omitted as the default mode.

**Parameters and return values:**

| Parameter | Type and Role |
| --- | --- |
| mem | GM `!pto.partition_tensor_view`, the source data table |
| idx | 32-bit integer index; GM→VEC uses a `vec` Tile, and GM→L1 uses a GM partition view |
| scratch | Used only in the elem mode of GM→L1, a GM partition view whose element type is the same as the destination |
| dst | A pre-allocated `vec` or `mat` Tile |

No SSA return value.

**Attributes:**

- `coalesce = #pto<coalesce row>`: reads by row; `#pto<coalesce elem>`: reads element by element.
- `gatherOob` defaults to `#pto<gather_oob undefined>`, which requires the index to be within the valid range; Ascend 950PR&950DT products can explicitly select `clamp`, `wrap`, or `zero` (out-of-bounds results are zero).

**Constraints:**

- Atlas A3 products/Ascend 950PR&950DT products both support the basic gather form; the non-default out-of-bounds mode is only used for Ascend 950PR&950DT products.
- The mem/dst element types are the same, supporting 8/16/32-bit integers, `f16`, `bf16`, and `f32`; Ascend 950PR&950DT products additionally support FP8 and HiFloat8 types.
- The source GM partition uses ND when the layout is inferable.
- GM→VEC: dst is `row_major/none_box`; idx is in `vec`, using `none_box`. The valid index shape in row mode is `[1,R]` for row_major or `[R,1]` for col_major, where R is the number of target valid rows; the elem mode matches the target valid shape. No scratch is provided.
- GM→L1: dst is in `mat`, using `col_major/row_major` and `fractal=512`; the number of physical rows is a multiple of 16, and the number of physical columns is a multiple of `32/sizeof(dtype)`. idx is a GM partition view and cannot use a VEC Tile; row mode does not provide scratch, while elem mode must provide a contiguous GM scratch with sufficient capacity.
- The index values and scratch capacity are guaranteed by the caller; the default undefined out-of-bounds mode provides no out-of-bounds protection.

**Example:**

```mlir
// Atlas A3 products/Ascend 950PR&950DT products: gather 32 rows from a 1024-row table.
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<1024x32xi32>,
  !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
    v_row=32, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  {coalesce = #pto<coalesce row>}
```

```mlir
// Ascend 950PR&950DT products: same row gather, with out-of-bounds rows zero-filled.
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<1024x32xi32>,
  !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
    v_row=32, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  {coalesce = #pto<coalesce row>, gatherOob = #pto<gather_oob zero>}
```

---

### `pto.mscatter`: Scatter Storage

```mlir
pto.mscatter ins(<src>, <idx> : <src_type>, <idx_type>)
  outs(<mem> : <mem_type>) {coalesce = #pto<coalesce row|elem>}
```

**Semantics:** Reads data from the VEC Tile and writes it to the GM destination table by row index or per-element index.

```text
coalesce=row:
    mem[row_index[r],j] = src[r,j]
coalesce=elem:
    flat_mem[indexes[r,j]] = src[r,j]
// In atomic mode, replace the assignment with the specified add/max/min update.
```

The index unit in row mode is the destination row, and in elem mode it is the destination element. GM positions that are not written retain their original values. Without atomic mode, users should avoid duplicate destination indices and must not rely on the order of conflicting writes.

**Parameters and return values:** src is a `vec` tile, idx is a 32-bit integer index Tile in `vec`, and mem is a GM `!pto.partition_tensor_view`; there is no SSA return value.

**Attributes:**

- `coalesce`: Explicit `row` or `elem`. When omitted, only the basic form distinguished by the index shape is supported: a row_major `[R,1]` row index, or an element index with the same shape as the src valid shape. It is recommended to write the mode explicitly.
- `scatterAtomicOp`: Default is `#pto<scatter_atomic_op none>`. On Ascend 950PR&950DT products, `add`, `max`, and `min` can be selected. add supports 32-bit integers, `f16`, and `f32`; max/min support 32-bit integers and `f32`.
- `scatterOob`: Default is `#pto<scatter_oob undefined>`. On Ascend 950PR&950DT products, `skip` (skip out-of-bounds), `clamp` (clamp), and `wrap` (around) can be selected.
- When a non-default atomic/oob or scatterConflict is specified, coalesce must be specified explicitly; these extensions do not change the index unit.

**Constraints:**

- Atlas A3 products/Ascend 950PR&950DT products both support the basic form; the non-default atomic and out-of-bounds modes are only for Ascend 950PR&950DT products.
- src and mem have the same element type, supporting 8/16/32-bit integers, `f16`, `bf16`, and `f32`; Ascend 950PR&950DT products additionally support FP8 and HiFloat8.
- src uses `row_major/none_box`; idx uses `none_box`; the GM view is ND when inferable.
- The valid index shape in explicit row mode is `[1,R]` for row_major or `[R,1]` for col_major, where R is the number of valid rows in the source; the valid index shape in explicit elem mode is the same as src.
- The default undefined mode does not protect against out-of-bounds access; the caller should provide indices within the valid range.

**Example:**

```mlir
// Atlas A3 products/Ascend 950PR&950DT products: write 32 rows to the specified rows in the GM table.
pto.mscatter ins(%src, %idx :
  !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
    v_row=32, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>,
  !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  outs(%mem : !pto.partition_tensor_view<1024x32xi32>)
  {coalesce = #pto<coalesce row>}
```

```mlir
// Ascend 950PR&950DT products: atomically accumulate and skip out-of-bounds rows.
pto.mscatter ins(%src, %idx :
  !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
    v_row=32, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>,
  !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  outs(%mem : !pto.partition_tensor_view<1024x32xi32>)
  {coalesce = #pto<coalesce row>,
   scatterAtomicOp = #pto<scatter_atomic_op add>,
   scatterOob = #pto<scatter_oob skip>}
```

---

### `pto.load`: Scalar Load

```mlir
%value = pto.load <ptr>[<offset>] : <ptr_type> -> <element_type>
```

**Semantics:** Returns `ptr[offset]` without modifying storage.

**Parameters and returns:** `ptr` is `!pto.ptr<T,space>` or `memref`; `offset` is `index`, in units of pointer elements, and cannot be a byte offset; the result type must equal the element type of the pointer or memref. The accessed address must be valid and satisfy the element alignment and capacity requirements.

**Example:**

```mlir
%offset = pto.constant 3 : index
%value = pto.load %ptr[%offset] : !pto.ptr<f32> -> f32
```

This reads the fourth f32 element, that is, the position 12 bytes from the base address.

---

### `pto.store`: Scalar Storage

```mlir
pto.store <value>, <ptr>[<offset>] : <ptr_type>, <element_type>
```

**Semantics:** Writes value to `ptr[offset]`; there is no SSA return value.

**Parameters and constraints:** ptr is `!pto.ptr<T,space>` or `memref`; offset is an `index` counted in elements; the value type must equal the element type of the pointer or memref. The type list specifies the pointer type first, then the value type; address validity, alignment, and capacity are guaranteed by the caller.

**Example:**

```mlir
%offset = pto.constant 3 : index
pto.store %value, %ptr[%offset] : !pto.ptr<f32>, f32
```

---

### `pto.tmov`: Local Memory Inter-Domain Data Movement

```mlir
// Basic form.
pto.tmov ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)

// Form with fp (scaling buffer) and attributes.
pto.tmov ins(<src> : <src_type>, <fp> : <fp_type>)
         outs(<dst> : <dst_type>)
         {accToVecMode = ..., reluPreMode = ...}

// Form with preQuantScalar.
pto.tmov ins(<src> : <src_type>, <preQuantScalar> : i64)
         outs(<dst> : <dst_type>)
         {attributes}
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[i, j]

(Supports optional precision conversion, ReLU pre-processing, and quantization.)
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer, located at `mat`, `vec`, or `acc` |
| `dst` | `pto.tile_buf` | Destination tile buffer, located at `left`, `right`, `bias`, `scaling`, etc. |
| `fp` | `pto.tile_buf<loc=scaling>` (optional) | Floating-point precision buffer, used only in specific conversions |
| `preQuantScalar` | `i64` (optional) | Scalar value before quantization |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `accToVecMode`: Conversion mode from accumulator to vec. Used only when `src.loc=acc` and `dst.loc=vec`.
  - `#pto<acc_to_vec_mode single_mode_vec0>`: Single vector execution unit mode, selects vec0
  - `#pto<acc_to_vec_mode single_mode_vec1>`: Single vector execution unit mode, selects vec1
  - `#pto<acc_to_vec_mode dual_mode_split_m>`: Dual vector execution unit mode, splits along the row dimension M
  - `#pto<acc_to_vec_mode dual_mode_split_n>`: Dual vector execution unit mode, splits along the column dimension N

- `reluPreMode`: ReLU pre-processing mode. The default value is `no_relu`. Supported only when `src.loc=acc`.
  - `#pto<relu_pre_mode no_relu>`: Does not perform ReLU processing
  - `#pto<relu_pre_mode normal_relu>`: Applies standard ReLU (max(0, x))

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The static shapes must match.
  - Supported position pairs: `mat` → `left`/`right`/`bias`/`scaling`; `vec` → `vec`; `acc` → `mat`; `acc` → `vec`.
  - `accToVecMode` is used only for the `acc` → `vec` conversion.
  - `reluPreMode`, `fp`, and `preQuantScalar` are supported only when `src.loc=acc`; `fp` and `preQuantScalar` are mutually exclusive.
  - `fp` must use `loc=scaling`; the source element type with `fp` must be `f32` or a 32-bit integer.
  - When `src.loc=acc` and `fp` or ReLU is used, the source layout must be `blayout=col_major, slayout=row_major`.
  - The destination `fractal` of `acc` → `mat` must be `512`; when `fp` is used, the destination layout must also be `blayout=col_major, slayout=row_major`.

- **Implementation check (Ascend 950PR&950DT products)**
  - Supported position pairs: `mat` → `left`/`right`/`bias`/`scaling`; `vec` → `vec`/`mat`; `acc` → `vec`/`mat`.
  - The constraints on `preQuantScalar`, `fp`, ReLU, and `fp.loc=scaling` are the same as above.
  - When `src.loc=acc` and `fp` or ReLU is used, the source layout must be `blayout=col_major, slayout=row_major`.
  - Ascend 950PR&950DT products do not require the destination `fractal=512` for `acc` → `mat`.

**Hardware pipe:**

- `vec` → `vec` uses PIPE_V (vector pipe)
- `mat` → `left`/`right`/`bias`/`scaling` uses PIPE_MTE1 (MTE1 transfer)
- `acc` → `mat`/`vec` uses PIPE_FIX (floating-point fix)

**Example:**

```mlir
// Basic acc to vec conversion.
pto.tmov ins(%src : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=col_major, slayout=row_major,
                  fractal=1024, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
```

```mlir
// acc to vec conversion with ReLU.
pto.tmov ins(%src : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                  v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                  fractal=1024, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                  v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
         {reluPreMode = #pto<relu_pre_mode normal_relu>}
```

```mlir
// Move mat to scaling.
pto.tmov ins(%src : !pto.tile_buf<loc=mat, dtype=f16, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=scaling, dtype=f16, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
```

---

### `pto.ttrans`: Tile Transpose

```mlir
pto.ttrans ins(<src>, <tmp> : <src_type>, <tmp_type>)
           outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[j, i]

Use the temporary buffer tmp to complete the transpose operation.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer, which must use `blayout=row_major`. |
| `tmp` | `pto.tile_buf` | Temporary working buffer. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The element types of `src` and `dst` must match.
  - `src` must use `blayout=row_major`.
  - The element size must be 1, 2, or 4 bytes.
  - Supported types: `i32`/`f32` (4 bytes), `i16`/`f16`/`bf16` (2 bytes), `i8` (1 byte).
  - The transpose is performed on the valid region of `src`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The element sizes of `src` and `dst` must match.
  - 32-byte alignment is required on the main dimension.
  - `src` and `dst` must have the same element type.
  - The transpose operates on static tile shapes.

**Hardware pipe:** PIPE_V (vector pipe)

**Example:**

```mlir
pto.ttrans ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                       v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                       fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                       v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                       fractal=512, pad=0>)
           outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.tmov` `fp` Form

```mlir
pto.tmov ins(<src> : <src_type>, <fp> : <fp_type>)
  outs(<dst> : <dst_type>) {attributes}
```

**Semantics:** Uses the scale tile fp to perform supported precision conversion on accumulator data and writes it to the destination local Tile; no SSA return value is produced.

**Constraints:** src is located in acc, with element type f32 or 32-bit integer; fp is located in scaling; fp and preQuantScalar are mutually exclusive. The source uses col_major/row_major layout, and the destination path and layout, conversion type, and ReLU follow the constraints of `tmov`. For Atlas A3 products, the acc→mat destination fractal is 512, and the source and destination physical shapes match; Ascend 950PR&950DT products do not require this destination fractal value.

**Example:**

```mlir
pto.tmov ins(%src : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=256,
    v_row=16, v_col=256, blayout=col_major, slayout=row_major,
    fractal=1024, pad=0>, %fp : !pto.tile_buf<loc=scaling, dtype=f32, rows=16, cols=256,
    v_row=16, v_col=256, blayout=row_major, slayout=row_major,
    fractal=512, pad=0>)
  outs(%dst : !pto.tile_buf<loc=mat, dtype=i8, rows=16, cols=256,
    v_row=16, v_col=256, blayout=col_major, slayout=row_major,
    fractal=512, pad=0>)
```
