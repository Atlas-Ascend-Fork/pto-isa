# Reduction Operations

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:50:13.200Z pushedAt=2026-11-02T03:00:52.139Z nl=2 -->

This section describes all operations in the PTO ISA that perform reduction along rows or columns. All operations act on local buffers (`tile_buf`, located in the `loc=vec` space) and adopt the Destination-Passing Style (DPS): the operation itself does not produce an SSA return value, but instead writes the result directly into a pre-allocated destination `tile_buf`. All operations execute on the **vector pipe** (`PIPE_V`).

This category of operations generally has the following assembly form:

```mlir
pto.op ins(%src : !pto.tile_buf<...>)
       outs(%dst : !pto.tile_buf<...>)
```

The general constraints typically include:

- All tiles must be located in `loc=vec` (VEC/UB storage space).
- The input tile uses an ND-style layout (`blayout=row_major`, `slayout=none_box`).
- The input and output element types are consistent (except for `argmax`/`argmin`, whose output is an integer index type).

---

## Contents

- [`pto.trowsum`: Row Sum Reduction](#ptotrowsum-row-sum-reduction)
- [`pto.trowprod`: Row Product Reduction](#ptotrowprod-row-product-reduction)
- [`pto.trowmax`: Row Maximum Value Reduction](#ptotrowmax-row-maximum-reduction)
- [`pto.trowmin`: Row Minimum Value Reduction](#ptotrowmin-row-minimum-reduction)
- [`pto.trowargmax`: Row Maximum Value Index Reduction](#ptotrowargmax-row-maximum-value-index-reduction)
- [`pto.trowargmin`: Row Minimum Value Index Reduction](#ptotrowargmin-row-minimum-value-index-reduction)
- [`pto.tcolsum`: Column Sum Reduction](#ptotcolsum-column-sum-reduction)
- [`pto.tcolprod`: Column Product Reduction](#ptotcolprod-column-product-reduction)
- [`pto.tcolmax`: Column Maximum Value Reduction](#ptotcolmax-column-maximum-value-reduction)
- [`pto.tcolmin`: Column Minimum Value Reduction](#ptotcolmin-column-minimum-value-reduction)
- [`pto.tcolargmax`: Column Maximum Value Index Reduction](#ptotcolargmax-column-maximum-value-index-reduction)
- [`pto.tcolargmin`: Column Minimum Value Index Reduction](#ptotcolargmin-column-minimum-value-index-reduction)

---

## Operation Details

### `pto.trowsum`: Row Sum Reduction

```mlir
pto.trowsum ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each row i:
    dst[i, 0] = sum over j of src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile, column vector, stores the sum result of each row |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src` and `dst` must use `loc=vec`
  - `src` uses an ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - `dst` layout: recommended to use a DN-style 1D column vector (`cols=1`, `blayout=col_major`); also compatible with an ND-style 2D tile (`valid column == 1`)
  - Data type: `i16`, `i32`, `f16`, `f32`
  - Element type consistent: `src_type == dst_type`
  - `src valid column != 0` and `src valid row != 0`
  - `src valid row == dst valid row`

**Example:**

```mlir
pto.trowsum ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.trowprod`: Row Product Reduction

```mlir
pto.trowprod ins(<src>, <tmp> : <src_type>, <tmp_type>)
             outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each row i:
    dst[i, 0] = product over j of src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile |
| `tmp` | `pto.tile_buf` | Temporary buffer with the same shape and element type as `src` |
| `dst` | `pto.tile_buf` | Destination tile, a column vector that stores the product result of each row |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src` and `dst` must use `loc=vec`
  - `src` uses an ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - `tmp` must have the same shape and element type as `src`
  - `dst` layout: recommended to use a DN-style 1D column vector (`cols=1`, `blayout=col_major`); also compatible with an ND-style 2D tile (`valid column == 1`)
  - Data type: `i16`, `i32`, `f16`, `f32`
  - Element type consistency: `src_type == dst_type`
  - `src valid column != 0` and `src valid row != 0`
  - `src valid row == dst valid row`

**Example:**

```mlir
pto.trowprod ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=1,
                 v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### `pto.trowmax`: Row Maximum Reduction

```mlir
pto.trowmax ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each row i:
    dst[i, 0] = max over j of src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile, column vector, stores the maximum of each row |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src` and `dst` must use `loc=vec`
  - `src` uses an ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - `dst` layout: recommended to use a DN-style 1D column vector (`cols=1`, `blayout=col_major`); also compatible with an ND-style 2D tile (`valid column == 1`)
  - Data types: `i16`, `i32`, `f16`, `f32`
  - Element types are consistent: `src_type == dst_type`
  - `src valid column != 0` and `src valid row != 0`
  - `src valid row == dst valid row`

**Example:**

```mlir
pto.trowmax ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.trowmin`: Row Minimum Reduction

```mlir
pto.trowmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
            outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each row i:
    dst[i, 0] = min over j of src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile |
| `tmp` | `pto.tile_buf` | Temporary buffer, used for intermediate computation |
| `dst` | `pto.tile_buf` | Destination tile, column vector, stores the minimum value of each row |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src` and `dst` must use `loc=vec`
  - `src` uses an ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - `dst` layout: recommended to use a DN-style 1D column vector (`cols=1`, `blayout=col_major`); also compatible with an ND-style 2D tile (`valid column == 1`)
  - Data types: `i16`, `i32`, `f16`, `f32`
  - Element type consistent: `src_type == dst_type`
  - `src valid column != 0` and `src valid row != 0`
  - `src valid row == dst valid row`

**Example:**

```mlir
pto.trowmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.trowargmax`: Row Maximum Value Index Reduction

```mlir
pto.trowargmax ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each row i:
    dst[i, 0] = argmax over j of src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile |
| `tmp` | `pto.tile_buf` | Temporary buffer with the same shape and element type as `src` |
| `dst` | `pto.tile_buf` | Destination tile that stores the column index of the maximum value in each row |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src`, `tmp`, and `dst` must use `loc=vec`
  - `src` uses an ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - `tmp` must have the same shape, valid shape, and element type as `src`
  - `dst` uses `slayout=none_box` and is a DN-style column vector (`blayout=col_major`, `cols=1`) or an ND-style tile (`valid column == 1`)
  - `src` element type: `i16`, `i32`, `f16`, `f32`
  - `dst` element type: `i32` or `ui32`
  - `src valid row != 0` and `src valid column != 0`
  - `src valid row == dst valid row`
  - `dst valid column == 1`

**Example:**

```mlir
pto.trowargmax ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=1,
                   v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.trowargmin`: Row Minimum Value Index Reduction

```mlir
pto.trowargmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each row i:
    dst[i, 0] = argmin over j of src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile |
| `tmp` | `pto.tile_buf` | Temporary buffer with the same shape and element type as `src` |
| `dst` | `pto.tile_buf` | Destination tile that stores the column index of the minimum value in each row |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src`, `tmp`, and `dst` must use `loc=vec`
  - `src` uses an ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - `tmp` must have the same shape, valid shape, and element type as `src`
  - `dst` uses `slayout=none_box` and is a DN-style column vector (`blayout=col_major`, `cols=1`) or an ND-style tile (`valid column == 1`)
  - `src` element type: `i16`, `i32`, `f16`, `f32`
  - `dst` element type: `i32` or `ui32`
  - `src valid row != 0` and `src valid column != 0`
  - `src valid row == dst valid row`
  - `dst valid column == 1`

**Example:**

```mlir
pto.trowargmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=1,
                   v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.tcolsum`: Column Sum Reduction

```mlir
pto.tcolsum ins(<src>, <tmp> {isBinary = false} : <src_type>, <tmp_type>)
            outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each column j:
    dst[0, j] = sum over i of src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile |
| `tmp` | `pto.tile_buf` | Temporary buffer used for intermediate computation |
| `dst` | `pto.tile_buf` | Destination tile, a row vector that stores the sum result of each column |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `isBinary`: Whether to use a binary reduction tree. The default value is `false`.
  - `true`: Uses a binary reduction tree
  - `false`: Uses the default reduction method

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src`, `tmp`, and `dst` must use `loc=vec`
  - All tiles use the ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - Data types: `f16`, `f32`, `i16`, `i32`
  - Element types are consistent: `dst_type == tmp_type == src_type`
  - `src valid column == dst valid column`

- **Implementation check (Ascend 950PR&950DT products)**
  - `src`, `tmp`, and `dst` must use `loc=vec`
  - All tiles use the ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - Data types: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`
  - Element types are consistent: `dst_type == tmp_type == src_type`
  - `src valid row` and `src valid column` must be non-zero
  - `src valid column == dst valid column`

**Example:**

```mlir
pto.tcolsum ins(%src, %tmp {isBinary = false} : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.tcolprod`: Column Product Reduction

```mlir
pto.tcolprod ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each column j:
    dst[0, j] = product over i of src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile, a row vector that stores the product result of each column |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src` and `dst` must use `loc=vec`
  - All tiles use ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - Data type: `f16`, `f32`, `i16`, `i32`
  - Element type consistent: `dst_type == src_type`
  - `src valid column == dst valid column`

- **Implementation check (Ascend 950PR&950DT products)**
  - `src` and `dst` must use `loc=vec`
  - All tiles use ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - Data type: `i16`, `ui16`, `i32`, `ui32`, `f16`, `bf16`, `f32`
  - Element type consistent: `dst_type == src_type`
  - `src valid column == dst valid column`

**Example:**

```mlir
pto.tcolprod ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                 v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### `pto.tcolmax`: Column Maximum Value Reduction

```mlir
pto.tcolmax ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each column j:
    dst[0, j] = max over i of src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile, row vector, stores the maximum value of each column |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src` and `dst` must use `loc=vec`
  - All tiles use the ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - Data types: `f16`, `f32`, `i16`, `i32`
  - Element types are consistent: `dst_type == src_type`
  - `src valid column == dst valid column`

- **Implementation check (Ascend 950PR&950DT products)**
  - `src` and `dst` must use `loc=vec`
  - All tiles use the ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - Data types: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`
  - Element types are consistent: `dst_type == src_type`
  - `src valid row` and `src valid column` must be non-zero
  - `src valid column == dst valid column`

**Example:**

```mlir
pto.tcolmax ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.tcolmin`: Column Minimum Value Reduction

```mlir
pto.tcolmin ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each column j:
    dst[0, j] = min over i of src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile, row vector, stores the minimum of each column |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src` and `dst` must use `loc=vec`
  - All tiles use ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - Data type: `f16`, `f32`, `i16`, `i32`
  - Element type consistent: `dst_type == src_type`
  - `src valid column == dst valid column`

- **Implementation check (Ascend 950PR&950DT products)**
  - `src` and `dst` must use `loc=vec`
  - All tiles use ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - Data types: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`
  - Element types are consistent: `dst_type == src_type`
  - `src valid row` and `src valid column` must be non-zero
  - `src valid column == dst valid column`

**Example:**

```mlir
pto.tcolmin ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.tcolargmax`: Column Maximum Value Index Reduction

```mlir
pto.tcolargmax ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each column j:
    dst[0, j] = argmax over i of src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile |
| `tmp` | `pto.tile_buf` | Temporary buffer with the same shape and element type as `src` |
| `dst` | `pto.tile_buf` | Destination tile that stores the row index of the maximum value in each column |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src`, `tmp`, and `dst` must use `loc=vec`
  - All tiles use ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - `tmp` must have the same shape, valid shape, and element type as `src`
  - The `src` element type must be `f16` or `f32`
  - The `dst` element type must be `i32` or `ui32`
  - `src valid row != 0` and `src valid column != 0`
  - `dst valid row == 1`
  - `src valid column == dst valid column`

**Example:**

```mlir
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=32,
                   v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.tcolargmin`: Column Minimum Value Index Reduction

```mlir
pto.tcolargmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each column j:
    dst[0, j] = argmin over i of src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile |
| `tmp` | `pto.tile_buf` | Temporary buffer with the same shape and element type as `src` |
| `dst` | `pto.tile_buf` | Destination tile that stores the row index of the minimum value in each column |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src`, `tmp`, and `dst` must use `loc=vec`
  - All tiles use the ND-style layout (`blayout=row_major`, `slayout=none_box`)
  - `tmp` must have the same shape, valid shape, and element type as `src`
  - The `src` element type must be `f16` or `f32`
  - The `dst` element type must be `i32` or `ui32`
  - `src valid row != 0` and `src valid column != 0`
  - `dst valid row == 1`
  - `src valid column == dst valid column`

**Example:**

```mlir
pto.tcolargmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
                   v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```
