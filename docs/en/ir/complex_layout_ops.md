# Layout Transformation

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:07:50.811Z pushedAt=2026-11-02T03:00:52.117Z nl=2 -->

This section describes the instruction names, signatures, and semantics of layout transformation operations in the PTO ISA. All operations act on the local `tile_buf`.

Generic assembly form:

```mlir
pto.op ins(<src>, ... : <src_type>, ...) outs(<dst> : <dst_type>)
```
---

## Contents

- [`pto.treshape`: Tile Shape Reinterpretation](#ptotreshape-tile-shape-reinterpretation)
- [`pto.tconcat`: Column-Direction Tile Concatenation](#ptotconcat-column-direction-tile-concatenation)
- [`pto.tconcatidx`: Index Control Column Concatenation](#ptotconcatidx-index-control-column-concatenation)
- [`pto.textract`: Sub-Tile Extraction](#ptotextract-sub-tile-extraction)
- [`pto.textract` `fp` Form](#ptotextract-fp-form)
- [`pto.tinsert`: Sub-Tile Insertion](#ptotinsert-sub-tile-insertion)
- [`pto.tinsert` `fp` Form](#ptotinsert-fp-form)
- [`pto.tgather`: Gathering/Selecting Elements](#ptotgather-gatheringselecting-elements)
- [`pto.tgatherb`: Gathering 32-Byte Blocks by Byte Offset](#ptotgatherb-gathering-32-byte-blocks-by-byte-offset)
- [`pto.tscatter`: Scattering Elements](#ptotscatter-scattering-elements)

---

## Operation Details

### `pto.treshape`: Tile Shape Reinterpretation

```mlir
%view = pto.treshape <src> : <src_type> -> <result_type>
```

**Semantics:**

```text
view = reinterpret_view(src, result_type)
// view shares the same underlying storage as src, without allocating or copying elements.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |

**Returns:** A `!pto.tile_buf` view of the specified result type. After writing to the shared storage through either the source or result view, the other view observes the change; it is not an independent copy of the data.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The source and result views must use the same storage location: `src.loc == view.loc`.
  - The source and result must have static physical sizes, and their total byte sizes must be equal.
  - Conversion between boxed and non-boxed layouts is not supported.

**Example:**

```mlir
%view = pto.treshape %src
    : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                             v_row=16, v_col=32, blayout=row_major,
                             slayout=none_box, fractal=512, pad=0>
    -> !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=16,
                              v_row=32, v_col=16, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>
```

---

### `pto.tconcat`: Column-Direction Tile Concatenation

```mlir
pto.tconcat ins(<src0>, <src1> : !pto.tile_buf, !pto.tile_buf)
            outs(<dst> : !pto.tile_buf)
```

**Semantics:**

```text
For each row i:
    dst[i, 0:C0) = src0[i, 0:C0)
    dst[i, C0:C0+C1) = src1[i, 0:C1)
```

Where C0 is the column count of **src0** and C1 is the column count of **src1**.

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer (left side). |
| `src1` | `pto.tile_buf` | Second source tile buffer (right side). |
| `dst` | `pto.tile_buf` | Target tile buffer (concatenation result). |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - **src0**, **src1**, and **dst** must use the same element type, which must be one of the following: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - All tiles must use `loc=vec`.
  - The three tiles must be rank-2, and the valid row count of **src0** and **src1** must be the same as the valid row count of **dst**.
  - The valid column count of **src0** + the valid column count of **src1** <= the column count of **dst**.
  - Concatenation changes the length in the column direction, so the physical static shapes of the three are not required to be identical.

- **Implementation check (Ascend 950PR&950DT products)**
  - Same as the requirements for Atlas A2 products/Atlas A3 products, with the additional requirement that all tiles must use `blayout=row_major`.

**Example:**

```mlir
pto.tconcat
    ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=16,
                               v_row=32, v_col=16, blayout=row_major,
                               slayout=none_box, fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=16,
                               v_row=32, v_col=16, blayout=row_major,
                               slayout=none_box, fractal=512, pad=0>)
    outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                            v_row=32, v_col=32, blayout=row_major,
                            slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.tconcatidx`: Index Control Column Concatenation

```mlir
pto.tconcatidx ins(<src0>, <src1>, <src0Idx>, <src1Idx>
                   : <src0_type>, <src1_type>, <idx0_type>, <idx1_type>)
               outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each row i:
    idx0_num = src0Idx[i, 0]
    idx1_num = src1Idx[i, 0]
    copy from src0: min(idx0_num, src0_valid_col, dst_valid_col) columns
    copy from src1: min(idx1_num, src1_valid_col, dst_valid_col - copied_from_src0) columns
```

Performs a column-direction operation that concatenates data from two source tiles into the destination tile row by row under index control. Unlike `pto.tconcat`, the number of concatenated columns in each row is dynamically controlled by the index tile.

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer. |
| `src1` | `pto.tile_buf` | Second source tile buffer. |
| `src0Idx` | `pto.tile_buf` | Per-row index tile of **src0** (specifies the number of columns to copy in each row). |
| `src1Idx` | `pto.tile_buf` | Per-row index tile of **src1** (specifies the number of columns to copy in each row). |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - All operands must use `loc=vec`.
  - The element types of the data tiles (**src0**, **src1**, **dst**) must be identical and must be one of the following: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - The element types of the index tiles (**src0Idx**, **src1Idx**) must be identical, and signless `i8`, `i16`, or `i32` is allowed.

- **Implementation check (Ascend 950PR&950DT products)**
  - Same as the constraints for Atlas A2 products/Atlas A3 products.
  - Additionally requires all operands to use `blayout=row_major`.

**Example:**

```mlir
pto.tconcatidx
    ins(%src0, %src1, %idx0, %idx1 :
        !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=8,
                      v_row=16, v_col=1, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=8,
                      v_row=16, v_col=1, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=64,
                              v_row=16, v_col=64, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.textract`: Sub-Tile Extraction

```mlir
pto.textract ins(<src>, <indexRow>, <indexCol> : !pto.tile_buf, index, index)
             outs(<dst> : !pto.tile_buf)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[i + indexRow, j + indexCol]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `indexRow` | `index` | Extraction start offset in rows. |
| `indexCol` | `index` | Extraction start column offset in elements. |
| `dst` | `pto.tile_buf` | Destination tile buffer (sub-region). |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2/A3 training products)**
  - The basic `vec` → `vec` and `mat` → `left`/`right` forms use the same element type, supporting `i8`, `f16`, `bf16`, and `f32`.
  - Supports `vec` → `vec`, `mat` → `left`/`right`, and `acc` → `mat`. The accumulator conversion form supports `f32` → `f16`/`bf16`; for additional quantization parameters, see the `fp` form in this chapter.
  - The layout/fractal of **src** must be compatible with the combinations supported by **dst**.
  - Runtime constraint: `indexRow + dst.rows <= src.rows` and `indexCol + dst.cols <= src.cols`
  - The targets of `mat` → `left`/`right` use `row_major/row_major` and `row_major/col_major` layouts respectively; the source and target of `acc` → `mat` use `col_major/row_major`, with target `fractal=512`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The basic non-accumulator form uses the same element type; the accumulator form can perform supported precision conversion, with additional quantization parameters described in the `fp` form of this chapter.
  - Supports Mat->Left/Right/Scaling, Vec->Mat, Acc->Mat/Vec, and
    Vec->Vec with ND layout; ND refers to `blayout=row_major, slayout=none_box`.
  - Runtime constraint: `indexRow + dst.rows <= src.rows` and `indexCol + dst.cols <= src.cols`

**Example:**

```mlir
pto.textract
    ins(%src, %row, %col :
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                      v_row=32, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        index, index)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                              v_row=16, v_col=16, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.textract` `fp` Form

```mlir
pto.textract ins(<src>, <indexRow>, <indexCol> : <src_type>, index, index
                fp <fp> : <fp_type>)
             outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = dequant_extract(src[i + indexRow, j + indexCol], fp)
// Extract a sub-window from the accumulator tile while performing dequantization/type conversion through the scaling factor.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer, which must be `loc=acc`. |
| `fp` | `pto.tile_buf` | Scaling factor tile buffer, which must be `loc=scaling`. |
| `indexRow` | `index` | Extraction start row offset. |
| `indexCol` | `index` | Extraction start column offset. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2/A3 products)**
  - The locations must be `src=acc`, `fp=scaling`, and `dst=mat`.
  - Supported type pairs: `(src=f32, dst=i8)` or `(src=i32, dst=i8/f16/i16)`.
  - The fractal of `dst` must be 512.

- **Implementation check (Ascend 950PR&950DT products)**
  - The locations must be `src=acc`, `fp=scaling`, and `dst` can be `mat` or `vec`.
  - Supported type pairs: `(src=f32, dst=i8/fp8/f16/bf16/f32)` or `(src=i32, dst=i8/f16/bf16)`.
  - No fractal 512 restriction.

**Example:**

```mlir
pto.textract ins(%src, %row, %col : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                 fractal=1024, pad=0>, index, index
                 fp %fp : !pto.tile_buf<loc=scaling, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=row_major, slayout=row_major,
                 fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=mat, dtype=i8, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                 fractal=512, pad=0>)
```

---

### `pto.tinsert`: Sub-Tile Insertion

```mlir
pto.tinsert ins(<src>, <indexRow>, <indexCol> : !pto.tile_buf, index, index)
            outs(<dst> : !pto.tile_buf)
```

**Semantics:**

```text
For each element (i, j):
    dst[i + indexRow, j + indexCol] = src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer (to be inserted). |
| `indexRow` | `index` | Insertion start row offset. |
| `indexCol` | `index` | Insertion start column offset. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src` and `dst` must be rank-2 `tile_buf`; `indexRow` and `indexCol` must be non-negative, and the insertion region must not exceed the static boundary of `dst`.
  - Supports Vec->Vec (same element type, `i8`/`f16`/`bf16`/`f32`) and Acc->Mat.
  - Acc->Mat requires both ends to use the NZ layout (`col_major` + `row_major`), with the destination `fractal=512`.
  - `fp` and `preQuantScalar` are mutually exclusive and apply only to `src.loc=acc`; `fp` must use `loc=scaling`.

- **Implementation check (Ascend 950PR&950DT products)**
  - Supports Acc->Mat/Vec, Vec->Mat, and Vec->Vec.
  - For Vec->Vec, the layouts on both ends must both be ND (`row_major` + `none_box`) or both be NZ (`col_major` + `row_major`).
  - For Vec->Mat, the destination must be NZ; the source can be ND or NZ, and the source and destination element types must be the same.
  - `accToVecMode` applies only to Acc->Vec; `tinsertMode` applies only to NZ Vec->Mat.
  - `fp`, `preQuantScalar`, and the ReLU form require `src.loc=acc`; `fp` must use `loc=scaling`.

**Example:**

```mlir
pto.tinsert
    ins(%src, %row, %col :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        index, index)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                              v_row=32, v_col=32, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.tinsert` `fp` Form

```mlir
pto.tinsert ins(<src>, <indexRow>, <indexCol> : <src_type>, index, index
               fp <fp> : <fp_type>)
            outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i + indexRow, j + indexCol] = quant_insert(src[i, j], fp)
// Quantize the vector tile by the scaling factor and insert it into the specified sub-window of the accumulator tile.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer, which must be `loc=acc`. |
| `fp` | `pto.tile_buf` | Scaling factor tile buffer, which must be `loc=scaling`. |
| `indexRow` | `index` | Insertion start row offset. |
| `indexCol` | `index` | Insertion start column offset. |
| `dst` | `pto.tile_buf` | Target tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The locations must be `src=acc`, `fp=scaling`, `dst=mat`.
  - Supported type pairs: `(src=f32, dst=i8)` or `(src=i32, dst=i8/f16/i16)`.
  - The fractal of `dst` must be 512.

- **Implementation check (Ascend 950PR&950DT products)**
  - The locations must be `src=acc`, `fp=scaling`, and `dst` can be `mat` or `vec`.
  - Supported type pairs: `(src=f32, dst=i8/fp8/f16/bf16/f32)` or `(src=i32, dst=i8/f16/bf16)`.
  - No fractal 512 restriction.

**Example:**

```mlir
pto.tinsert ins(%src, %row, %col : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                fractal=1024, pad=0>, index, index
                fp %fp : !pto.tile_buf<loc=scaling, dtype=f32, rows=32, cols=32,
                v_row=32, v_col=32, blayout=row_major, slayout=row_major,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=mat, dtype=i8, rows=32, cols=32,
                v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                fractal=512, pad=0>)
```

---

### `pto.tgather`: Gathering/Selecting Elements

`pto.tgather` has three usage forms: index form, comparison form, and mask form.

```mlir
// Index form.
pto.tgather ins(<src>, <indices> : <src_type>, <indices_type>)
            outs(<dst> : <dst_type>)

// Comparison form.
pto.tgather ins(<src>, <kValue> : <src_type>, <scalar_type>)
            outs(<dst>, <cdst> : <dst_type>, <cdst_type>)
            {cmpMode = #pto<cmp <mode>>, offset = <i32>}

// Mask form.
pto.tgather ins(<src>, {maskPattern = #pto.mask_pattern<<pattern>>} : <src_type>, "row")
            outs(<dst> : <dst_type>)
```

**Semantics:**

```text
Index form:
    For each element (i, j):
        dst[i, j] = src[indices[i, j]]

Comparison form:
    dst stores the indices that satisfy the scalar comparison condition.
    cdst stores the number of selected elements in each row.

Mask form:
    Scans the valid region of src in row-major order, keeping only the elements whose column positions satisfy maskPattern,
    and writes these elements sequentially and contiguously into dst.

    Let the valid region of src be R x C. dst is written sequentially starting from linear position 0.
    If the element at row r, column c is selected by maskPattern, then:
        dst[linear_out] = src[r, c]
        linear_out = linear_out + 1
```

For single-row input, or when observing only the selected column positions within each row, the default mask form can be written as:

```text
P1111:
    dst[r, k] = src[r, k]

P0101:
    dst[r, k] = src[r, 2*k]

P1010:
    dst[r, k] = src[r, 2*k + 1]

P0001:
    dst[r, k] = src[r, 4*k]

P0010:
    dst[r, k] = src[r, 4*k + 1]

P0100:
    dst[r, k] = src[r, 4*k + 2]

P1000:
    dst[r, k] = src[r, 4*k + 3]
```

Where, `k` is taken until the index on the right side is still less than the valid column count of `src`. The write result of `dst` is contiguously packed, so when `src` has multiple rows, the trailing elements selected from the previous row and the leading elements selected from the next row are also stored contiguously in `dst`.

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `dst` | `pto.tile_buf` | Primary destination tile buffer. |
| `cdst` | `pto.tile_buf` | Auxiliary destination tile in the comparison form (comparison form only). |
| `indices` | `pto.tile_buf` | Index tile in the index form (index form only). |
| `kValue` | Scalar type | Scalar comparison value in the comparison form (comparison form only). |

**Returns:** None. Writes to `dst` (and `cdst` in the comparison form) in DPS form.

**Attributes:**

- `maskPattern`: Mask pattern, used only in the mask form.
  - `#pto.mask_pattern<P0101>`: Takes the 1st of every 2 elements, that is, selects columns `0, 2, 4, ...`.
  - `#pto.mask_pattern<P1010>`: Takes the 2nd of every 2 elements, that is, selects columns `1, 3, 5, ...`.
  - `#pto.mask_pattern<P0001>`: Takes the 1st of every 4 elements, that is, selects columns `0, 4, 8, ...`.
  - `#pto.mask_pattern<P0010>`: Takes the 2nd of every 4 elements, that is, selects columns `1, 5, 9, ...`.
  - `#pto.mask_pattern<P0100>`: Takes the 3rd of every 4 elements, that is, selects columns `2, 6, 10, ...`.
  - `#pto.mask_pattern<P1000>`: Takes the 4th of every 4 elements, that is, selects columns `3, 7, 11, ...`.
  - `#pto.mask_pattern<P1111>`: Selects all columns (no filtering).

- `cmpMode`: Comparison mode, used only in the comparison form. The default value is `eq`.
  - `#pto<cmp eq>`: equal comparison
  - `#pto<cmp gt>`: greater-than comparison

- `offset`: gather base index offset in the comparison form. The default value is `0`.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - Index form: the `src` and `dst` element types must be identical and be one of `i16`, `i32`, `f16`, or `f32`. The `indices` element type must be `i32`, and the physical shape and valid shape must be static values. The `valid_shape[1]` of `dst` must be equal to `dst.cols`.
  - Comparison form: the `dst` and `cdst` element types must be `i32`. The `src` element type must be `f16`, `f32`, or `i32` when `cmpMode=eq`; the physical shape of `src` must be a static value. The `kValue` type must be consistent with the `src` element type. `cmpMode` must be `eq` or `gt`. `src`, `dst`, and `cdst` must use `loc=vec`.
  - Mask form: the `src` element size must be 2 or 4 bytes. `src` and `dst` must use `loc=vec` and `blayout=row_major`. The `src` and `dst` element sizes must be identical. The `valid_shape[1]` of `dst` must be equal to `dst.cols`.

- **Implementation check (Ascend 950PR&950DT products)**
  - Index form: the `src` and `dst` element types must be identical and be one of `i8`, `i16`, `i32`, `f16`, or `f32`. The `indices` element type can be `i16` or `i32`. The `valid_shape[1]` of `dst` must be equal to `dst.cols`.
  - Comparison form: the `dst` and `cdst` element types must be `i32`. The `src` element type must be one of `i16`, `i32`, `f16`, or `f32`. The `kValue` type must be consistent with the `src` element type. `cmpMode` must be `eq` or `gt`. `src`, `dst`, and `cdst` must use `loc=vec`.
  - Mask form: The `src` element size must be 1, 2, or 4 bytes. `src` and `dst` must use `loc=vec` and `blayout=row_major`. The `src`/`dst` element type must be one of `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`, or an fp8-class supported type. The `src` and `dst` element sizes must be identical. The `valid_shape[1]` of `dst` must equal `dst.cols`.

**Example:**

```mlir
// Index form.
pto.tgather ins(%src, %indices :
                !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
            outs(%index_dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
```

---

### `pto.tgatherb`: Gathering 32-Byte Blocks by Byte Offset

```mlir
pto.tgatherb ins(<src>, <offsets> : <src_type>, <offsets_type>)
             outs(<dst> : <dst_type>)
```

**Semantics:**

```text
Each element of offsets gives the first-byte address of a 32-byte block on the source side:
    The b-th 32-byte block of the i-th row of dst = the 32 bytes in src starting at offsets[i, b]
Where the number of blocks per row of dst = ceil(dst valid column count / (32 / sizeof(dst element type)))
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `offsets` | `pto.tile_buf` | Tile of byte addresses of 32-byte blocks; the valid row count must match `dst`, and the valid column count is `align_up(ceil(dst valid column count / (32 / sizeof(dst element type))), 8)`. |
| `dst` | `pto.tile_buf` | Target tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `dst` and `offsets` must use the row-major layout (`blayout=row_major`).
  - The `dst` element size must be 1, 2, or 4 bytes.
  - The `offsets` element type must be a 32-bit integer (`i32`/`ui32`).
  - The valid row count of `offsets` must be equal to the valid row count of `dst`.
  - The valid column count of `offsets` must be the compact 32-byte block address count aligned to 8: `align_up(ceil(dst valid column count / (32 / sizeof(dst element type))), 8)`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The `dst` element size must be 1, 2, or 4 bytes.
  - No row-major layout requirement.

**Example:**

```mlir
// dst is an 8x32 f32 tile: each row is 32*4=128 bytes, i.e., 4 32-byte blocks; after alignment to 8, offsets has 8 columns per row.
pto.tgatherb ins(%src, %offsets :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=32,
                     v_row=8, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=i32, rows=8, cols=8,
                     v_row=8, v_col=8, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=32,
                     v_row=8, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tscatter`: Scattering Elements

`pto.tscatter` has two usage forms: index form and mask form.

```mlir
// Index form.
pto.tscatter ins(<src>, <indexes> : <src_type>, <indexes_type>)
             outs(<dst> : <dst_type>)

// Mask form.
pto.tscatter ins(<src>, {maskPattern = #pto.mask_pattern<<pattern>>} : <src_type>, <axis>)
             outs(<dst> : <dst_type>)
```

**Semantics:**

```text
Index form:
    For each element (i, j):
        flat_dst[indexes[i, j]] = src[i, j]

Mask form:
    Scatters the elements stored contiguously in row-major order in src back to dst at the positions specified by maskPattern;
    Positions not matched by maskPattern are filled with 0.
```

The mask form with `axis="row"` can be regarded as the reverse expansion of the mask form of `pto.tgather`. Let the valid region of `src` be `R x Csrc` and the valid region of `dst` be `R x Cdst`, satisfying:

- `P0101` / `P1010`: `Cdst = 2 * Csrc`
- `P0001` / `P0010` / `P0100` / `P1000`: `Cdst = 4 * Csrc`
- `P1111`: `Cdst = Csrc`

Then:

```text
P1111:
    dst[r, k] = src[r, k]

P0101:
    dst[r, 2*k]     = src[r, k]
    dst[r, 2*k + 1] = 0

P1010:
    dst[r, 2*k]     = 0
    dst[r, 2*k + 1] = src[r, k]

P0001:
    dst[r, 4*k]     = src[r, k]
    dst[r, 4*k + 1] = 0
    dst[r, 4*k + 2] = 0
    dst[r, 4*k + 3] = 0

P0010:
    dst[r, 4*k]     = 0
    dst[r, 4*k + 1] = src[r, k]
    dst[r, 4*k + 2] = 0
    dst[r, 4*k + 3] = 0

P0100:
    dst[r, 4*k]     = 0
    dst[r, 4*k + 1] = 0
    dst[r, 4*k + 2] = src[r, k]
    dst[r, 4*k + 3] = 0

P1000:
    dst[r, 4*k]     = 0
    dst[r, 4*k + 1] = 0
    dst[r, 4*k + 2] = 0
    dst[r, 4*k + 3] = src[r, k]
```

For example, when one row of source data is `[a, b, c, d]`:

```text
P0101 -> [a, 0, b, 0, c, 0, d, 0]
P1010 -> [0, a, 0, b, 0, c, 0, d]
P0010 -> [0, a, 0, 0, 0, b, 0, 0, 0, c, 0, 0, 0, d, 0, 0]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `indexes` | `pto.tile_buf` | Per-element destination index tile with the same valid region as src; the index unit is the destination element (index form only). |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `axis`: The mask form must specify `"row"` or `"col"`, written after the source type of **ins**; there is no default value, and the index form does not use this attribute.
  - `"row"`: Keeps the row count and expands along the column direction within each row.
  - `"col"`: Keeps the column count and expands along the row direction; replace the column coordinates in the above formulas with row coordinates.

- `maskPattern`: Mask pattern, used only in the mask form.
  - `#pto.mask_pattern<P0101>`: Writes `src` to the 1st position of every 2 positions, and fills the remaining positions with 0.
  - `#pto.mask_pattern<P1010>`: Writes `src` to the 2nd position of every 2 positions, and fills the remaining positions with 0.
  - `#pto.mask_pattern<P0001>`: Writes `src` to the 1st position of every 4 positions, and fills the remaining positions with 0.
  - `#pto.mask_pattern<P0010>`: Writes `src` to the 2nd position of every 4 positions, and fills the remaining positions with 0.
  - `#pto.mask_pattern<P0100>`: Writes `src` to the 3rd position of every 4 positions, and fills the remaining positions with 0.
  - `#pto.mask_pattern<P1000>`: Writes `src` to the 4th position of every 4 positions, and fills the remaining positions with 0.
  - `#pto.mask_pattern<P1111>`: Full copy without inserting 0.

**Constraints:**

- Both the Atlas A3 products and the Ascend 950PR&950DT products support the index form and the mask form; **indexes** and **maskPattern** must be specified, and only one of them can be selected.
- **src**/**dst** are both located in `vec`, have the same element type, and support 8/16/32-bit integers as well as `f16`, `bf16`, and `f32`.
- Index form: indexes are located in `vec` and have the same valid shape as **src**; each dimension of **dst** has a valid size no smaller than that of **src**. 4-byte data uses 32-bit indexes, and 2-byte and 1-byte data use 16-bit indexes.
- Each index represents the target element position counted from the **dst** storage start, not a row number or byte offset. Users should provide indexes that are within range and do not conflict with each other; the operation does not perform per-value out-of-bounds checks and does not guarantee the overwrite order for duplicate target indexes.
- In the index form on the Ascend 950PR&950DT products, the target is first cleared to zero, and then the positions hit by the indexes are written; positions not hit are zero.
- Mask form: **src**/**dst** both use `blayout=row_major`. Let the expansion factor be F: `P0101/P1010` is 2, a four-bit pattern with a single 1 is 4, and `P1111` is 1.
- `axis="row"` requires the same valid row count, and `dst.valid_cols=F*src.valid_cols`; `axis="col"` requires the same valid column count, and `dst.valid_rows=F*src.valid_rows`. Positions not hit by the mask are zero-filled.

**Example:**

In the index form, if the source valid data is `[11,22,33,44]` and the corresponding indexes are `[2,0,3,1]`, the first four elements of the target are `[22,44,11,33]`. Each index corresponds to one source element.

```mlir
pto.tscatter ins(%values, %indexes :
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8,
    v_row=1, v_col=4, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>,
  !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=8,
    v_row=1, v_col=4, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  outs(%result : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8,
    v_row=1, v_col=4, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
```

```mlir
// Atlas A3 products/Ascend 950PR&950DT products: expand along the column direction of each row.
pto.tscatter ins(%src, {maskPattern = #pto.mask_pattern<P0101>} :
                 !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32,
                     v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>, "row")
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=64,
                     v_row=1, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

Expansion along the row direction of columns is specified by `"col"`. The following example expands two rows of valid data into four rows, with rows 2 and 4 zero-filled.

```mlir
pto.tscatter ins(%src, {maskPattern = #pto.mask_pattern<P0101>} :
  !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=8,
    v_row=2, v_col=8, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>, "col")
  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=8,
    v_row=4, v_col=8, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
```
