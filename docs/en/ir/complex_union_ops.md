# Union Computation

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:13:33.786Z pushedAt=2026-11-02T03:00:52.120Z nl=2 -->

This section describes the instruction names, signatures, and semantics of the Union computation operations in the PTO ISA. All operations act on the local `tile_buf`.

Generic assembly form:

```mlir
pto.op ins(<src>, ... : <src_type>, ...) outs(<dst> : <dst_type>)
```
---

## Contents

- [`pto.tpartadd`: Partial Element-Wise Addition](#ptotpartadd-partial-element-wise-addition)
- [`pto.tpartmul`: Partial Element-Wise Multiplication](#ptotpartmul-partial-element-wise-multiplication)
- [`pto.tpartmax`: Partial Element-Wise Maximum](#ptotpartmax-partial-element-wise-maximum)
- [`pto.tpartmin`: Partial Element-Wise Minimum](#ptotpartmin-partial-element-wise-minimum)
- [`pto.tpartargmax`: Partial Element-Wise Maximum and Index](#ptotpartargmax-partial-element-wise-maximum-and-index)
- [`pto.tpartargmin`: Partial Element-Wise Minimum and Index](#ptotpartargmin-partial-element-wise-minimum-and-index)

---

## Operation Details

### `pto.tpartadd`: Partial Element-Wise Addition

```mlir
pto.tpartadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
             outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j) in the valid region:
    dst[i, j] = src0[i, j] + src1[i, j]

The valid region is the intersection of the valid rectangles defined by `v_row`/`v_col` of each tile;
when the valid regions of src0 and src1 differ, the behavior of the non-overlapping region is implementation-defined.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer. |
| `src1` | `pto.tile_buf` | Second source tile buffer. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The element types of `dst`/`src0`/`src1` must be consistent, and must be one of `i32`, `i16`, `f16`, or `f32`.
  - The three tiles must be rank-2.
  - It is required that at least one input's valid region is consistent with the valid region of `dst`, and the valid region of another input does not exceed the valid region of `dst`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The element types of `dst`/`src0`/`src1` must be consistent and be one of `i8`, `i16`, `i32`, `f16`, `bf16`, or `f32`.
  - The three tiles must be rank-2.
  - Only specific partial valid region patterns are supported (for example, one source equals `dst`, and the other source is smaller than `dst` in valid-rows or valid-cols).

**Example:**

```mlir
pto.tpartadd ins(%a, %b :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tpartmul`: Partial Element-Wise Multiplication

```mlir
pto.tpartmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
             outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j) in the valid region:
    dst[i, j] = src0[i, j] * src1[i, j]

The valid region is the intersection of the valid rectangles defined by `v_row`/`v_col` of each tile;
when the valid regions of src0 and src1 differ, the behavior of the non-overlapping region is implementation-defined.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer. |
| `src1` | `pto.tile_buf` | Second source tile buffer. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The element types of `dst`/`src0`/`src1` must be consistent and be one of `i32`, `i16`, `f16`, or `f32`.
  - The three tiles must be rank-2.
  - It is required that at least one input's valid region is consistent with the valid region of `dst`, and the valid region of another input does not exceed the valid region of `dst`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The element types of `dst`/`src0`/`src1` must be consistent and be one of `i8`, `i16`, `i32`, `f16`, `bf16`, or `f32`.
  - The three tiles must be rank-2.
  - It is required that the valid regions of `src0` and `src1` do not exceed the valid region of `dst` in both dimensions.

**Example:**

```mlir
pto.tpartmul ins(%src0, %src1 :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                     v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                     v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                     v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tpartmax`: Partial Element-Wise Maximum

```mlir
pto.tpartmax ins(<src0>, <src1> : <src0_type>, <src1_type>)
             outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j) in the valid region:
    dst[i, j] = max(src0[i, j], src1[i, j])

The valid region is the intersection of the valid rectangles defined by `v_row`/`v_col` of each tile;
when the valid regions of src0 and src1 differ, the behavior of the non-overlapping region is implementation-defined.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer. |
| `src1` | `pto.tile_buf` | Second source tile buffer. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The element types of `dst`/`src0`/`src1` must be consistent, and must be one of `i32`, `i16`, `f16`, or `f32`.
  - The three tiles must be rank-2, and their shapes must be consistent.
  - It is required that at least one input's valid region is consistent with the valid region of `dst`, and the valid region of another input does not exceed the valid region of `dst`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The element types of `dst`/`src0`/`src1` must be consistent, and must be one of `i8`, `i16`, `i32`, `f16`, `bf16`, or `f32`.
  - The three tiles must be rank-2, and their shapes must be consistent.
  - it is required that the valid regions of `src0` and `src1` do not exceed the valid region of `dst` in both dimensions.

**Example:**

```mlir
pto.tpartmax ins(%a, %b :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tpartmin`: Partial Element-Wise Minimum

```mlir
pto.tpartmin ins(<src0>, <src1> : <src0_type>, <src1_type>)
             outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j) in the valid region:
    dst[i, j] = min(src0[i, j], src1[i, j])

The valid region is the intersection of the valid rectangles defined by `v_row`/`v_col` of each tile;
when the valid regions of src0 and src1 differ, the behavior in the non-overlapping region is implementation-defined.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer. |
| `src1` | `pto.tile_buf` | Second source tile buffer. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The element types of `dst`/`src0`/`src1` must be consistent and be one of `i32`, `i16`, `f16`, or `f32`.
  - The three tiles must be rank-2 and have the same shape.
  - It is required that at least one input's valid region is consistent with the valid region of `dst`, and the valid region of another input does not exceed the valid region of `dst`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The element types of `dst`/`src0`/`src1` must be consistent and must be one of `i8`, `i16`, `i32`, `f16`, `bf16`, or `f32`.
  - The three tiles must be rank-2 and have consistent shapes.
  - It is required that the valid regions of `src0` and `src1` do not exceed the valid region of `dst` in both dimensions.

**Example:**

```mlir
pto.tpartmin ins(%a, %b :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tpartargmax`: Partial Element-Wise Maximum and Index

```mlir
pto.tpartargmax ins(<src0>, <src1>, <src0Idx>, <src1Idx>
                    : <src0_type>, <src1_type>, <idx0_type>, <idx1_type>)
                outs(<dst>, <dstIdx> : <dst_type>, <dstIdx_type>)
```

**Semantics:**

```text
For each element (i, j):
    if src0[i, j] >= src1[i, j]:
        dst[i, j] = src0[i, j]
        dstIdx[i, j] = src0Idx[i, j]
    else:
        dst[i, j] = src1[i, j]
        dstIdx[i, j] = src1Idx[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First value tile. |
| `src1` | `pto.tile_buf` | Second value tile. |
| `src0Idx` | `pto.tile_buf` | First index tile (`ui32`). |
| `src1Idx` | `pto.tile_buf` | Second index tile (`ui32`). |
| `dst` | `pto.tile_buf` | Output value tile. |
| `dstIdx` | `pto.tile_buf` | Output index tile (`ui32`). |

**Returns:** None. Writes to `dst` and `dstIdx` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The element types of all value tiles (src0, src1, dst) must be consistent, and must be `f16` or `f32`.
  - The element type of all index tiles (**src0Idx**, **src1Idx**, **dstIdx**) must be `ui32`.
  - All tiles must use `loc=vec`.

**Example:**

```mlir
pto.tpartargmax
    ins(%src0, %src1, %src0_idx, %src1_idx :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst, %dst_idx :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.tpartargmin`: Partial Element-Wise Minimum and Index

```mlir
pto.tpartargmin ins(<src0>, <src1>, <src0Idx>, <src1Idx>
                    : <src0_type>, <src1_type>, <idx0_type>, <idx1_type>)
                outs(<dst>, <dstIdx> : <dst_type>, <dstIdx_type>)
```

**Semantics:**

```text
For each element (i, j):
    if src0[i, j] <= src1[i, j]:
        dst[i, j] = src0[i, j]
        dstIdx[i, j] = src0Idx[i, j]
    else:
        dst[i, j] = src1[i, j]
        dstIdx[i, j] = src1Idx[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First value tile. |
| `src1` | `pto.tile_buf` | Second value tile. |
| `src0Idx` | `pto.tile_buf` | First index tile (`ui32`). |
| `src1Idx` | `pto.tile_buf` | Second index tile (`ui32`). |
| `dst` | `pto.tile_buf` | Output value tile. |
| `dstIdx` | `pto.tile_buf` | Output index tile (`ui32`). |

**Returns:** None. Writes to `dst` and `dstIdx` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The element type of all value tiles (src0, src1, dst) must be consistent, and must be `f16` or `f32`.
  - The element type of all index tiles (**src0Idx**, **src1Idx**, **dstIdx**) must be `ui32`.
  - All tiles must use `loc=vec`.

**Example:**

```mlir
pto.tpartargmin
    ins(%src0, %src1, %src0_idx, %src1_idx :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst, %dst_idx :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
```
