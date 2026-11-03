# Broadcast Operations

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:07:47.670Z pushedAt=2026-11-02T03:00:52.119Z nl=2 -->

This section describes all operations in the PTO ISA that perform broadcast along rows or columns. All operations act on local buffers (`tile_buf`, located in the `loc=vec` space) and adopt the Destination-Passing Style (DPS): the operation itself does not produce an SSA return value, but instead writes the result directly into a pre-allocated target `tile_buf`. All operations are executed on the **vector pipe** (`PIPE_V`).

This category of operations typically has the following assembly form:

```mlir
pto.op ins(%src : !pto.tile_buf<...>)
       outs(%dst : !pto.tile_buf<...>)
```

Common constraints typically include:

- All tiles must be located in `loc=vec` (VEC/UB storage space).
- Input tiles use ND-style layout (`blayout=row_major`, `slayout=none_box`).

---

## Contents

- [`pto.tcolexpand`: Column Broadcast](#ptotcolexpand-column-broadcast)
- [`pto.trowexpand`: Row Broadcast](#ptotrowexpand-row-broadcast)
- [`pto.trowexpandsub`: Row Broadcast Subtraction](#ptotrowexpandsub-row-broadcast-subtraction)
- [`pto.trowexpandmul`: Row Broadcast Multiplication](#ptotrowexpandmul-row-broadcast-multiplication)
- [`pto.trowexpanddiv`: Row Broadcast Division](#ptotrowexpanddiv-row-broadcast-division)
- [`pto.tcolexpandmax`: Column Broadcast Maximum Value](#ptotcolexpandmax-column-broadcast-maximum-value)
- [`pto.tcolexpandmin`: Column Broadcast Minimum Value](#ptotcolexpandmin-column-broadcast-minimum-value)
- [`pto.tcolexpandmul`: Column Broadcast Multiplication](#ptotcolexpandmul-column-broadcast-multiplication)
- [`pto.tcolexpandadd`: Column Broadcast Addition](#ptotcolexpandadd-column-broadcast-addition)
- [`pto.tcolexpanddiv`: Column Broadcast Division](#ptotcolexpanddiv-column-broadcast-division)
- [`pto.tcolexpandexpdif`: Column Broadcast Exponential Difference](#ptotcolexpandexpdif-column-broadcast-exponential-difference)
- [`pto.tcolexpandsub`: Column Broadcast Subtraction](#ptotcolexpandsub-column-broadcast-subtraction)
- [`pto.trowexpandadd`: Row Broadcast Addition](#ptotrowexpandadd-row-broadcast-addition)
- [`pto.trowexpandexpdif`: Row Broadcast Exponential Difference](#ptotrowexpandexpdif-row-broadcast-exponential-difference)
- [`pto.trowexpandmax`: Row Broadcast Maximum](#ptotrowexpandmax-row-broadcast-maximum)
- [`pto.trowexpandmin`: Row Broadcast Minimum](#ptotrowexpandmin-row-broadcast-minimum)

---

## Operation Details

### `pto.tcolexpand`: Column Broadcast

```mlir
pto.tcolexpand ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[0, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile, a row vector, where each column carries one logical scalar. |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src` and `dst` must use `loc=vec`.
  - `src` and `dst` must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
  - Element types must be consistent: `dst_type == src_type`.
  - Data types: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`
  - `src valid column == dst valid column`

**Example:**

```mlir
pto.tcolexpand ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                   v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.trowexpand`: Row Broadcast

```mlir
pto.trowexpand ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[i, 0]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile, a column vector, where each row carries one logical scalar. |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src` and `dst` must use `loc=vec`.
  - `src` must use `slayout=none_box`.
  - `dst` must use an ND-style layout (`blayout=row_major`, `slayout=none_box`).
  - Element types must be consistent: `dst_type == src_type`.
  - Data types: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`
  - `src valid row == dst valid row`
  - `src valid row != 0`, `src valid column != 0`, `dst valid row != 0`, and `dst valid column != 0`

**Example:**

```mlir
pto.trowexpand ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=8,
                   v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.trowexpandsub`: Row Broadcast Subtraction

```mlir
pto.trowexpandsub ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, 0]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Primary source tile. |
| `src1` | `pto.tile_buf` | Per-row scalar vector (subtrahend broadcast source). |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i16`, `i32`, `f16`, `f32`
  - `dst` uses `blayout=row_major`.
  - `src0` and `dst` have the same shape and **valid_shape**.
  - Optional `tmp` operand: used for overloads that require **tmp** in pto-isa.

- **Implementation check (Ascend 950PR&950DT products)**
  - `src0`, `src1`, and `dst` have consistent element types.
  - Data types: `i8`, `i16`, `i32`, `f16`, `f32`
  - `dst` uses `blayout=row_major`.
  - `src0` and `dst` have the same shape and **valid_shape**.
  - Optional `tmp` operand: used for overloads that require **tmp** in pto-isa.

**Example:**

```mlir
pto.trowexpandsub ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandmul`: Row Broadcast Multiplication

```mlir
pto.trowexpandmul ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[i, 0]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Primary source tile. |
| `src1` | `pto.tile_buf` | Per-row scalar vector (multiplier broadcast source). |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `f16`, `f32`
  - `dst` uses `blayout=row_major`.
  - Optional `tmp` operand: used for overloads that require **tmp** in pto-isa.

**Example:**

```mlir
pto.trowexpandmul ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpanddiv`: Row Broadcast Division

```mlir
// Default precision.
pto.trowexpanddiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)

// High precision (requires tmp).
pto.trowexpanddiv ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
                  outs(<dst> : <dst_type>)
                  {precisionType = #pto<div_precision high_precision>}
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[i, 0]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Primary source tile (dividend). |
| `src1` | `pto.tile_buf` | Per-row scalar vector (divisor broadcast source). |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `precisionType`: division precision mode. The default value is `#pto<div_precision default>`.
  - `#pto<div_precision default>`: standard precision division.
  - `#pto<div_precision high_precision>`: high-precision division, which requires a floating-point element type and an additional `tmp` operand.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Only floating-point types are supported: `f16`, `f32`.
  - `dst` uses `blayout=row_major`.
  - In high-precision mode, the `tmp` operand must be provided.

**Example:**

```mlir
// Default precision.
pto.trowexpanddiv ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)

// High precision.
pto.trowexpanddiv ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  {precisionType = #pto<div_precision high_precision>}
```

---

### `pto.tcolexpandmax`: Column Broadcast Maximum Value

```mlir
pto.tcolexpandmax ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[0, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Primary source tile. |
| `src1` | `pto.tile_buf` | Per-column scalar vector. |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` have the same shape and **valid_shape**.
  - `src0`, `src1`, and `dst` use `blayout=row_major`.
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **Implementation check (Ascend 950PR&950DT Products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i8`, `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` have the same shape and **valid_shape**.
  - `src0`, `src1`, and `dst` use `blayout=row_major`.
  - `src1 valid_shape[1] == dst valid_shape[1]`

**Example:**

```mlir
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandmin`: Column Broadcast Minimum Value

```mlir
pto.tcolexpandmin ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[0, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Primary source tile. |
| `src1` | `pto.tile_buf` | Per-column scalar vector. |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` have the same shape and valid_shape.
  - `src0`, `src1`, and `dst` use `blayout=row_major`.
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **Implementation check (Ascend 950PR&950DT Products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i8`, `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` have the same shape and **valid_shape**.
  - `src0`, `src1`, and `dst` use `blayout=row_major`.
  - `src1 valid_shape[1] == dst valid_shape[1]`

**Example:**

```mlir
pto.tcolexpandmin ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandmul`: Column Broadcast Multiplication

```mlir
pto.tcolexpandmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[0, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Primary source tile. |
| `src1` | `pto.tile_buf` | Per-column scalar vector. |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` have the same shape and valid_shape.
  - `src0`, `src1`, and `dst` use `blayout=row_major`.
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **Implementation check (Ascend 950PR&950DT Products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i8`, `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` have the same shape and **valid_shape**.
  - `src0`, `src1`, and `dst` use `blayout=row_major`.
  - `src1 valid_shape[1] == dst valid_shape[1]`

**Example:**

```mlir
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandadd`: Column Broadcast Addition

```mlir
pto.tcolexpandadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[0, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Primary source tile. |
| `src1` | `pto.tile_buf` | Per-column scalar vector. |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` have the same shape and valid_shape.
  - `src0`, `src1`, and `dst` use `blayout=row_major`.
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **Implementation check (Ascend 950PR&950DT Products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i8`, `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` have the same shape and **valid_shape**.
  - `src0`, `src1`, and `dst` use `blayout=row_major`.
  - `src1 valid_shape[1] == dst valid_shape[1]`

**Example:**

```mlir
pto.tcolexpandadd ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpanddiv`: Column Broadcast Division

```mlir
pto.tcolexpanddiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
                  {precisionType = <precision>}
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[0, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Primary source tile (dividend). |
| `src1` | `pto.tile_buf` | Per-column scalar vector (divisor). |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `precisionType`: division precision mode. The default value is `#pto<div_precision default>`.
  - `#pto<div_precision default>`: standard precision division.
  - `#pto<div_precision high_precision>`: high precision division, valid only when the element type is `f16` or `f32`.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Only floating-point types are supported: `f16` and `f32`.
  - `src0` and `dst` have the same shape and **valid_shape**.
  - `src0`, `src1`, and `dst` use `blayout=row_major`.
  - `src1 valid_shape[1] == dst valid_shape[1]`

**Example:**

```mlir
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandexpdif`: Column Broadcast Exponential Difference

```mlir
pto.tcolexpandexpdif ins(<src0>, <src1> : <src0_type>, <src1_type>)
                     outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = exp(src0[i, j] - src1[0, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Primary source tile. |
| `src1` | `pto.tile_buf` | Per-column scalar vector (the subtrahend in the exponential difference). |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Only floating-point types are supported: `f16`, `f32`.
  - `src0` and `dst` have the same shape and **valid_shape**.
  - `src0`, `src1`, and `dst` use `blayout=row_major`.
  - `src1 valid_shape[1] == dst valid_shape[1]`

**Example:**

```mlir
pto.tcolexpandexpdif ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>,
                         !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                         v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>)
                     outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>)
```

---

### `pto.tcolexpandsub`: Column Broadcast Subtraction

```mlir
pto.tcolexpandsub ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[0, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Primary source tile. |
| `src1` | `pto.tile_buf` | Per-column scalar vector (subtrahend broadcast source). |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` have the same shape and **valid_shape**.
  - `src0`, `src1`, and `dst` use `blayout=row_major`.
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **Implementation check (Ascend 950PR&950DT products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i8`, `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` have the same shape and **valid_shape**.
  - `src0`, `src1`, and `dst` use `blayout=row_major`.
  - `src1 valid_shape[1] == dst valid_shape[1]`

**Example:**

```mlir
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandadd`: Row Broadcast Addition

```mlir
pto.trowexpandadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, 0]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Primary source tile. |
| `src1` | `pto.tile_buf` | Per-row scalar vector. |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` have the same shape and **valid_shape**.
  - `src0` and `dst` use `blayout=row_major`.
  - `src1 valid_shape[0] == dst valid_shape[0]`
  - When `src1` is row_major: `src1 valid_shape[1] == 32 / sizeof(dtype)`; otherwise: `src1 valid_shape[1] == 1`

- **Implementation check (Ascend 950PR&950DT products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i8`, `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` have the same shape and **valid_shape**.
  - `src0` and `dst` use `blayout=row_major`.
  - `src1 valid_shape[0] == dst valid_shape[0]`
  - When `src1` is row_major: `src1 valid_shape[1] == 32 / sizeof(dtype)`; otherwise: `src1 valid_shape[1] == 1`

**Example:**

```mlir
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandexpdif`: Row Broadcast Exponential Difference

```mlir
pto.trowexpandexpdif ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                     outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = exp(src0[i, j] - src1[i, 0])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Primary source tile. |
| `src1` | `pto.tile_buf` | Per-row scalar vector (the subtrahend in the exponential difference). |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Only floating-point types are supported: `f16`, `f32`.
  - `dst` uses `blayout=row_major`.
  - Optional `tmp` operand: used for overloads that require **tmp** in pto-isa.

**Example:**

```mlir
pto.trowexpandexpdif ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>,
                         !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                         v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                         fractal=512, pad=0>)
                     outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>)
```

---

### `pto.trowexpandmax`: Row Broadcast Maximum

```mlir
pto.trowexpandmax ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[i, 0])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Primary source tile. |
| `src1` | `pto.tile_buf` | Per-row scalar vector. |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i16`, `i32`, `f16`, `f32`
  - `dst` uses `blayout=row_major`.
  - Optional `tmp` operand: used for overloads that require **tmp** in pto-isa.

- **Implementation check (Ascend 950PR&950DT products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`
  - `dst` uses `blayout=row_major`.
  - Optional `tmp` operand: used for overloads that require **tmp** in pto-isa.

**Example:**

The per-row scalar vector in the example uses a column-major single-column tile. If a row-major vector is used instead, its valid column count must be `32/sizeof(dtype)`, which is 8 for f32.

```mlir
pto.trowexpandmax ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandmin`: Row Broadcast Minimum

```mlir
pto.trowexpandmin ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[i, 0])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Primary source tile. |
| `src1` | `pto.tile_buf` | Per-row scalar vector. |
| `dst` | `pto.tile_buf` | Target tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2/A3 products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i16`, `i32`, `f16`, `f32`
  - `dst` uses `blayout=row_major`.
  - Optional `tmp` operand: used for overloads that require **tmp** in pto-isa.

- **Implementation check (Ascend 950PR&950DT products)**
  - `src0`, `src1`, and `dst` have the same element type.
  - Data types: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`
  - `dst` uses `blayout=row_major`.
  - Optional `tmp` operand: used for overloads that require **tmp** in pto-isa.

**Example:**

The per-row scalar vector in the example uses a column-major single-column tile. If a row-major vector is used instead, its valid column count must be `32/sizeof(dtype)`, which is 8 for f32.

```mlir
pto.trowexpandmin ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```
