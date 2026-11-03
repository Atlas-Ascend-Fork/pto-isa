# Tile-Scalar/Tile-Immediate Operations

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:00:18.222Z pushedAt=2026-11-02T03:00:52.145Z nl=2 -->

This section describes the instruction family for element-wise operations between a tile and a scalar (or immediate) in the PTO ISA. All operations act on local buffers (`tile_buf`, located in the `loc=vec` space) and adopt the Destination-Passing Style (DPS): the operation itself does not produce an SSA return value, but instead writes the result directly into a pre-allocated destination `tile_buf`.

This class of operations typically has the following assembly form:

```mlir
 pto.op ins(%src, %scalar : !pto.tile_buf<...>, <scalar_type>)
        outs(%dst : !pto.tile_buf<...>)
```

Common constraints typically include:

- The element type of the input tile is compatible with the scalar type.
- The shape and valid-shape of the input tile and output tile are compatible.
- All tiles use `loc=vec`.

---

## Contents

- [`pto.tadds`: Tile-Scalar Addition](#ptotadds-tile-scalar-addition)
- [`pto.taddsc`: Tile-Scalar Fusion Addition](#ptotaddsc-tile-scalar-fusion-addition)
- [`pto.tands`: Tile-Scalar Bitwise AND](#ptotands-tile-scalar-bitwise-and)
- [`pto.tcmps`: Tile-Scalar Comparison](#ptotcmps-tile-scalar-comparison)
- [`pto.tdivs`: Tile-Scalar Division](#ptotdivs-tile-scalar-division)
- [`pto.texpands`: Scalar Broadcast Fill](#ptotexpands-scalar-broadcast-fill)
- [`pto.tlrelu`: Leaky ReLU](#ptotlrelu-leaky-relu)
- [`pto.tmaxs`: Tile-Scalar Maximum](#ptotmaxs-tile-scalar-maximum)
- [`pto.tmins`: Tile-Scalar Minimum](#ptotmins-tile-scalar-minimum)
- [`pto.tmuls`: Tile-Scalar Multiplication](#ptotmuls-tile-scalar-multiplication)
- [`pto.tors`: Tile-Scalar Bitwise OR](#ptotors-tile-scalar-bitwise-or)
- [`pto.trems`: Tile-Scalar Remainder](#ptotrems-tile-scalar-remainder)
- [`pto.tsels`: Tile-Scalar Mask Select](#ptotsels-tile-scalar-mask-select)
- [`pto.tshls`: Tile-Scalar Left Shift](#ptotshls-tile-scalar-left-shift)
- [`pto.tshrs`: Tile-Scalar Right Shift](#ptotshrs-tile-scalar-right-shift)
- [`pto.tsubs`: Tile-Scalar Subtraction](#ptotsubs-tile-scalar-subtraction)
- [`pto.tsubsc`: Tile-Scalar Fusion Subtraction](#ptotsubsc-tile-scalar-fusion-subtraction)
- [`pto.txors`: Tile-Scalar Bitwise XOR](#ptotxors-tile-scalar-bitwise-xor)
- [`pto.tfmods`: Tile-Scalar Floating-Point Remainder](#ptotfmods-tile-scalar-floating-point-remainder)
- [`pto.taxpy`: Tile-Scalar Fused Multiplication-Addition](#ptotaxpy-tile-scalar-fused-multiplication-addition)

---

## Operation Details

### `pto.tadds`: Tile-Scalar Addition

```mlir
pto.tadds ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[i, j] + scalar
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | `ScalarType` (signless integer/float) | Scalar value added to each element |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be one of the following: `i32`, `i16`, `f16`, or `f32`.
  - The tile must use `loc=vec`.
  - Valid region boundary: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be one of the following: `i8`, `i16`, `i32`, `f16`, `f32`, or `bf16`.
  - The tile must use `loc=vec`.
  - Valid region boundary: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

**Example:**

```mlir
pto.tadds ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.taddsc`: Tile-Scalar Fusion Addition

```mlir
pto.taddsc ins(<src0>, <scalar>, <src1> : <type0>, <scalar_type>, <type1>)
           outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] + scalar + src1[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer. |
| `scalar` | `ScalarType` (signless integer/float) | Scalar value. |
| `src1` | `pto.tile_buf` | Second source tile buffer. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src0`, `src1`, and `dst` must be PTO shaped-like types.
  - `src0`, `src1`, and `dst` must have the same shape.
  - The implementation uses `dst valid row`/`dst valid column` as the iteration domain.

**Example:**

```mlir
pto.taddsc ins(%a, %s, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tands`: Tile-Scalar Bitwise AND

```mlir
pto.tands ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[i, j] & scalar
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | `AnySignlessInteger` | Integer scalar operand |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src` and `dst` must have the same element type.
  - The element type must be an 8-bit or 16-bit unsigned integer type: `i8`, `i16`.
  - `src` and `dst` must use the row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.
  - The source tile and destination tile must not point to the same block of memory.

- **Implementation check (Ascend 950PR&950DT products)**
  - `src` and `dst` must have the same element type.
  - The element type must be an 8-bit, 16-bit, or 32-bit unsigned integer type: `i8`, `i16`, `i32`.
  - `src` and `dst` must use the row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

**Example:**

```mlir
pto.tands ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, i32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tcmps`: Tile-Scalar Comparison

```mlir
pto.tcmps ins(<src>, <scalar> {cmpMode = <mode>} : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = (src[i, j] <cmpMode> scalar) ? 1 : 0
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | `ScalarType` (signless integer / float) | Scalar value used for comparison |
| `dst` | `pto.tile_buf` | Destination tile buffer (packed predicate mask) |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `cmpMode`: Comparison mode. The default value is `EQ`.
  - `#pto<cmp eq>`: Equal to (`==`)
  - `#pto<cmp ne>`: Not equal to (`!=`)
  - `#pto<cmp lt>`: Less than (`<`)
  - `#pto<cmp le>`: Less than or equal to (`<=`)
  - `#pto<cmp gt>`: Greater than (`>`)
  - `#pto<cmp ge>`: Greater than or equal to (`>=`)

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The input tile element type must be one of the following: `i32`, `f16`, `f32`, `i16`.
  - `src` and `dst` must use `loc=vec`.
  - Valid region boundary: `src valid row <= src.rows` and `src valid column <= src.cols`.
  - `src` and `dst` must have the same valid row.

- **Implementation check (Ascend 950PR&950DT products)**
  - The input tile element type must be one of the following: `i32`, `f16`, `f32`, `i16`, `i8`.
  - `src` and `dst` must use `loc=vec`.
  - Valid region boundary: `src valid row <= src.rows`, `src valid column <= src.cols`, `dst valid row <= dst.rows`, `dst valid column <= dst.cols`.
  - `src` and `dst` must have the same valid row.

**Example:**

```mlir
pto.tcmps ins(%a, %s {cmpMode = #pto<cmp lt>} :
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f16)
          outs(%mask : !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=32,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tdivs`: Tile-Scalar Division

```mlir
// Tile / scalar
pto.tdivs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)

// Scalar / tile (reverse mode)
pto.tdivs ins(<scalar>, <src> : <scalar_type>, <src_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[i, j] / scalar    (default)
    dst[i, j] = scalar / src[i, j]    (reverse mode)
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src/scalar` | `pto.tile_buf` / scalar | Source tile buffer or scalar (depending on the operand order) |
| `scalar/src` | scalar / `pto.tile_buf` | Scalar divisor or source tile (depending on the operand order) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `precisionType`: Division precision mode. The default value is `default`.
  - `#pto<div_precision default>`: Default precision
  - `#pto<div_precision high_precision>`: High precision mode (currently valid only when the tile element type is `f16` or `f32`)

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be one of the following: `i32`, `i16`, `f16`, `f32`.
  - The tile must use `loc=vec`.
  - Valid region boundary: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be one of the following: `i8`, `i16`, `i32`, `f16`, `f32`.
  - The tile must use `loc=vec`.
  - Valid region boundary: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

- **Special behavior**
  - Division by zero behavior is target-defined; on the Ascend 950PR&950DT products, the tile/scalar form is mapped to reciprocal multiplication, and the result is `+inf` when `scalar == 0`.

**Example:**

```mlir
// tile / scalar
pto.tdivs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)

// scalar / tile (reverse mode)
pto.tdivs ins(%s, %a : f32, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.texpands`: Scalar Broadcast Fill

```mlir
pto.texpands ins(<scalar> : <scalar_type>)
              outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = scalar
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `scalar` | `ScalarType` (signless integer/float) | Scalar value to broadcast |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be one of the following: `i32`, `i16`, `f16`, or `f32`.
  - The tile must use `loc=vec` or `loc=mat`.
  - If `loc=vec`: the tile must use row-major layout (`blayout=row_major`); valid region boundaries: `valid row <= rows` and `valid column <= cols`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be one of the following: `i8`, `i16`, `i32`, `f16`, `bf16`, or `f32`.
  - The tile must use `loc=vec` or `loc=mat`.
  - If `loc=vec`: valid region boundaries: `valid row <= rows` and `valid column <= cols`.

**Example:**

```mlir
pto.texpands ins(%scalar : f32)
              outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### `pto.tlrelu`: Leaky ReLU

```mlir
pto.tlrelu ins(<src>, <slope> : <src_type>, <slope_type>)
           outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[i, j] > 0 ? src[i, j] : slope * src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `slope` | `F32` | Negative half-axis slope coefficient |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be one of the following: `f16`, `f32`.
  - The tile must use `loc=vec`.
  - Valid region boundary: `0 < valid row <= rows` and `0 < valid column <= cols`.
  - `src` and `dst` must have the same `validRow/validCol`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be one of the following: `f16`, `f32`.
  - The tile must use `loc=vec`.
  - `slope` must be of type `f32`.
  - Valid region boundary: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` must have the same `validRow/validCol`.

**Example:**

```mlir
pto.tlrelu ins(%a, %slope : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### `pto.tmaxs`: Tile-Scalar Maximum

```mlir
pto.tmaxs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = max(src[i, j], scalar)
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | `ScalarType` (signless integer/float) | Scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be one of the following: `i32`, `i16`, `f16`, `f32`.
  - The tile must use `loc=vec`.
  - Valid region boundary: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be one of the following: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`.
  - The tile must use `loc=vec`.
  - Valid region boundary: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

**Example:**

```mlir
pto.tmaxs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tmins`: Tile-Scalar Minimum

```mlir
pto.tmins ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = min(src[i, j], scalar)
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | `ScalarType` (signless integer/float) | Scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be one of the following: `i32`, `i16`, `f16`, `f32`.
  - The tile must use `loc=vec`.
  - Valid region boundary: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be one of the following: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - The tile must use `loc=vec`.
  - Valid region boundary: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

**Example:**

```mlir
pto.tmins ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tmuls`: Tile-Scalar Multiplication

```mlir
pto.tmuls ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[i, j] * scalar
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | `ScalarType` (signless integer/float) | Scalar multiplier |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be one of the following: `i32`, `i16`, `f16`, `f32`.
  - The tile must use `loc=vec`.
  - Valid region boundary: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be one of the following: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - The tile must use `loc=vec`.
  - Valid region boundary: `valid row <= rows` and `valid column <= cols`.
  - `src valid column == dst valid column`.

**Example:**

```mlir
pto.tmuls ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tors`: Tile-Scalar Bitwise OR

```mlir
pto.tors ins(<src>, <scalar> : <src_type>, <scalar_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[i, j] | scalar
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | `AnySignlessInteger` | Integer scalar operand |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src` and `dst` must have the same element type.
  - The element type must be an 8-bit or 16-bit unsigned integer type: `i8`, `i16`.
  - `src` and `dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.
  - The source tile and target tile must not point to the same block of memory.

- **Implementation check (Ascend 950PR&950DT products)**
  - `src` and `dst` must have the same element type.
  - The element type must be an 8-bit, 16-bit, or 32-bit unsigned integer type: `i8`, `i16`, `i32`.
  - `src` and `dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

**Example:**

```mlir
// Only Ascend 950PR&950DT products support: i32 element type.
pto.tors ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>, i32)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>)

// Both Atlas A3 products and Ascend 950PR&950DT products support: i16 element type.
pto.tors ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>, i16)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>)
```

---

### `pto.trems`: Tile-Scalar Remainder

```mlir
pto.trems ins(<src>, <scalar>, <tmp> : <src_type>, <scalar_type>, <tmp_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = fmod(src[i, j], scalar)
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | `ScalarType` (signless integer/float) | Scalar divisor |
| `tmp` | `pto.tile_buf` | Temporary tile buffer required by the ISA API |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The `src`/`dst` element types must match and must be `i32` or `f32`.
  - The `scalar` type must match the tile element type.
  - The `tmp` element type must match `dst`.
  - `src`/`tmp`/`dst` must use the row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same `validRow/validCol`.
  - `tmp` must provide at least `1` valid row, and `tmp.validCol >= dst.validCol`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The `src`/`dst` element types must match and must be one of the following: `i32`, `i16`, `f16`, `f32`.
  - The `scalar` type must match the tile element type.
  - The `tmp` element type must match `dst`.
  - `src`/`tmp`/`dst` must use the row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same `validRow/validCol`.
  - `tmp` must provide at least `1` valid row, and `tmp.validCol >= dst.validCol`.

- **Special Behavior**
  - Division by zero behavior is target-defined; the CPU simulator triggers an assert in debug builds.

**Example:**

```mlir
pto.trems ins(%a, %s, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32,
             !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32,
             v_row=1, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tsels`: Tile-Scalar Mask Select

```mlir
pto.tsels ins(<mask>, <src>, <tmp>, <scalar> : <mask_type>, <src_type>, <tmp_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = mask[i, j] ? src[i, j] : scalar
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `mask` | `pto.tile_buf` | Mask tile (selection predicate carrier) |
| `src` | `pto.tile_buf` | Source tile selected when the mask bit is true |
| `tmp` | `pto.tile_buf` | Temporary scratch tile required by the current DPS form |
| `scalar` | `ScalarType` | Scalar value selected when the mask bit is false |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src` and `dst` must have the same element type.
  - The element type must be a 16-bit or 32-bit type: `i16`, `i32`, `f16`, or `f32`.
  - `src` and `dst` must use the row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

- **Implementation check (Ascend 950PR&950DT products)**
  - `src` and `dst` must have the same element type.
  - The element type must be an 8-bit, 16-bit, or 32-bit type: `i8`, `i16`, `i32`, `f16`, or `f32`.
  - `src` and `dst` must use the row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

**Example:**

```mlir
pto.tsels ins(%mask, %src, %tmp, %scalar : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, i32)
          outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tshls`: Tile-Scalar Left Shift

```mlir
pto.tshls ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[i, j] << scalar
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | `AnySignlessInteger` | Shift count. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src` and `dst` must have the same element type, and the element type must be an integer type; the verifier does not restrict the integer bit width.
  - `scalar` must be an integer type; when its value is known at compile time, it must be non-negative.
  - The `valid_shape` of `src` and `dst` must fall within the static shape range.
- **Target support boundary**
  - The hardware and TileLib templates support only tiles with `loc=vec` and `blayout=row_major`.
  - The element types of the TileLib templates cover only 8/16/32-bit integers; bit widths such as `i64` pass at the verifier level, but fail during template selection because no valid template exists.
  - The TileLib templates require `src` and `dst` to have the same valid shape; when only the `valid_shape` differs (for example, two tiles with valid column counts of 64 and 32), it likewise fails because no valid template exists.

**Example:**

```mlir
pto.tshls ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, i32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tshrs`: Tile-Scalar Right Shift

```mlir
pto.tshrs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[i, j] >> scalar
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | `AnySignlessInteger` | Shift count. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src` and `dst` must have the same element type.
  - The element type must be one of the following: `i16`, `i32`.
  - `src` and `dst` must use `loc=vec`.
  - `src` and `dst` must use the row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

- **Implementation check (Ascend 950PR&950DT products)**
  - `src` and `dst` must have the same element type.
  - The element type must be one of the following: `i8`, `i16`, `i32`.
  - `src` and `dst` must use `loc=vec`.
  - `src` and `dst` must use the row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

**Example:**

```mlir
pto.tshrs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, i32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tsubs`: Tile-Scalar Subtraction

```mlir
pto.tsubs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[i, j] - scalar
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | `ScalarType` (signless integer/float) | Subtracted scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be one of the following: `i32`, `i16`, `f16`, or `f32`.
  - The tile must use `loc=vec`.
  - Valid region boundary: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be one of the following: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - The tile must use `loc=vec`.
  - Valid region boundary: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

**Example:**

```mlir
pto.tsubs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
             v_row=32, v_col=32, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tsubsc`: Tile-Scalar Fusion Subtraction

```mlir
pto.tsubsc ins(<src0>, <scalar>, <src1> : <type0>, <scalar_type>, <type1>)
           outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] - scalar + src1[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer |
| `scalar` | `ScalarType` (signless integer/float) | Subtracted scalar value |
| `src1` | `pto.tile_buf` | Second source tile buffer (value added back) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src0`, `src1`, and `dst` must be PTO shaped-like types.
  - `src0`, `src1`, and `dst` must have the same rank.
  - The implementation uses `dst valid row`/`dst valid column` as the iteration domain.

**Example:**

```mlir
pto.tsubsc ins(%a, %s, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.txors`: Tile-Scalar Bitwise XOR

```mlir
pto.txors ins(<src>, <scalar>, <tmp> : <src_type>, <scalar_type>, <tmp_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src[i, j] ^ scalar
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | `AnySignlessInteger` | Integer scalar operand |
| `tmp` | `pto.tile_buf` | Temporary scratch tile (used for computation on Atlas A2 products/Atlas A3 products; codegen may ignore it on Ascend 950PR&950DT products, but the PTO ISA operand is still required) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - `src` and `dst` must have the same element type.
  - The element type must be an 8-bit or 16-bit unsigned integer type: `i8`, `i16`.
  - `src` and `dst` must use the row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.
  - The DPS form accepts a `tmp` scratch tile. It is used for actual computation on Atlas A2 products/Atlas A3 products; codegen may ignore it on Ascend 950PR&950DT products, but the PTO ISA operand is still required.
  - The source tile and destination tile must not point to the same memory block.

- **Implementation check (Ascend 950PR&950DT products)**
  - `src` and `dst` must have the same element type.
  - The element type must be an 8-bit, 16-bit, or 32-bit unsigned integer type: `i8`, `i16`, `i32`.
  - `src` and `dst` must use the row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

**Example:**

```mlir
// Only Ascend 950PR&950DT products support the i32 element type.
pto.txors ins(%a, %s, %tmp : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, i32,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)

// Both Atlas A3 products and Ascend 950PR&950DT products support the i16 element type.
pto.txors ins(%a, %s, %tmp : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, i16,
             !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tfmods`: Tile-Scalar Floating-Point Remainder

```mlir
pto.tfmods ins(<src>, <scalar> : <src_type>, <scalar_type>)
           outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = fmod(src[i, j], scalar)
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | `ScalarType` (signless integer / float) | Floating-point scalar divisor |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src` and `dst` must have the same element type and the same valid region.
  - `src` and `dst` must use row-major layout (`blayout=row_major`).
  - The type of `scalar` must be consistent with the tile element type.
  - The element type supports only `i16`, `i32`, `f16`, and `f32`.

- **NPU constraints**
  - `src` and `dst` must be shape-compatible `loc=vec` tile buffers.
  - Division by zero behavior is undefined behavior.

**Example:**

```mlir
pto.tfmods ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, f32)
           outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.taxpy`: Tile-Scalar Fused Multiplication-Addition

```mlir
pto.taxpy ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = dst[i, j] + src[i, j] * scalar
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `scalar` | Scalar type (`f16`/`bf16`/`f32`) | Multiplication factor |
| `dst` | `pto.tile_buf` | Destination tile buffer (also serves as the accumulation input) |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The element types of `src` and `dst` must be `f16` or `f32`.
  - The element types of `src` and `dst` must be the same, or the extended combination of `src=f16` and `dst=f32` must be used.
  - All tiles must use `loc=vec`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The element types of `src` and `dst` must be `f16`, `bf16`, or `f32`.
  - The element types of `src` and `dst` must be the same, or the extended combination of `src=f16` and `dst=f32` must be used.
  - All tiles must use `loc=vec`.

**Example:**

```mlir
pto.taxpy
    ins(%src, %alpha :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        f32)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                              v_row=16, v_col=16, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```
