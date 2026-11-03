# Element-Wise Binary Operation

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:27:05.460Z pushedAt=2026-11-02T03:00:52.135Z nl=2 -->

This section describes the instruction names, signatures, and semantics of the PTO ISA element-wise binary operations. All operations act on local buffers (`tile_buf`, located in the `loc=vec` space) and adopt the Destination-Passing Style (DPS): the operation itself does not produce an SSA return value, but instead writes the result directly into a pre-allocated destination `tile_buf`.

This class of operations typically has the following assembly form:

```mlir
 pto.op ins(%lhs, %rhs : !pto.tile_buf<...>, !pto.tile_buf<...>)
        outs(%dst : !pto.tile_buf<...>)
```

Common constraints typically include:

- The element types of the input tiles are compatible.
- The shapes and valid-shapes of the input tiles are compatible.
- The type of the output tile matches the target semantics.

---

## Contents

- [`pto.tadd`: Element-Wise Addition](#ptotadd-element-wise-addition)
- [`pto.tand`: Element-Wise Bitwise AND](#ptotand-element-wise-bitwise-and)
- [`pto.tor`: Element-Wise Bitwise OR](#ptotor-element-wise-bitwise-or)
- [`pto.tsub`: Element-Wise Subtraction](#ptotsub-element-wise-subtraction)
- [`pto.tmul`: Element-Wise Multiplication](#ptotmul-element-wise-multiplication)
- [`pto.tmin`: Element-Wise Minimum](#ptotmin-element-wise-minimum)
- [`pto.tmax`: Element-Wise Maximum](#ptotmax-element-wise-maximum)
- [`pto.tcmp`: Element-Wise Comparison](#ptotcmp-element-wise-comparison)
- [`pto.tshl`: Element-Wise Left Shift](#ptotshl-element-wise-left-shift)
- [`pto.tshr`: Element-Wise Right Shift](#ptotshr-element-wise-right-shift)
- [`pto.txor`: Element-Wise Bitwise XOR](#ptotxor-element-wise-bitwise-xor)
- [`pto.tsel`: Mask Selection](#ptotsel-mask-selection)
- [`pto.tprelu`: Parameterized ReLU](#ptotprelu-parameterized-relu)
- [`pto.taddc`: Ternary Element-Wise Addition](#ptotaddc-ternary-element-wise-addition)
- [`pto.tsubc`: Ternary Element-Wise Subtraction and Addition](#ptotsubc-ternary-element-wise-subtraction-and-addition)

---

## Operation Details

### `pto.tadd`: Element-Wise Addition

```mlir
pto.tadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be one of the following: `i32`, `i16`, `f16`, or `f32`.
  - The tile must use row-major layout (`blayout=row_major`).

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be one of the following: `i32`, `f32`, `i16`, `f16`, `bf16`, or `i8`.
  - The tile must use the row-major layout (`blayout=row_major`).

**Example:**

```mlir
pto.tadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tand`: Element-Wise Bitwise AND

```mlir
pto.tand ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] & src1[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The element types of `src0`, `src1`, and `dst` must be consistent.
  - The shared element type must be `i8` or `i16`.
  - The three tiles must use the row-major layout (`blayout=row_major`).
  - `src0` and `src1` must each have the same valid region as `dst`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The element types of `src0`, `src1`, and `dst` must be consistent.
  - The shared element type must be `i8`, `i16`, or `i32`.
  - The three tiles must use the row-major layout (`blayout=row_major`).
  - `src0` and `src1` must each have the same valid region as `dst`.

**Example:**

```mlir
pto.tand ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tor`: Element-Wise Bitwise OR

```mlir
pto.tor ins(<src0>, <src1> : <src0_type>, <src1_type>)
        outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] | src1[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The element types of `src0`, `src1`, and `dst` must be consistent.
  - The shared element type must be `i8` or `i16`.
  - The three tiles must use the row-major layout (`blayout=row_major`).
  - `src0` and `src1` must each have the same valid region as `dst`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The element types of `src0`, `src1`, and `dst` must be consistent.
  - The shared element type must be `i8`, `i16`, or `i32`.
  - The three tiles must use the row-major layout (`blayout=row_major`).
  - `src0` and `src1` must each have the same valid region as `dst`.

**Example:**

```mlir
pto.tor ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>,
            !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>)
        outs(%c : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>)
```

---

### `pto.tsub`: Element-Wise Subtraction

```mlir
pto.tsub ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Minuend tile buffer |
| `src1` | `pto.tile_buf` | Subtrahend tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be `i32`, `i16`, `f16`, or `f32`.
  - The tile must use row-major layout (`blayout=row_major`).
  - The valid region must be within the static tile shape range.
  - `src0`, `src1`, and `dst` should have the same `validRow/validCol`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be `i32`, `i16`, `i8`, `f32`, or `f16`.
  - The tile must use row-major layout (`blayout=row_major`).
  - The valid region must be within the static tile shape range.
  - `src0`, `src1`, and `dst` should have the same `validRow/validCol`.

**Example:**

```mlir
pto.tsub ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tmul`: Element-Wise Multiplication

```mlir
pto.tmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be `i32`, `i16`, `f16`, or `f32`.
  - The tile must use `loc=vec` and row-major layout (`blayout=row_major`).
  - The valid region must be within the static tile shape range.
  - `src0`, `src1`, and `dst` should have the same `validRow/validCol`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be `i32`, `f32`, `i16`, or `f16`.
  - The tile must use `loc=vec` and row-major layout (`blayout=row_major`).
  - The valid region must be within the static tile shape range.
  - `src0`, `src1`, and `dst` should have the same `validRow/validCol`.

**Example:**

```mlir
pto.tmul ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tmin`: Element-Wise Minimum

```mlir
pto.tmin ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[i, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be `i32`, `i16`, `f16`, or `f32`.
  - The tile must use row-major layout (`blayout=row_major`).
  - The valid region must be within the static tile shape range.
  - `src0`, `src1`, and `dst` should have the same `validRow/validCol`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be `i32`, `i16`, `i8`, `f32`, `f16`, or `bf16`.
  - The tile must use the row-major layout (`blayout=row_major`).
  - The valid region must be within the static tile shape range.
  - `src0`, `src1`, and `dst` should have the same `validRow/validCol`.

**Example:**

```mlir
pto.tmin ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tmax`: Element-Wise Maximum

```mlir
pto.tmax ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[i, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be `i32`, `i16`, `f16`, or `f32`.
  - The tile must use a row-major layout (`blayout=row_major`).
  - The valid region must be within the static tile shape range.
  - `src0`, `src1`, and `dst` should have the same `validRow/validCol`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be `i32`, `i16`, `i8`, `f32`, or `f16`.
  - The tile must use the row-major layout (`blayout=row_major`).
  - The valid region must be within the static tile shape range.
  - `src0`, `src1`, and `dst` should have the same `validRow/validCol`.

**Example:**

```mlir
pto.tmax ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tcmp`: Element-Wise Comparison

```mlir
pto.tcmp ins(<src0>, <src1> {cmpMode = <mode>} : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = (src0[i, j] <cmpMode> src1[i, j]) ? 1 : 0
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First comparison input |
| `src1` | `pto.tile_buf` | Second comparison input |
| `dst` | `pto.tile_buf` | Target mask tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `cmpMode`: comparison mode. The default value is `eq`.
  - `#pto<cmp eq>`: equal to
  - `#pto<cmp ne>`: not equal to
  - `#pto<cmp lt>`: less than
  - `#pto<cmp le>`: less than or equal to
  - `#pto<cmp gt>`: greater than
  - `#pto<cmp ge>`: greater than or equal to

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The input element type must be `i32`, `f16`, or `f32`.
  - The output mask element type must be `i8`.
  - `src0`, `src1`, and `dst` must use `loc=vec`.
  - The valid region must be within the static tile shape range, and the valid regions of `src0/src1/dst` must be consistent.

- **Implementation check (Ascend 950PR&950DT products)**
  - The input element type must be `i32`, `i16`, `i8`, `f32`, `f16`, or `bf16`.

**Example:**

```mlir
pto.tcmp ins(%a, %b {cmpMode = #pto<cmp lt>} :
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%mask : !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=32,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tshl`: Element-Wise Left Shift

```mlir
pto.tshl ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] << src1[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Source tile buffer to be left-shifted |
| `src1` | `pto.tile_buf` | Tile buffer of left shift amounts |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The element types of `src0` and `src1` must be identical.
  - The shared element type must be `i8`, `i16`, or `i32`.
  - `src0`, `src1`, and `dst` must use the row-major layout (`blayout=row_major`).
  - `src0` and `src1` must each have the same valid region as `dst`.

**Example:**

```mlir
pto.tshl ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tshr`: Element-Wise Right Shift

```mlir
pto.tshr ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] >> src1[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Source tile buffer to be right-shifted |
| `src1` | `pto.tile_buf` | Tile buffer of shift amounts |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `src0` and `src1` must have the same element type.
  - The shared element type must be `i8`, `i16`, or `i32`.
  - `src0`, `src1`, and `dst` must use the row-major layout (`blayout=row_major`).
  - `src0` and `src1` must each have the same valid region as `dst`.

**Example:**

```mlir
pto.tshr ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.txor`: Element-Wise Bitwise XOR

```mlir
pto.txor ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] ^ src1[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `tmp` | `pto.tile_buf` | Temporary tile buffer; reusable placeholder in Ascend 950PR&950DT products |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The element types of `src0`, `src1`, `tmp`, and `dst` must be consistent.
  - The shared element type must be `i8` or `i16`.
  - The four tiles must use the row-major layout (`blayout=row_major`).
  - `src0`, `src1`, and `tmp` must each have the same valid region as `dst`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The element types of `src0`, `src1`, and `dst` must be consistent.
  - The shared element type must be `i8`, `i16`, or `i32`.
  - `src0`, `src1`, and `dst` must use the row-major layout (`blayout=row_major`).
  - `src0` and `src1` must have the same valid region as `dst`, respectively.
  - `tmp` serves only as a placeholder parameter in the Ascend 950PR&950DT product path.

**Example:**

```mlir
// Atlas A2 products/Atlas A3 products: require an independent tmp tile.
pto.txor ins(%src0, %src1, %tmp :
             !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)

// Ascend 950PR&950DT products: tmp can reuse dst as a placeholder.
pto.txor ins(%src0, %src1, %dst :
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tsel`: Mask Selection

```mlir
pto.tsel ins(<mask>, <src0>, <src1>, <tmp> : <mask_type>, <src0_type>, <src1_type>, <tmp_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = mask[i, j] ? src0[i, j] : src1[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `mask` | `pto.tile_buf` | Predicate mask tile buffer |
| `src0` | `pto.tile_buf` | Source tile buffer selected when mask is true |
| `src1` | `pto.tile_buf` | Source tile buffer selected when mask is false |
| `tmp` | `pto.tile_buf` | Temporary scratch tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The element types of `src0`, `src1`, and `dst` must be identical.
  - The shared element type must be `i16`, `i32`, `f16`, `bf16`, or `f32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).

- **Implementation check (Ascend 950PR&950DT products)**
  - The element types of `src0`, `src1`, and `dst` must be consistent.
  - The shared element type must be `i8`, `i16`, `i32`, `f16`, `bf16`, or `f32`.
  - `src0`, `src1`, and `dst` must use the row-major layout (`blayout=row_major`).

- **Temporary tile**
  - `tmp` is the temporary scratch tile required by the current DPS/ISA form.

**Example:**

```mlir
// Supported only on Ascend 950PR&950DT products: tmp uses a 2-byte element type (f16).
pto.tsel ins(%mask, %a, %b, %tmp :
             !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=32,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)

// Supported on both Atlas A3 products and Ascend 950PR&950DT products: tmp uses a 4-byte element type (f32).
pto.tsel ins(%mask, %a, %b, %tmp :
             !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=32,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tprelu`: Parameterized ReLU

```mlir
pto.tprelu ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
           outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] > 0 ? src0[i, j] : src1[i, j] * src0[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Input activation tile buffer |
| `src1` | `pto.tile_buf` | Element-wise negative half-axis slope tile buffer |
| `tmp` | `pto.tile_buf` | Temporary tile buffer; reusable placeholder on Ascend 950PR&950DT products |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The element types of `dst/src0/src1` must be identical, and must be `f16` or `f32`.
  - The element type of `tmp` uses `ui8`. Let R/C be the valid row/column counts of the destination: the physical row count of the temporary space must be at least R+1, the valid column count must be at least `ceil(C/8)`, and the total capacity must be at least `(R+1)*align_up(ceil(C/8),32)` bytes. The destination valid region must be of static size.
  - All related tiles must use row-major layout (`blayout=row_major`).
  - `src0` and `src1` must each have the same valid region as `dst`.
  - In Atlas A3 products, the two source tiles, the destination tile, and the temporary space must not overlap in memory.

- **Implementation check (Ascend 950PR&950DT products)**
  - The element types of `dst/src0/src1` must be consistent and must be `f16` or `f32`.
  - All related tiles must use the row-major layout (`blayout=row_major`).
  - `src0` and `src1` must each have the same valid region as `dst`.
  - `tmp` can be used as a placeholder parameter in the Ascend 950PR&950DT product path.

**Example:**

```mlir
// Atlas A2 products/Atlas A3 products: require a separate tmp tile (element type ui8).
pto.tprelu ins(%a, %slopes, %tmp :
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=ui8, rows=17, cols=32,
               v_row=17, v_col=32, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

Ascend 950PR&950DT products use the destination tile as a placeholder parameter:

```mlir
// Ascend 950PR&950DT products: tmp can reuse dst as a placeholder.
pto.tprelu ins(%a, %slopes, %c :
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

### `pto.taddc`: Ternary Element-Wise Addition

```mlir
pto.taddc ins(<src0>, <src1>, <src2> : <src0_type>, <src1_type>, <src2_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, j] + src2[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `src2` | `pto.tile_buf` | Third source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation constraints**
  - The implementation uses the valid rows/columns of `dst` as the iteration domain.
  - The specific types and valid region constraints of `src0`, `src1`, `src2`, and `dst` are subject to the verifier.

**Example:**

```mlir
pto.taddc ins(%a, %b, %c :
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%d : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### `pto.tsubc`: Ternary Element-Wise Subtraction and Addition

```mlir
pto.tsubc ins(<src0>, <src1>, <src2> : <src0_type>, <src1_type>, <src2_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, j] + src2[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Subtrahend tile buffer |
| `src2` | `pto.tile_buf` | Addend tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation constraints**
  - The implementation uses the valid rows/columns of `dst` as the iteration domain.
  - The specific types and valid region constraints of `src0`, `src1`, `src2`, and `dst` are subject to the verifier.

**Example:**

```mlir
pto.tsubc ins(%a, %b, %c :
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%d : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```
