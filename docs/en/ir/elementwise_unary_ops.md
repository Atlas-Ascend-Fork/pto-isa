# Element-Wise Unary Operation

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:38:18.602Z pushedAt=2026-11-02T03:00:52.134Z nl=2 -->

This section describes the instruction names, signatures, and semantics of PTO ISA element-wise unary operations. All operations act on local buffers (`tile_buf`, located in the `loc=vec` space) and adopt the Destination-Passing Style (DPS): the operation itself does not produce an SSA return value; instead, it writes the result directly into a pre-allocated destination `tile_buf`.

This category of operations generally has the following assembly form:

```mlir
 pto.op ins(%src : !pto.tile_buf<...>)
        outs(%dst : !pto.tile_buf<...>)
```

The general constraints typically include:

- The element types of the input tiles are compatible.
- The shape and valid-shape of the input tiles are compatible.
- The type of the output tile matches the target semantics.

---

## Contents

- [`pto.tabs`: Element-Wise Absolute](#ptotabs-element-wise-absolute)
- [`pto.tnot`: Element-Wise Bitwise NOT](#ptotnot-element-wise-bitwise-not)
- [`pto.trelu`: ReLU Activation](#ptotrelu-relu-activation)
- [`pto.tneg`: Element-Wise Negation](#ptotneg-element-wise-negation)

---

## Operation Details

### `pto.tabs`: Element-Wise Absolute

```mlir
pto.tabs ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = abs(src[i, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **NPU constraints**
  - The tile element type must be `f32` or `f16`.
  - `src` and `dst` must use `loc=vec`.
  - The valid region must be within the static tile shape range.
  - `src` and `dst` must have the same valid region.
  - The tile must use the row-major layout (`blayout=row_major`).

**Example:**

```mlir
pto.tabs ins(%a : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tnot`: Element-Wise Bitwise NOT

```mlir
pto.tnot ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = ~src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be `i16`.
  - The element types of `src` and `dst` must be identical.
  - The tile must use `loc=vec` and a row-major layout (`blayout=row_major`).
  - The valid region must be within the static tile shape range, and the valid regions of `src` and `dst` must be identical.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be `i32`, `i16`, or `i8`.
  - The element types of `src` and `dst` must be identical.
  - The tile must use `loc=vec` and the row-major layout (`blayout=row_major`).
  - The valid region must be within the static tile shape range, and the valid regions of `src` and `dst` must be consistent.

**Example:**

```mlir
// Only Ascend 950PR&950DT products support the i32 element type.
pto.tnot ins(%a : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)

// Both Atlas A3 products and Ascend 950PR&950DT products support the i16 element type.
pto.tnot ins(%a : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.trelu`: ReLU Activation

```mlir
pto.trelu ins(<src> : <src_type>)
          outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = max(0, src[i, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The tile element type must be `f16`, `f32`, or `i32`.
  - The tile must use `loc=vec` and a row-major layout (`blayout=row_major`).
  - The valid region must be within the static tile shape range.
  - `src` and `dst` should have the same `validRow/validCol`.

**Example:**

```mlir
pto.trelu ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### `pto.tneg`: Element-Wise Negation

```mlir
pto.tneg ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = -src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be `i32`, `i16`, `f16`, or `f32`.
  - The tile must use `loc=vec`.
  - The valid region must be within the static tile shape range.
  - `src` and `dst` must have the same valid region.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be `i8`, `i16`, `i32`, `f16`, `f32`, or `bf16`.
  - The tile must use `loc=vec`.
  - The valid region must be within the static tile shape range.
  - `src` and `dst` must have at least the same valid columns, subject to the verifier.

**Example:**

```mlir
pto.tneg ins(%a : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```
