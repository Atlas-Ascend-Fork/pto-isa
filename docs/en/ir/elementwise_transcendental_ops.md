# Element-Wise Arithmetic and Transcendental Functions

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:29:27.215Z pushedAt=2026-11-02T03:00:52.133Z nl=2 -->

This section describes the instruction names, signatures, and semantics of the PTO ISA element-wise arithmetic and transcendental function operations. All operations act on local buffers (`tile_buf`, located in the `loc=vec` space) and adopt the Destination-Passing Style (DPS): the operation itself does not produce an SSA return value; instead, it writes the result directly into a pre-allocated destination `tile_buf`.

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

- [`pto.tdiv`: Element-Wise Division](#ptotdiv-element-wise-division)
- [`pto.tlog`: Element-Wise Natural Logarithm](#ptotlog-element-wise-natural-logarithm)
- [`pto.trecip`: Element-Wise Reciprocal](#ptotrecip-element-wise-reciprocal)
- [`pto.trsqrt`: Element-Wise Reciprocal Square Root](#ptotrsqrt-element-wise-reciprocal-square-root)
- [`pto.tsqrt`: Element-Wise Square Root](#ptotsqrt-element-wise-square-root)
- [`pto.texp`: Element-Wise Exponential Function](#ptotexp-element-wise-exponential-function)
- [`pto.trem`: Element-Wise Remainder (with Temporary Tile)](#ptotrem-element-wise-remainder-with-temporary-tile)
- [`pto.tfmod`: Element-Wise Remainder (Without Temporary Tile)](#ptotfmod-element-wise-remainder-without-temporary-tile)

---

## Operation Details

### `pto.tdiv`: Element-Wise Division

```mlir
pto.tdiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
         {precisionType = <precision>}
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Dividend tile buffer |
| `src1` | `pto.tile_buf` | Divisor tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `precisionType`: division precision mode. The default value is `default`.
  - `#pto<div_precision default>`: default precision
  - `#pto<div_precision high_precision>`: high precision

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The tile element type must be `f16` or `f32`.
  - `src0`, `src1`, and `dst` must have consistent element types and use the row-major layout (`blayout=row_major`).
  - The valid range must be within the static tile shape range, and the valid ranges of the three tiles must be consistent.

- **Implementation check (Ascend 950PR&950DT products)**
  - The tile element type must be `i32`, `i16`, `f16`, or `f32`.
  - `src0`, `src1`, and `dst` must have consistent element types and use the row-major layout (`blayout=row_major`).
  - The valid range must be within the static tile shape range, and the valid ranges of the three tiles must be consistent.

- **Division-by-zero behavior**
  - The division-by-zero behavior is defined by the target implementation.

**Example:**

```mlir
pto.tdiv ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
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

### `pto.tlog`: Element-Wise Natural Logarithm

```mlir
pto.tlog ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
         {precisionType = <precision>}
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = ln(src[i, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `precisionType`: Logarithm precision mode. The default value is `default`.
  - `#pto<log_precision default>`: Default precision
  - `#pto<log_precision high_precision>`: High precision

**Constraints:**

- **NPU constraints**
  - The tile element type must be `f32` or `f16`.
  - `src` and `dst` must use `loc=vec`.
  - The valid range must be within the static tile shape range, and the valid ranges of `src` and `dst` must be consistent.
  - The tile must use the row-major layout (`blayout=row_major`).

- **Domain behavior**
  - The behavior for inputs such as `src <= 0` is implementation-defined.

**Example:**

```mlir
pto.tlog ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.trecip`: Element-Wise Reciprocal

```mlir
pto.trecip ins(<src> : <src_type>)
           outs(<dst> : <dst_type>)
           {precisionType = <precision>}
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = 1.0 / src[i, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `precisionType`: Reciprocal precision mode. The default value is `default`.
  - `#pto<recip_precision default>`: Default precision
  - `#pto<recip_precision high_precision>`: High precision

**Constraints:**

- **NPU constraints**
  - The tile element type must be `f32` or `f16`.
  - The tile must use `loc=vec` and row-major layout (`blayout=row_major`).
  - The valid range must be within the static tile shape range.
  - `src` and `dst` must have the same valid range.
  - The `TRECIP` instruction of Atlas A3 products does not support the source tile and destination tile using the same memory region.

- **Division-by-zero behavior**
  - Division-by-zero behavior is defined by the target implementation; the debug build of the CPU simulator may assert.

**Example:**

```mlir
pto.trecip ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

### `pto.trsqrt`: Element-Wise Reciprocal Square Root

```mlir
// Default precision (no tmp).
pto.trsqrt ins(<src> : <src_type>)
           outs(<dst> : <dst_type>)

// High precision (tmp required).
pto.trsqrt ins(<src>, <tmp> : <src_type>, <tmp_type>)
           outs(<dst> : <dst_type>)
           {precisionType = #pto<rsqrt_precision high_precision>}
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = 1.0 / sqrt(src[i, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer |
| `tmp` | `pto.tile_buf` (optional) | Temporary tile buffer; required only in `HighPrecision` mode, at least 32 bytes |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `precisionType`: Reciprocal square root precision mode. Default value is `default`.
  - `#pto<rsqrt_precision default>`: Default precision
  - `#pto<rsqrt_precision high_precision>`: High precision, requires the `tmp` operand.

**Constraints:**

- **Optional tmp operand**
  - `tmp` is an optional temporary tile buffer (`Optional<PTODpsType>`) and must be provided only when `precisionType = HighPrecision`.
  - When `precisionType` is the default value `Default`, `tmp` can be omitted.
  - `tmp` must be located at `loc=vec` and provide at least 32 bytes of storage space.

- **NPU constraints**
  - The tile element type must be `f32` or `f16`.
  - `src` and `dst` must use `loc=vec`.
  - The valid range must be within the static tile shape range, and the valid ranges of `src` and `dst` must be consistent.
  - The tile must use row-major layout (`blayout=row_major`).

- **Domain behavior**
  - The behavior for inputs such as `src == 0` or negative numbers is defined by the target implementation.

**Example:**

```mlir
// Default precision, no tmp.
pto.trsqrt ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)

// High precision, with tmp.
pto.trsqrt ins(%a, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
               v_row=1, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           {precisionType = #pto<rsqrt_precision high_precision>}
```

---

### `pto.tsqrt`: Element-Wise Square Root

```mlir
pto.tsqrt ins(<src> : <src_type>)
          outs(<dst> : <dst_type>)
          {precisionType = <precision>}
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = sqrt(src[i, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `precisionType`: Square root precision mode. The default value is `default`. All optional values: `#pto<sqrt_precision default>` (default precision), `#pto<sqrt_precision high_precision>` (high precision).

**Constraints:**

- **NPU constraints**
  - The tile element type must be `f32` or `f16`.
  - `src` and `dst` must use `loc=vec`.
  - The valid range must be within the static tile shape range, and the valid ranges of `src` and `dst` must be consistent.
  - The tile must use the row-major layout (`blayout=row_major`).

- **Domain behavior**
  - The behavior for cases such as negative inputs is defined by the target implementation.

**Example:**

```mlir
pto.tsqrt ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### `pto.texp`: Element-Wise Exponential Function

```mlir
pto.texp ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
         {precisionType = <precision>}
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = exp(src[i, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `precisionType`: Exponential precision mode. The default value is `default`. All optional values: `#pto<exp_precision default>` (default precision), `#pto<exp_precision high_precision>` (high precision).

**Constraints:**

- **NPU constraints**
  - The tile element type must be `f32` or `f16`.
  - `src` and `dst` must use `loc=vec`.
  - The valid range must be within the static tile shape range, and the valid ranges of `src` and `dst` must be consistent.
  - The tile must use row-major layout (`blayout=row_major`).

**Example:**

```mlir
pto.texp ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.trem`: Element-Wise Remainder (With Temporary Tile)

```mlir
pto.trem ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
         outs(<dst> : <dst_type>)
         {precisionType = <precision>}
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = fmod(src0[i, j], src1[i, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Dividend tile buffer |
| `src1` | `pto.tile_buf` | Divisor tile buffer |
| `tmp` | `pto.tile_buf` | Temporary tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `precisionType`: Remainder precision mode. The default value is `default`. All optional values: `#pto<rem_precision default>` (default precision), `#pto<rem_precision high_precision>` (high precision).

**Constraints:**

- **General checks**
  - The implementation uses the valid rows/columns of `dst` as the iteration domain.
  - The element types of `src0`, `src1`, and `dst` must be consistent.
  - The element type of `tmp` must match that of `dst`.
  - `src0`, `src1`, `tmp`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0`, `src1`, and `dst` must have the same valid range.
  - `tmp` must provide at least 1 valid row, and `tmp.validCol >= dst.validCol`.

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The element type of `src0/src1/dst` must be `i32` or `f32`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The element type of `src0/src1/dst` must be `i32`, `i16`, `f16`, or `f32`.

**Example:**

```mlir
pto.trem ins(%a, %b, %tmp :
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tfmod`: Element-Wise Remainder (Without Temporary Tile)

```mlir
pto.tfmod ins(<src0>, <src1> : <src0_type>, <src1_type>)
          outs(<dst> : <dst_type>)
          {precisionType = <precision>}
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = fmod(src0[i, j], src1[i, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | Dividend tile buffer |
| `src1` | `pto.tile_buf` | Divisor tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `precisionType`: fmod precision mode. The default value is `default`. All optional values: `#pto<fmod_precision default>` (default precision), `#pto<fmod_precision high_precision>` (high precision).

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The element types of `src0`, `src1`, and `dst` must be consistent.
  - The tile element type must be `i32`, `i16`, `f16`, or `f32`.
  - The three tiles must satisfy the shape/valid range consistency check for binary tile operations.
  - The tiles must use the row-major layout (`blayout=row_major`).

**Example:**

```mlir
pto.tfmod ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```
