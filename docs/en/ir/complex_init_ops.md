# Initialization

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:07:04.893Z pushedAt=2026-11-02T03:00:52.116Z nl=2 -->

This section describes the instruction names, signatures, and semantics of the initialization operations in the PTO ISA. All operations act on the local `tile_buf`.

Generic assembly form:

```mlir
pto.op ins(<src>, ... : <src_type>, ...) outs(<dst> : <dst_type>)
```
---

## Contents

- [`pto.tci`: Consecutive Integer Sequence Generation](#ptotci-consecutive-integer-sequence-generation)
- [`pto.trandom`: Random Number Generation](#ptotrandom-random-number-generation)
- [`pto.ttri`: Triangular Mask Generation](#ptottri-triangular-mask-generation)
- [`pto.tfillpad`: Padding Region Filling](#ptotfillpad-padding-region-filling)

---

## Operation Details

### `pto.tci`: Consecutive Integer Sequence Generation

```mlir
pto.tci ins(<S> : <int_type>)
         outs(<dst> : <dst_type>)
         {descending = <bool>}
```

**Semantics:**

```text
For each column j in the single valid row:
    if descending == false:
        dst[0, j] = S + j
    else:
        dst[0, j] = S - j
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `S` | `AnyInteger` | Starting integer value. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `descending`: Indicates whether to generate a descending sequence. The default value is `false`.
  - `false`: Generates an ascending sequence (S, S+1, S+2, ...).
  - `true`: Decrements from S (S, S-1, S-2, ...).

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The `dst` element type must be an integer type, supporting 16-bit or 32-bit integers, such as `i16`, `ui16`, `i32`, and `ui32`.
  - The type of `S` must be exactly the same as the `dst` element type.
  - `dst` uses `loc=vec` and is a rank-2 single-row sequence tile; the valid row count is 1.
  - `dst.cols` cannot be 1.

**Example:**

```mlir
// Generate an i16 ascending sequence.
pto.tci ins(%c0_i16 : i16)
        outs(%tile : !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=16,
            v_row=1, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=1>)
```

When the start value is **10** and the valid column count is **4**, the ascending result is `[10,11,12,13]`, and the descending result is `[10,9,8,7]`.

```mlir
%start = pto.constant 10 : i16
pto.tci ins(%start : i16)
  outs(%sequence : !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=16,
    v_row=1, v_col=4, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>) {descending = true}
```

---

### `pto.trandom`: Random Number Generation

```mlir
pto.trandom ins(<key0>, <key1>, <counter0>, <counter1>, <counter2>, <counter3>
                : i32, i32, i32, i32, i32, i32)
            outs(<dst> : <dst_type>)
```

**Semantics:**

```text
dst = philox_random(key0, key1, counter0..counter3, rounds)
// Generate pseudo-random numbers using the Philox algorithm with a key/counter pair to fill the dst tile.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `key0` | `i32` | Key word 0. |
| `key1` | `i32` | Key word 1. |
| `counter0` | `i32` | Counter word 0. |
| `counter1` | `i32` | Counter word 1. |
| `counter2` | `i32` | Counter word 2. |
| `counter3` | `i32` | Counter word 3. |
| `dst` | `pto.tile_buf` | Destination tile buffer (`i32`/`ui32`). |

**Returns:** None. Writes to `dst` in the DPS form.

**Attributes:**

- `rounds`: Number of Philox iteration rounds. The default value is `10`.
  - `7`: 7 rounds (faster, slightly weaker randomness)
  - `10`: 10 rounds (standard)

**Constraints:**

- **Implementation check (Atlas A2/A3 products)**
  - Not supported. Available only on the Ascend 950PR&950DT products.

- **Implementation check (Ascend 950PR&950DT products)**
  - All key/counter operands must be `i32`/`ui32`.
  - The `dst` element type must be `i32` or `ui32`.
  - `dst` must use `blayout=row_major`.
  - `rounds` must be 7 or 10.

**Example:**

```mlir
pto.trandom
    ins(%k0, %k1, %c0, %c1, %c2, %c3 : i32, i32, i32, i32, i32, i32)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=4, cols=256,
                              v_row=4, v_col=256, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)

// Use 7 rounds.
pto.trandom
    ins(%k0, %k1, %c0, %c1, %c2, %c3 {rounds = 7 : i32}
        : i32, i32, i32, i32, i32, i32)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=2, cols=256,
                              v_row=2, v_col=256, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.ttri`: Triangular Mask Generation

```mlir
pto.ttri ins(<diagonal> : <integer_type>)
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    if upperOrLower == 0:  // lower triangular
        dst[i, j] = (j <= i + diagonal) ? 1 : 0
    else:                  // upper triangular
        dst[i, j] = (j >= i + diagonal) ? 1 : 0
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `diagonal` | Integer type | Diagonal offset. |
| `dst` | `pto.tile_buf` | Destination mask tile. |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `upperOrLower`: triangular type. The default value is `0`.
  - `0`: lower triangular mask
  - `1`: upper triangular mask

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The `dst` element type must be `f16`, `f32`, `i16`, `i32`, `u16`, or `u32`.
  - `dst` must use `loc=vec`.
  - `upperOrLower` must be 0 or 1.

- **Implementation check (Ascend 950PR&950DT products)**
  - The `dst` element type must be `f16`, `f32`, `bf16`, `i8`, `i16`, `i32`, `u8`, `u16`, or `u32`.
  - `dst` must use `loc=vec`.
  - `upperOrLower` must be 0 or 1.

**Example:**

```mlir
// Lower triangular mask.
pto.ttri
    ins(%diag : i32)
    outs(%lower : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
                                v_row=32, v_col=32, blayout=row_major,
                                slayout=none_box, fractal=512, pad=0>)

// Upper triangular mask.
pto.ttri
    ins(%diag {upperOrLower = 1 : i32} : i32)
    outs(%upper : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                v_row=16, v_col=16, blayout=row_major,
                                slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.tfillpad`: Padding Region Filling

```mlir
pto.tfillpad ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Semantics:** Copies the source valid region to the same coordinates in the destination, and fills the remaining physical region according to the padding policy of the destination type. This API supports equal-capacity filling, VEC destination capacity expansion, and in-place filling of the same storage.

```text
For each physical position (i,j) in dst:
    if i < src.valid_rows and j < src.valid_cols:
        dst[i,j] = src[i,j]
    else:
        dst[i,j] = padding_value(dst.pad, dst.dtype)
// When src and dst are the same tile, the source valid region keeps its original values.
```

**Parameters and returns:** **src** is the source tile, and **dst** is the pre-allocated destination tile; there is no SSA return value. The physical capacity is determined by rows/cols, and the region participating in the copy is determined by the source **v_row**/**v_col**.

**Constraints:**

- **src**/**dst** are rank-2 tiles; the element storage sizes are the same, each being 1, 2, or 4 bytes.
- The pad of **dst** cannot be `null` (0); `zero` (1), `max` (2), and `min` (3) represent the zero-fill, element-type maximum, and element-type minimum policies respectively, and the integer encoding is not an arbitrary filling value.
- The destination physical size in each dimension is not smaller than the source size; physical size expansion applies only to sources and destinations with `loc=vec`. Mismatched dynamic physical sizes are not allowed to represent expansion.
- When `loc=mat`, the source and destination tile types are the same, including the valid region and pad; the optional `padValue = #pto<pad_value zero|max|min>` attribute must be consistent with the pad of the destination type, and when omitted, the destination type policy is used. This attribute cannot be used for VEC.
- No separate mode attribute is used to select the behavior; the same tile can serve as both **ins** and **outs**, indicating in-place filling.

**Example:**

The VEC `16x16` data is expanded to a `32x32` capacity, with zero-fill outside the source region; then in-place zero-fill on the same tile is demonstrated.

```mlir
pto.tfillpad
  ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)
  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
    fractal=512, pad=1>)
```

```mlir
pto.tfillpad
  ins(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
    fractal=512, pad=1>)
  outs(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
    fractal=512, pad=1>)
```
