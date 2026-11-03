# Matrix Computation Operations

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:40:50.887Z pushedAt=2026-11-02T03:00:52.138Z nl=2 -->

This section describes the instruction names, signatures, and semantics of all matrix computation operations in the PTO ISA. Matrix computation operations execute on the Cube (matrix) pipe to perform matrix multiplication (TMATMUL) and matrix-vector multiplication (TGEMV). All operations adopt the Destination-Passing Style (DPS): the operation itself does not produce an SSA return value; instead, it writes the result directly into the pre-allocated destination `tile_buf`.

This category of operations generally has the following assembly form:

```mlir
 pto.op ins(%lhs, %rhs : !pto.tile_buf<...>, !pto.tile_buf<...>)
        outs(%dst : !pto.tile_buf<...>)
```

The general constraints typically include:

- The left matrix tile is located at `loc=left` (L0A buffer).
- The right matrix tile is located at `loc=right` (L0B buffer).
- The destination accumulator tile is located at `loc=acc` (L0C buffer).
- Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, `rhs.cols == dst.cols`
- All operations execute on the matrix pipe (`PIPE_M`)
- **(Ascend 950PR&950DT products)** Layout constraints: `lhs.blayout=col_major, lhs.slayout=row_major`; `rhs.blayout=row_major, rhs.slayout=col_major`; `dst.blayout=col_major, dst.slayout=row_major`. Atlas A2 products/Atlas A3 products do not enforce `blayout`/`slayout` validation at the IR level

---

## Contents

- [`pto.tmatmul`: Matrix Multiplication](#ptotmatmul-matrix-multiplication)
- [`pto.tmatmul.acc`: Matrix Multiplication-Addition (Accumulation)](#ptotmatmulacc-matrix-multiplication-addition-accumulation)
- [`pto.tmatmul.bias`: Matrix Multiplication With Bias Addition](#ptotmatmulbias-matrix-multiplication-with-bias-addition)
- [`pto.tmatmul.mx`: Mixed-Precision Matrix Multiplication](#ptotmatmulmx-mixed-precision-matrix-multiplication)
- [`pto.tmatmul.mx.acc`: Mixed-Precision Matrix Multiplication-Addition](#ptotmatmulmxacc-mixed-precision-matrix-multiplication-addition)
- [`pto.tmatmul.mx.bias`: Mixed-Precision Matrix Multiplication With Bias Addition](#ptotmatmulmxbias-mixed-precision-matrix-multiplication-with-bias-addition)
- [`pto.tgemv`: Matrix-Vector Multiplication](#ptotgemv-matrix-vector-multiplication)
- [`pto.tgemv.acc`: Matrix-Vector Multiplication-Addition (Accumulation)](#ptotgemvacc-matrix-vector-multiplication-addition-accumulation)
- [`pto.tgemv.bias`: Matrix-Vector Multiplication With Bias Addition](#ptotgemvbias-matrix-vector-multiplication-with-bias-addition)
- [`pto.tgemv.mx`: Mixed-Precision Matrix-Vector Multiplication](#ptotgemvmx-mixed-precision-matrix-vector-multiplication)
- [`pto.tgemv.mx.acc`: Mixed-Precision Matrix-Vector Multiplication-Addition](#ptotgemvmxacc-mixed-precision-matrix-vector-multiplication-addition)
- [`pto.tgemv.mx.bias`: Mixed-Precision Matrix-Vector Multiplication With Bias Addition](#ptotgemvmxbias-mixed-precision-matrix-vector-multiplication-with-bias-addition)

---

## Operation Details

### `pto.tmatmul`: Matrix Multiplication

```mlir
pto.tmatmul ins(<lhs>, <rhs> : <lhs_type>, <rhs_type>)
            outs(<dst> : <dst_type>)
            {accPhase = <phase>}
```

**Semantics:**

```text
For each (i, j):
    dst[i, j] = sum_k lhs[i, k] * rhs[k, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | Left matrix tile buffer (`loc=left`, L0A) |
| `rhs` | `pto.tile_buf` | Right matrix tile buffer (`loc=right`, L0B) |
| `dst` | `pto.tile_buf` | Destination accumulator tile buffer (`loc=acc`, L0C) |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `accPhase`: Accumulation phase control. The default value is `unspecified`.
  - `#pto<acc_phase unspecified>`: Default behavior, with an implementation-defined accumulation strategy
  - `#pto<acc_phase partial>`: Partial accumulation, where the result is an intermediate value and will continue to accumulate
  - `#pto<acc_phase final>`: Final accumulation, marking this as the last accumulation

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - Supported `(dst element type, lhs element type, rhs element type)` combinations: `(i32, i8, i8)`, `(f32, f16, f16)`, `(f32, f32, f32)`, `(f32, bf16, bf16)`.
  - Runtime constraint: `m/k/n` (respectively taken from `lhs valid row`, `lhs valid column`, `rhs valid column`) must be in the range `[1, 4095]`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The element type of `dst` must be `i32` or `f32`.
  - If `dst` is `i32`, the element types of both `lhs` and `rhs` must be `i8`.
  - If `dst` is `f32`, the element types of `lhs`/`rhs` support `f16`, `bf16`, `f32`, and some fp8 type pairs (target-defined).
  - Runtime constraint: `m/k/n` (respectively taken from `lhs valid row`, `lhs valid column`, `rhs valid column`) must be in the range `[1, 4095]`.

**Example:**

```mlir
pto.tmatmul ins(%lhs, %rhs :
                !pto.tile_buf<loc=left, dtype=f16, rows=32, cols=32,
                    v_row=32, v_col=32, blayout=col_major,
                    slayout=row_major, fractal=512, pad=0>,
                !pto.tile_buf<loc=right, dtype=f16, rows=32, cols=32,
                    v_row=32, v_col=32, blayout=row_major,
                    slayout=col_major, fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                    v_row=32, v_col=32, blayout=col_major,
                    slayout=row_major, fractal=1024, pad=0>)
```

---

### `pto.tmatmul.acc`: Matrix Multiplication-Addition (Accumulation)

```mlir
pto.tmatmul.acc ins(<acc_in>, <lhs>, <rhs> : <acc_in_type>, <lhs_type>, <rhs_type>)
                outs(<dst> : <dst_type>)
                {accPhase = <phase>}
```

**Semantics:**

```text
For each (i, j):
    dst[i, j] = acc_in[i, j] + sum_k lhs[i, k] * rhs[k, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `acc_in` | `pto.tile_buf` | Previous accumulator value (`loc=acc`) |
| `lhs` | `pto.tile_buf` | Left matrix tile buffer (`loc=left`) |
| `rhs` | `pto.tile_buf` | Right matrix tile buffer (`loc=right`) |
| `dst` | `pto.tile_buf` | Destination accumulator tile buffer (`loc=acc`) |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `accPhase`: Accumulation phase control. The default value is `unspecified`.
  - `#pto<acc_phase unspecified>`: Default behavior, with an implementation-defined accumulation strategy
  - `#pto<acc_phase partial>`: Partial accumulation, where the result is an intermediate value and will continue to accumulate subsequently
  - `#pto<acc_phase final>`: Final accumulation, marking this as the last accumulation

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The type combination constraints of `lhs`, `rhs`, and `dst` are the same as those of `pto.tmatmul`.
  - `acc_in` must be located at `loc=acc`, and its element type and shape must be consistent with `dst`.
  - The runtime constraints are the same as those of `pto.tmatmul`.

**Example:**

```mlir
pto.tmatmul.acc ins(%acc_in, %lhs, %rhs :
                    !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                        v_row=32, v_col=32, blayout=col_major,
                        slayout=row_major, fractal=1024, pad=0>,
                    !pto.tile_buf<loc=left, dtype=f16, rows=32, cols=32,
                        v_row=32, v_col=32, blayout=col_major,
                        slayout=row_major, fractal=512, pad=0>,
                    !pto.tile_buf<loc=right, dtype=f16, rows=32, cols=32,
                        v_row=32, v_col=32, blayout=row_major,
                        slayout=col_major, fractal=512, pad=0>)
                outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                        v_row=32, v_col=32, blayout=col_major,
                        slayout=row_major, fractal=1024, pad=0>)
```

---

### `pto.tmatmul.bias`: Matrix Multiplication With Bias Addition

```mlir
pto.tmatmul.bias ins(<lhs>, <rhs>, <bias> : <lhs_type>, <rhs_type>, <bias_type>)
                 outs(<dst> : <dst_type>)
                 {accPhase = <phase>}
```

**Semantics:**

```text
For each (i, j):
    dst[i, j] = sum_k lhs[i, k] * rhs[k, j] + bias[0, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | Left matrix tile buffer (`loc=left`) |
| `rhs` | `pto.tile_buf` | Right matrix tile buffer (`loc=right`) |
| `bias` | `pto.tile_buf` | Bias tile buffer (`loc=bias`, `rows=1`) |
| `dst` | `pto.tile_buf` | Destination accumulator tile buffer (`loc=acc`) |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `accPhase`: Accumulation phase control. The default value is `unspecified`.
  - `#pto<acc_phase unspecified>`: Default behavior, with the accumulation strategy implementation-defined
  - `#pto<acc_phase partial>`: Partial accumulation, where the result is in an intermediate value and will continue to accumulate subsequently
  - `#pto<acc_phase final>`: Final accumulation, marking this accumulation as the last one

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The type combination constraints of `lhs`, `rhs`, and `dst` are the same as those of `pto.tmatmul`.
  - The element type of `bias` must be consistent with the element type of `dst`.
  - `bias` must use `loc=bias`, and `rows=1`.
  - The runtime constraints are the same as those of `pto.tmatmul`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The type combination constraints of `lhs`, `rhs`, and `dst` are the same as those of `pto.tmatmul`.
  - The element type of `bias` must be the same as the element type of `dst`.
  - `bias` must use `loc=bias`, `rows=1`, and `blayout=row_major`.
  - The runtime constraints are the same as those of `pto.tmatmul`.

**Example:**

```mlir
pto.tmatmul.bias ins(%lhs, %rhs, %bias :
                     !pto.tile_buf<loc=left, dtype=f16, rows=32, cols=32,
                         v_row=32, v_col=32, blayout=col_major,
                         slayout=row_major, fractal=512, pad=0>,
                     !pto.tile_buf<loc=right, dtype=f16, rows=32, cols=32,
                         v_row=32, v_col=32, blayout=row_major,
                         slayout=col_major, fractal=512, pad=0>,
                     !pto.tile_buf<loc=bias, dtype=f32, rows=1, cols=32,
                         v_row=1, v_col=32, blayout=row_major,
                         slayout=none_box, fractal=512, pad=0>)
                 outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                         v_row=32, v_col=32, blayout=col_major,
                         slayout=row_major, fractal=1024, pad=0>)
```

---

### `pto.tmatmul.mx`: Mixed-Precision Matrix Multiplication

```mlir
pto.tmatmul.mx ins(<lhs>, <lhs_scale>, <rhs>, <rhs_scale> :
                   <lhs_type>, <lhs_scale_type>, <rhs_type>, <rhs_scale_type>)
               outs(<dst> : <dst_type>)
               {accPhase = <phase>}
```

**Semantics:**

```text
k_group = floor(k / 32)

For each (i, j):
    dst[i, j] =
        sum_k (lhs[i, k] * lhs_scale[i, k_group]) *
              (rhs[k, j] * rhs_scale[k_group, j])
```

`lhs_scale` and `rhs_scale` are MX scaling factor tiles. The K dimension is the reduction dimension of the matrix multiplication, that is, the column dimension of `lhs` and the row dimension of `rhs`; the `sum_k` in the formula above accumulates along this dimension. Scaling is grouped along the K dimension, with every 32 consecutive K elements sharing one scaling factor: `lhs_scale[i, g]` scales `lhs[i, 32*g .. 32*g+31]`, and `rhs_scale[g, j]` scales `rhs[32*g .. 32*g+31, j]`.

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | Left matrix tile buffer (`loc=left`) |
| `lhs_scale` | `pto.tile_buf` | Left matrix scaling tile buffer (`loc=scaling`) |
| `rhs` | `pto.tile_buf` | Right matrix tile buffer (`loc=right`) |
| `rhs_scale` | `pto.tile_buf` | Right matrix scaling tile buffer (`loc=scaling`) |
| `dst` | `pto.tile_buf` | Destination accumulator tile buffer (`loc=acc`) |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `accPhase`: Accumulation phase control. The default value is `unspecified`.
  - `#pto<acc_phase unspecified>`: Default behavior, with an implementation-defined accumulation strategy
  - `#pto<acc_phase partial>`: Partial accumulation, the result is in an intermediate state and will continue to accumulate subsequently
  - `#pto<acc_phase final>`: Final accumulation, marking this accumulation as the last one

**Constraints:**

- **Atlas A2 products/Atlas A3 products do not support this operation.**
- **Ascend 950PR&950DT product constraints**
  - `lhs`/`rhs` must be located at `left`/`right`, the destination at `acc`, and must satisfy the matrix shape and layout constraints in this chapter; the `dst` element type is `f32`.
  - The examples in this section use FP8 type pairs of `f8E4M3FN` or `f8E5M2`; the following size relationships use the physical and valid extents recorded in the data types respectively.
  - The physical column count of the left data and the physical row count of the right data must be positive integer multiples of 64; the valid `M/K/N` must be in `[1,4095]`.
  - Both scales are located at `scaling`. Calculated separately for the physical and valid sizes: the left scale is `[M,ceil(K/32)]`, and the right scale is `[ceil(K/32),N]`.
  - The left scale uses `blayout=row_major, slayout=row_major`; the right scale uses `blayout=col_major, slayout=col_major`; both use `fractal=32`.

**Example:**

Ascend 950PR&950DT product example: the valid region equals the physical size, M=16, K=64, N=16; the left scale is `16x2`, and the right scale is `2x16`. The input data and scaling factors are already prepared in the corresponding tiles.

```mlir
pto.tmatmul.mx ins(%a, %a_scale, %b, %b_scale :
    !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=16, cols=64,
        v_row=16, v_col=64, blayout=col_major, slayout=row_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=16, cols=2,
        v_row=16, v_col=2, blayout=row_major, slayout=row_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=right, dtype=f8E4M3FN, rows=64, cols=16,
        v_row=64, v_col=16, blayout=row_major, slayout=col_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=2, cols=16,
        v_row=2, v_col=16, blayout=col_major, slayout=col_major,
        fractal=32, pad=0>)
  outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16,
      v_row=16, v_col=16, blayout=col_major, slayout=row_major,
      fractal=512, pad=0>)
```

---

### `pto.tmatmul.mx.acc`: Mixed-Precision Matrix Multiplication-Addition

```mlir
pto.tmatmul.mx.acc ins(<acc_in>, <lhs>, <lhs_scale>, <rhs>, <rhs_scale> :
                       <acc_in_type>, <lhs_type>, <lhs_scale_type>,
                       <rhs_type>, <rhs_scale_type>)
                   outs(<dst> : <dst_type>)
                   {accPhase = <phase>}
```

**Semantics:**

```text
k_group = floor(k / 32)

For each (i, j):
    dst[i, j] = acc_in[i, j] +
        sum_k (lhs[i, k] * lhs_scale[i, k_group]) *
              (rhs[k, j] * rhs_scale[k_group, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `acc_in` | `pto.tile_buf` | Previous accumulator value (`loc=acc`) |
| `lhs` | `pto.tile_buf` | Left matrix tile buffer (`loc=left`) |
| `lhs_scale` | `pto.tile_buf` | Left matrix scaling tile buffer (`loc=scaling`) |
| `rhs` | `pto.tile_buf` | Right matrix tile buffer (`loc=right`) |
| `rhs_scale` | `pto.tile_buf` | Right matrix scaling tile buffer (`loc=scaling`) |
| `dst` | `pto.tile_buf` | Destination accumulator tile buffer (`loc=acc`) |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `accPhase`: Accumulation phase control. The default value is `unspecified`.
  - `#pto<acc_phase unspecified>`: Default behavior, with the accumulation strategy implementation-defined
  - `#pto<acc_phase partial>`: Partial accumulation, where the result is in an intermediate value and will continue to accumulate later
  - `#pto<acc_phase final>`: Final accumulation, marking this accumulation as the last one

**Constraints:**

- **Implementation check (Ascend 950PR&950DT products)**
  - The constraints on `lhs`, `rhs`, `lhs_scale`, `rhs_scale`, and `dst` are the same as those for `pto.tmatmul.mx`.
  - `acc_in` must be located at `loc=acc`, and its element type and shape must be consistent with `dst`.
  - The runtime constraints are the same as those for `pto.tmatmul.mx`.

**Example:**

Ascend 950PR&950DT products example: the valid region equals the physical size, with M=16, K=64, and N=32; the left scale is `16x2`, and the right scale is `2x32`. The input data and scaling factors are already prepared in the corresponding tiles.

```mlir
pto.tmatmul.mx.acc ins(%c_in, %a, %a_scale, %b, %b_scale :
    !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=32,
        v_row=16, v_col=32, blayout=col_major, slayout=row_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=16, cols=64,
        v_row=16, v_col=64, blayout=col_major, slayout=row_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=16, cols=2,
        v_row=16, v_col=2, blayout=row_major, slayout=row_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=right, dtype=f8E4M3FN, rows=64, cols=32,
        v_row=64, v_col=32, blayout=row_major, slayout=col_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=2, cols=32,
        v_row=2, v_col=32, blayout=col_major, slayout=col_major,
        fractal=32, pad=0>)
  outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=32,
      v_row=16, v_col=32, blayout=col_major, slayout=row_major,
      fractal=512, pad=0>)
```

---

### `pto.tmatmul.mx.bias`: Mixed-Precision Matrix Multiplication With Bias Addition

```mlir
pto.tmatmul.mx.bias ins(<lhs>, <lhs_scale>, <rhs>, <rhs_scale>, <bias> :
                        <lhs_type>, <lhs_scale_type>, <rhs_type>,
                        <rhs_scale_type>, <bias_type>)
                    outs(<dst> : <dst_type>)
```

**Semantics:**

```text
k_group = floor(k / 32)

For each (i, j):
    dst[i, j] =
        sum_k (lhs[i, k] * lhs_scale[i, k_group]) *
              (rhs[k, j] * rhs_scale[k_group, j]) + bias[0, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | Left matrix tile buffer (`loc=left`) |
| `lhs_scale` | `pto.tile_buf` | Left matrix scaling tile buffer (`loc=scaling`) |
| `rhs` | `pto.tile_buf` | Right matrix tile buffer (`loc=right`) |
| `rhs_scale` | `pto.tile_buf` | Right matrix scaling tile buffer (`loc=scaling`) |
| `bias` | `pto.tile_buf` | Bias tile buffer (`loc=bias`, `rows=1`) |
| `dst` | `pto.tile_buf` | Destination accumulator tile buffer (`loc=acc`) |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Ascend 950PR&950DT products)**
  - The constraints on `lhs`, `rhs`, `lhs_scale`, `rhs_scale`, and `dst` are the same as those for `pto.tmatmul.mx`.
  - The element type of `bias` must be `f32`.
  - `bias` must use `loc=bias`, `rows=1`, and `blayout=row_major`.
  - The runtime constraints are the same as those for `pto.tmatmul.mx`.

**Example:**

Ascend 950PR&950DT product example: the valid region equals the physical size, with M=16, K=64, and N=16; the left scale is `16x2` and the right scale is `2x16`. The input data and scaling factors are already prepared in the corresponding tiles.

```mlir
pto.tmatmul.mx.bias ins(%a, %a_scale, %b, %b_scale, %bias :
    !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=16, cols=64,
        v_row=16, v_col=64, blayout=col_major, slayout=row_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=16, cols=2,
        v_row=16, v_col=2, blayout=row_major, slayout=row_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=right, dtype=f8E4M3FN, rows=64, cols=16,
        v_row=64, v_col=16, blayout=row_major, slayout=col_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=2, cols=16,
        v_row=2, v_col=16, blayout=col_major, slayout=col_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=bias, dtype=f32, rows=1, cols=16,
        v_row=1, v_col=16, blayout=row_major, slayout=none_box,
        fractal=512, pad=0>)
  outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16,
      v_row=16, v_col=16, blayout=col_major, slayout=row_major,
      fractal=512, pad=0>)
```

---

### `pto.tgemv`: Matrix-Vector Multiplication

```mlir
pto.tgemv ins(<lhs>, <rhs> : <lhs_type>, <rhs_type>)
          outs(<dst> : <dst_type>)
          {accPhase = <phase>}
```

**Semantics:**

```text
For each (i, j):
    dst[i, j] = sum_k lhs[i, k] * rhs[k, j]
// The rows of lhs (i.e., m) must be 1, degenerating to vector-matrix multiplication.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | Matrix tile buffer (`loc=left`, `rows=1`) |
| `rhs` | `pto.tile_buf` | Vector tile buffer (`loc=right`) |
| `dst` | `pto.tile_buf` | Destination accumulator tile buffer (`loc=acc`) |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `accPhase`: Accumulation phase control. The default value is `unspecified`.
  - `#pto<acc_phase unspecified>`: Default behavior, implementation-defined accumulation strategy
  - `#pto<acc_phase partial>`: Partial accumulation, the result is an intermediate value, and will continue to accumulate.
  - `#pto<acc_phase final>`: Final accumulation, mark this as the last accumulation

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - Supported `(dst element type, lhs element type, rhs element type)` combinations: `(i32, i8, i8)`, `(f32, f16, f16)`, `(f32, f32, f32)`, `(f32, bf16, bf16)`.
  - Runtime constraint: `m` must be `1`; `k/n` (respectively taken from `rhs valid row` and `rhs valid column`) must be in the range `[1, 4095]`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The `dst` element type must be `i32` or `f32`.
  - If `dst` is `i32`, the `lhs` and `rhs` element types must both be `i8`.
  - If `dst` is `f32`, the `lhs`/`rhs` element types support `f16`, `bf16`, `f32`, and some fp8 type pairs (target-defined).
  - Runtime constraint: `m` must be `1`; `k/n` (respectively taken from `rhs valid row` and `rhs valid column`) must be in the range `[1, 4095]`.

**Example:**

```mlir
pto.tgemv ins(%lhs, %rhs :
              !pto.tile_buf<loc=left, dtype=f16, rows=1, cols=128,
                  v_row=1, v_col=128, blayout=col_major,
                  slayout=row_major, fractal=512, pad=0>,
              !pto.tile_buf<loc=right, dtype=f16, rows=128, cols=16,
                  v_row=128, v_col=16, blayout=row_major,
                  slayout=col_major, fractal=512, pad=0>)
          outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
                  v_row=1, v_col=16, blayout=col_major,
                  slayout=row_major, fractal=1024, pad=0>)
```

---

### `pto.tgemv.acc`: Matrix-Vector Multiplication-Addition (Accumulation)

```mlir
pto.tgemv.acc ins(<acc_in>, <lhs>, <rhs> : <acc_in_type>, <lhs_type>, <rhs_type>)
              outs(<dst> : <dst_type>)
              {accPhase = <phase>}
```

**Semantics:**

```text
For each (i, j):
    dst[i, j] = acc_in[i, j] + sum_k lhs[i, k] * rhs[k, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `acc_in` | `pto.tile_buf` | Previous accumulator value (`loc=acc`) |
| `lhs` | `pto.tile_buf` | Matrix tile buffer (`loc=left`, `rows=1`) |
| `rhs` | `pto.tile_buf` | Vector tile buffer (`loc=right`) |
| `dst` | `pto.tile_buf` | Destination accumulator tile buffer (`loc=acc`) |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `accPhase`: Accumulation phase control. The default value is `unspecified`.
  - `#pto<acc_phase unspecified>`: Default behavior, with an implementation-defined accumulation strategy
  - `#pto<acc_phase partial>`: Partial accumulation, where the result is an intermediate value and will continue to accumulate
  - `#pto<acc_phase final>`: Final accumulation, marking this as the last accumulation

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The type combination constraints of `lhs`, `rhs`, and `dst` are the same as those of `pto.tgemv`.
  - `acc_in` must be located at `loc=acc`, and its element type and shape are consistent with `dst`.
  - The runtime constraints are the same as those of `pto.tgemv`.

**Example:**

```mlir
pto.tgemv.acc ins(%acc_in, %lhs, %rhs :
                  !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=col_major,
                      slayout=row_major, fractal=1024, pad=0>,
                  !pto.tile_buf<loc=left, dtype=f16, rows=1, cols=128,
                      v_row=1, v_col=128, blayout=col_major,
                      slayout=row_major, fractal=512, pad=0>,
                  !pto.tile_buf<loc=right, dtype=f16, rows=128, cols=16,
                      v_row=128, v_col=16, blayout=row_major,
                      slayout=col_major, fractal=512, pad=0>)
              outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=col_major,
                      slayout=row_major, fractal=1024, pad=0>)
```

---

### `pto.tgemv.bias`: Matrix-Vector Multiplication With Bias Addition

```mlir
pto.tgemv.bias ins(<lhs>, <rhs>, <bias> : <lhs_type>, <rhs_type>, <bias_type>)
               outs(<dst> : <dst_type>)
               {accPhase = <phase>}
```

**Semantics:**

```text
For each (i, j):
    dst[i, j] = sum_k lhs[i, k] * rhs[k, j] + bias[0, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | Matrix tile buffer (`loc=left`, `rows=1`) |
| `rhs` | `pto.tile_buf` | Vector tile buffer (`loc=right`) |
| `bias` | `pto.tile_buf` | Bias tile buffer (`loc=bias`, `rows=1`) |
| `dst` | `pto.tile_buf` | Destination accumulator tile buffer (`loc=acc`) |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `accPhase`: Accumulation phase control. The default value is `unspecified`.
  - `#pto<acc_phase unspecified>`: Default behavior, with the accumulation strategy implementation-defined
  - `#pto<acc_phase partial>`: Partial accumulation, the result is an intermediate value and will continue to accumulate.
  - `#pto<acc_phase final>`: Final accumulation, marking this as the last accumulation

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - The type combination constraints for `lhs`, `rhs`, and `dst` are the same as those for `pto.tgemv`.
  - The `bias` element type must be consistent with the `dst` element type.
  - `bias` must use `loc=bias`, and `rows=1`.
  - The runtime constraints are the same as those of `pto.tgemv`.

- **Implementation check (Ascend 950PR&950DT products)**
  - The type combination constraints of `lhs`, `rhs`, and `dst` are the same as those of `pto.tgemv`.
  - The element type of `bias` must be consistent with the element type of `dst`.
  - `bias` must use `loc=bias`, `rows=1`, and `blayout=row_major`.
  - The runtime constraints are the same as those of `pto.tgemv`.

**Example:**

```mlir
pto.tgemv.bias ins(%lhs, %rhs, %bias :
                   !pto.tile_buf<loc=left, dtype=f16, rows=1, cols=128,
                       v_row=1, v_col=128, blayout=col_major,
                       slayout=row_major, fractal=512, pad=0>,
                   !pto.tile_buf<loc=right, dtype=f16, rows=128, cols=16,
                       v_row=128, v_col=16, blayout=row_major,
                       slayout=col_major, fractal=512, pad=0>,
                   !pto.tile_buf<loc=bias, dtype=f32, rows=1, cols=16,
                       v_row=1, v_col=16, blayout=row_major,
                       slayout=none_box, fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
                       v_row=1, v_col=16, blayout=col_major,
                       slayout=row_major, fractal=1024, pad=0>)
```

---

### `pto.tgemv.mx`: Mixed-Precision Matrix-Vector Multiplication

```mlir
pto.tgemv.mx ins(<lhs>, <lhs_scale>, <rhs>, <rhs_scale> :
                 <lhs_type>, <lhs_scale_type>, <rhs_type>, <rhs_scale_type>)
             outs(<dst> : <dst_type>)
             {accPhase = <phase>}
```

**Semantics:**

```text
k_group = floor(k / 32)

For each (i, j):
    dst[i, j] =
        sum_k (lhs[i, k] * lhs_scale[i, k_group]) *
              (rhs[k, j] * rhs_scale[k_group, j])
// The rows (m) of lhs must be 1; the scaling tile configures the quantization behavior defined by the target.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | Matrix tile buffer (`loc=left`, `rows=1`) |
| `lhs_scale` | `pto.tile_buf` | Left matrix scaling tile buffer (`loc=scaling`) |
| `rhs` | `pto.tile_buf` | Vector tile buffer (`loc=right`) |
| `rhs_scale` | `pto.tile_buf` | Right matrix scaling tile buffer (`loc=scaling`) |
| `dst` | `pto.tile_buf` | Destination accumulator tile buffer (`loc=acc`) |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `accPhase`: Accumulation phase control. The default value is `unspecified`.
  - `#pto<acc_phase unspecified>`: Default behavior, with an implementation-defined accumulation strategy
  - `#pto<acc_phase partial>`: Partial accumulation, the result is an intermediate value and will continue to accumulate
  - `#pto<acc_phase final>`: Final accumulation, marking this as the last accumulation

**Constraints:**

- **Atlas A2 products/Atlas A3 products do not support this operation.**
- **Ascend 950PR&950DT product constraints**
  - The data and target are respectively taken from `left`, `right`, and `acc`, using the Ascend 950PR&950DT products matrix layout in this chapter; the `dst` element type is `f32`.
  - The examples in this section use the FP8 type pair `f8E4M3FN` or `f8E5M2`.
  - The valid `M=1` and the valid `K/N` are in `[1,4095]`; the physical size of the data satisfies the matrix multiplication shape relationship.
  - Both scales are located in `scaling`. Let `G=ceil(K/32)`, where K is the valid reduction dimension: the physical and valid sizes of the left scale are both `[1,G]`; the physical size of the right scale is `[G,rhs.cols]`, and its valid size is `[G,N]`.
  - The physical column capacity of the right scale covers the physical column capacity of the right data; the aligned physical N cannot be replaced with the valid N.

**Example:**

Ascend 950PR&950DT product example: the valid region equals the physical size, with M=1, K=128, and N=16; the left scale is `1x4`, and the right scale is `4x16`. The input data and scaling factors have been prepared in the corresponding tiles.

```mlir
pto.tgemv.mx ins(%a, %a_scale, %b, %b_scale :
    !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=1, cols=128,
        v_row=1, v_col=128, blayout=col_major, slayout=row_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=1, cols=4,
        v_row=1, v_col=4, blayout=row_major, slayout=row_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=right, dtype=f8E4M3FN, rows=128, cols=16,
        v_row=128, v_col=16, blayout=row_major, slayout=col_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=4, cols=16,
        v_row=4, v_col=16, blayout=col_major, slayout=col_major,
        fractal=32, pad=0>)
  outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
      v_row=1, v_col=16, blayout=col_major, slayout=row_major,
      fractal=1024, pad=0>)
```

---

### `pto.tgemv.mx.acc`: Mixed-Precision Matrix-Vector Multiplication-Addition

```mlir
pto.tgemv.mx.acc ins(<acc_in>, <lhs>, <lhs_scale>, <rhs>, <rhs_scale> :
                     <acc_in_type>, <lhs_type>, <lhs_scale_type>,
                     <rhs_type>, <rhs_scale_type>)
                 outs(<dst> : <dst_type>)
                 {accPhase = <phase>}
```

**Semantics:**

```text
k_group = floor(k / 32)

For each (i, j):
    dst[i, j] = acc_in[i, j] +
        sum_k (lhs[i, k] * lhs_scale[i, k_group]) *
              (rhs[k, j] * rhs_scale[k_group, j])
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `acc_in` | `pto.tile_buf` | Previous accumulator value (`loc=acc`) |
| `lhs` | `pto.tile_buf` | Matrix tile buffer (`loc=left`, `rows=1`) |
| `lhs_scale` | `pto.tile_buf` | Left matrix scaling tile buffer (`loc=scaling`) |
| `rhs` | `pto.tile_buf` | Vector tile buffer (`loc=right`) |
| `rhs_scale` | `pto.tile_buf` | Right matrix scaling tile buffer (`loc=scaling`) |
| `dst` | `pto.tile_buf` | Destination accumulator tile buffer (`loc=acc`) |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `accPhase`: Accumulation phase control. The default value is `unspecified`.
  - `#pto<acc_phase unspecified>`: Default behavior, with an implementation-defined accumulation strategy
  - `#pto<acc_phase partial>`: Partial accumulation, the result is an intermediate value and will continue to accumulate later
  - `#pto<acc_phase final>`: Final accumulation, marking this as the last accumulation

**Constraints:**

- **Atlas A2 products/Atlas A3 products do not support this operation.**

- **Implementation check (Ascend 950PR&950DT products)**
  - The constraints on `lhs`, `rhs`, `lhs_scale`, `rhs_scale`, and `dst` are the same as those for `pto.tgemv.mx`.
  - `acc_in` must be located at `loc=acc`, and its element type and shape must be consistent with `dst`.
  - The runtime constraints are the same as those of `pto.tgemv.mx`.

**Example:**

Ascend 950PR&950DT product example: the valid region equals the physical size, with M=1, K=128, and N=16; the left scale is `1x4`, and the right scale is `4x16`. The input data and scaling factors are already prepared in the corresponding tiles.

```mlir
pto.tgemv.mx.acc ins(%c_in, %a, %a_scale, %b, %b_scale :
    !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
        v_row=1, v_col=16, blayout=col_major, slayout=row_major,
        fractal=1024, pad=0>,
    !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=1, cols=128,
        v_row=1, v_col=128, blayout=col_major, slayout=row_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=1, cols=4,
        v_row=1, v_col=4, blayout=row_major, slayout=row_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=right, dtype=f8E4M3FN, rows=128, cols=16,
        v_row=128, v_col=16, blayout=row_major, slayout=col_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=4, cols=16,
        v_row=4, v_col=16, blayout=col_major, slayout=col_major,
        fractal=32, pad=0>)
  outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
      v_row=1, v_col=16, blayout=col_major, slayout=row_major,
      fractal=1024, pad=0>)
```

---

### `pto.tgemv.mx.bias`: Mixed-Precision Matrix-Vector Multiplication With Bias Addition

```mlir
pto.tgemv.mx.bias ins(<lhs>, <lhs_scale>, <rhs>, <rhs_scale>, <bias> :
                      <lhs_type>, <lhs_scale_type>, <rhs_type>,
                      <rhs_scale_type>, <bias_type>)
                  outs(<dst> : <dst_type>)
```

**Semantics:**

```text
k_group = floor(k / 32)

For each (i, j):
    dst[i, j] =
        sum_k (lhs[i, k] * lhs_scale[i, k_group]) *
              (rhs[k, j] * rhs_scale[k_group, j]) + bias[0, j]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `lhs` | `pto.tile_buf` | Matrix tile buffer (`loc=left`, `rows=1`) |
| `lhs_scale` | `pto.tile_buf` | Left matrix scaling tile buffer (`loc=scaling`) |
| `rhs` | `pto.tile_buf` | Vector tile buffer (`loc=right`) |
| `rhs_scale` | `pto.tile_buf` | Right matrix scaling tile buffer (`loc=scaling`) |
| `bias` | `pto.tile_buf` | Bias tile buffer (`loc=bias`, `rows=1`) |
| `dst` | `pto.tile_buf` | Destination accumulator tile buffer (`loc=acc`) |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Atlas A2 products/Atlas A3 products do not support this operation.**

- **Implementation check (Ascend 950PR&950DT products)**
  - The constraints on `lhs`, `rhs`, `lhs_scale`, `rhs_scale`, and `dst` are the same as those for `pto.tgemv.mx`.
  - The `bias` element type must be `f32`.
  - `bias` must use `loc=bias`, `rows=1`, and `blayout=row_major`.
  - The runtime constraints are the same as those for `pto.tgemv.mx`.

**Example:**

Ascend 950PR&950DT product example: the valid region equals the physical size, M=1, K=128, N=16; the left scale is `1x4`, and the right scale is `4x16`. The input data and scaling factors are already prepared in the corresponding tiles.

```mlir
pto.tgemv.mx.bias ins(%a, %a_scale, %b, %b_scale, %bias :
    !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=1, cols=128,
        v_row=1, v_col=128, blayout=col_major, slayout=row_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=1, cols=4,
        v_row=1, v_col=4, blayout=row_major, slayout=row_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=right, dtype=f8E4M3FN, rows=128, cols=16,
        v_row=128, v_col=16, blayout=row_major, slayout=col_major,
        fractal=512, pad=0>,
    !pto.tile_buf<loc=scaling, dtype=f16, rows=4, cols=16,
        v_row=4, v_col=16, blayout=col_major, slayout=col_major,
        fractal=32, pad=0>,
    !pto.tile_buf<loc=bias, dtype=f32, rows=1, cols=16,
        v_row=1, v_col=16, blayout=row_major, slayout=none_box,
        fractal=512, pad=0>)
  outs(%dst : !pto.tile_buf<loc=acc, dtype=f32, rows=1, cols=16,
      v_row=1, v_col=16, blayout=col_major, slayout=row_major,
      fractal=1024, pad=0>)
```
