# Data Type Conversion

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:07:06.145Z pushedAt=2026-11-02T03:00:52.115Z nl=2 -->

This section describes the instruction names, signatures, and semantics of data type conversion operations in the PTO ISA. All operations act on the local `tile_buf`.

Generic assembly form:

```mlir
pto.op ins(<src>, ... : <src_type>, ...) outs(<dst> : <dst_type>)
```
---

## Contents

- [`pto.tcvt`: Element-wise Type Conversion](#ptotcvt-element-wise-type-conversion)
- [`pto.tquant`: Tile Quantization](#ptotquant-tile-quantization)
- [`pto.tdequant`: Tile Dequantization](#ptotdequant-tile-dequantization)

---

## Operation Details

### `pto.tcvt`: Element-wise Type Conversion

```mlir
pto.tcvt ins(<src>[, <tmp>] {rmode = <round_mode>, satmode = <saturation_mode>}
            : <src_type>[, <tmp_type>])
         outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = saturate(cast(src[i, j], rmode), satmode)
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer. |
| `tmp` | `pto.tile_buf` (optional) | Temporary scratch tile; required only for `f32 -> i16`, `f16 -> i16`, and `f16 -> i8` conversions with `satmode=OFF` on Atlas A2 products/Atlas A3 products. |
| `dst` | `pto.tile_buf` | Destination tile buffer, whose element type may differ from `src`. |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `rmode`: Rounding mode. The default value is `CAST_RINT`.
  - `#pto<round_mode NONE>`: no rounding
  - `#pto<round_mode RINT>`: round to nearest even
  - `#pto<round_mode ROUND>`: round half away from zero
  - `#pto<round_mode FLOOR>`: round down
  - `#pto<round_mode CEIL>`: round up
  - `#pto<round_mode TRUNC>`: truncate
  - `#pto<round_mode ODD>`: round to nearest odd
  - `#pto<round_mode CAST_RINT>`: default rounding for type conversion
- `satmode`: saturation mode, which controls whether to clamp to the target type range after rounding. The default value is `ON`.
  - `#pto<saturation_mode ON>`: enable saturation
  - `#pto<saturation_mode OFF>`: disable saturation

**Constraints:**

- **General checks**
  - `src` and `dst` must be compatible tile buffers.
  - The logical ranges and valid regions of `src` and `dst` must be compatible.

- **Low-precision restrictions for Atlas A2 products/Atlas A3 products and Ascend 950PR&950DT products**
  - Atlas A2 products/Atlas A3 products do not support low-precision `tcvt` operands.
  - Ascend 950PR&950DT products accept only the low-precision conversion pairs listed in the implementation, for example `f32 -> f8E4M3*`, `f32 -> f8E5M2*`, `f32 -> !pto.hif8`, `f16 -> !pto.hif8`, `bf16 <-> !pto.f4E1M2x2`, `bf16 <-> !pto.f4E2M1x2`, `f8E4M3* -> f32`, `f8E5M2* -> f32`, and `!pto.hif8 -> f32`.
  - Non-low-precision type pairs follow the conversion behavior defined by the target.

- **Optional tmp operand**
  - `tmp` is an optional operand; it is required only on Atlas A2 products/Atlas A3 products when `satmode=OFF` and the conversion pair is `f32 -> i16`, `f16 -> i16`, or `f16 -> i8`.
  - The default `satmode=ON` does not require `tmp`; narrowing conversions on Ascend 950PR&950DT products also do not require `tmp`.
  - When `tmp` is omitted, the implementation automatically fills in the temporary space.

**Hardware:**

- PIPE_V

**Example:**

```mlir
pto.tcvt ins(%src {rmode = #pto<round_mode FLOOR>, satmode = #pto<saturation_mode ON>} :
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tquant`: Tile Quantization

```mlir
pto.tquant ins(<src>, <fp> : !pto.tile_buf, !pto.tile_buf)
           outs(<dst> : !pto.tile_buf) {quant_type = <quant_type>}
```

**Semantics:**

```text
For each element (i, j):
    dst[i, j] = Quantize(src[i, j]; fp, quant_type)
```

Here, `fp` is the scaling factor tile (usually a single column or row).

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer (f32 type). |
| `fp` | `pto.tile_buf` | Scaling factor tile buffer. |
| `dst` | `pto.tile_buf` | Destination tile buffer (integer type). |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `quant_type`: Quantization type.
  - `#pto<quant_type INT8_SYM>`: Symmetric quantization, `dst` is `i8`.
  - `#pto<quant_type INT8_ASYM>`: Asymmetric quantization, `dst` is `ui8`.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - **src** must be of type `f32`.
  - The element type of the optional `offset` must be `f32`.
  - The effective shapes of **src** and **dst** must be consistent.
  - Atlas A2 products/Atlas A3 products: **src** and **dst** must use `blayout=row_major`.

**Example:**

```mlir
pto.tquant
    ins(%src, %fp :
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                      v_row=32, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1,
                      v_row=32, v_col=1, blayout=col_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32,
                              v_row=32, v_col=32, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
    {quant_type = #pto<quant_type INT8_SYM>}
```

---

### `pto.tdequant`: Tile Dequantization

```mlir
pto.tdequant ins(<src>, <scale>, <offset> : !pto.tile_buf, !pto.tile_buf, !pto.tile_buf)
             outs(<dst> : !pto.tile_buf)
```

**Semantics:**

```text
For each row i:
    For each column j:
        dst[i][j] = (float(src[i][j]) - offset[i][0]) * scale[i][0]
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer (integer type, i8 or i16). |
| `scale` | `pto.tile_buf` | Scaling factor tile buffer (usually a single column). |
| `offset` | `pto.tile_buf` | Offset tile buffer (usually a single column). |
| `dst` | `pto.tile_buf` | Destination tile buffer (f32 type). |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The element type of `src` must be an 8-bit or 16-bit integer.
  - The element types of `scale`, `offset`, and `dst` must be `f32`.
  - `src`, `scale`, `offset`, and `dst` must all be valid rank-2 `tile_buf`.
  - Atlas A2 products/Atlas A3 products additionally require `src` and `dst` to use row-major layout; Ascend 950PR&950DT products have no such additional layout restriction.

**Example:**

```mlir
pto.tdequant
    ins(%src, %scale, %offset :
        !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32,
                      v_row=32, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1,
                      v_row=32, v_col=1, blayout=col_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1,
                      v_row=32, v_col=1, blayout=col_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                              v_row=32, v_col=32, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```
