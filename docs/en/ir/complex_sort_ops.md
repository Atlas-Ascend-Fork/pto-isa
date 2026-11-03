# Sort

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:10:54.104Z pushedAt=2026-11-02T03:00:52.120Z nl=2 -->

This section describes the instruction names, signatures, and semantics of the sort operations in PTO ISA. All operations act on the local `tile_buf`.

Generic assembly form:

```mlir
pto.op ins(<src>, ... : <src_type>, ...) outs(<dst> : <dst_type>)
```
---

## Contents

- [pto.tsort32: 32-Element Block Sort](#ptotsort32-32-element-block-sort)
- [pto.tmrgsort: Merge Sort](#ptotmrgsort-merge-sort)
- [pto.thistogram: Row-Wise Histogram Accumulation](#ptothistogram-row-wise-histogram-accumulation)

---

## Operation Details

### `pto.tsort32`: 32-Element Block Sort

```mlir
// Basic form
pto.tsort32 ins(<src>, <idx> : <src_type>, <idx_type>)
            outs(<dst> : <dst_type>)

// Form with a temporary tile
pto.tsort32 ins(<src>, <idx>, <tmp> : <src_type>, <idx_type>, <tmp_type>)
            outs(<dst> : <dst_type>)
```

**Semantics:**

```text
dst = sort(src, idx)
// Sort fixed 32-element blocks in src.
// idx is the index tile, arranged together with the values in src according to the sort result.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Input value tile. |
| `idx` | `pto.tile_buf` | Input index tile, arranged together with `src`. |
| `tmp` | `pto.tile_buf` | Temporary scratch tile (optional). |
| `dst` | `pto.tile_buf` | Output tile that stores the sorted value-index pairs. |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR & 950DT products)**
  - The `src` and `dst` element types must be consistent, and must be `f16` or `f32`.
  - The `idx` element type must be a 32-bit unsigned integer (represented as `ui32` or `i32` in MLIR).
  - `src`, `dst`, and `idx` must be `loc=vec` and `blayout=row_major`.

**Example:**

```mlir
pto.tsort32 ins(%src, %idx :
                !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
            outs(%dst0 : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=64,
                    v_row=1, v_col=64, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
```

---

### `pto.tmrgsort`: Merge Sort

`pto.tmrgsort` has two formats: single-list merge sort (format1) and multi-list merge sort (format2).

```mlir
// format1: single-list merge sort.
pto.tmrgsort ins(<src>, <blockLen> : <src_type>, <int_type>)
             outs(<dst> : <dst_type>)

// format2: multi-list merge sort (2 to 4 ways).
pto.tmrgsort ins(<src0>, <src1>, ... , <tmp> {exhausted = <bool>} :
                 <src_type>, <src_type>, ... , <tmp_type>)
             outs(<dst>, <executed> : <dst_type>, vector<4xi16>)
```

**Semantics:**

```text
format1:
    dst = merge_sort(src, blockLen)
    // Group every blockLen*4 elements in src and merge-sort the ordered sub-blocks of blockLen length.

format2:
    dst = merge(src0, src1, ...)
    // Merge 2 to 4 sorted input lists into a single ordered output.
    executed = element count consumed by each way
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` / `src0..src3` | `pto.tile_buf` | Input tiles. format2 supports 2 to 4 sources. |
| `blockLen` | `AnyInteger` | Block length in format1. |
| `dst` | `pto.tile_buf` | Output tile. |
| `tmp` | `pto.tile_buf` | Temporary tile in format2 (format2 only). |
| `executed` | `vector<4xi16>` | Per-lane consumption count vector of the output in format2 (format2 only). |

**Returns:** No SSA return. Writes to `dst` in DPS form; format2 also writes to `executed`.

**Attributes:**

- `exhausted`: Whether to use the exhausted mode (used in format2). The default value is `false`.
  - `false`: Non-exhausted mode
  - `true`: Exhausted mode, which accepts additional source operands

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR & 950DT products)**
  - **format1**: The element type must be `f16` or `f32`, and the element types of `src` and `dst` must be consistent. `src` and `dst` must be rank-2 with `rows == 1` (data stored in a single row). The `cols` of `src` and `dst` must be consistent. `blockLen` must be greater than 0 and an integer multiple of 64. The valid column count of `src` must be an integer multiple of `blockLen * 4`. `repeatTimes = valid column count of src / (blockLen * 4)` must be within the range `[1, 255]`.
  - **format2**: Accepts 2-way, 3-way, or 4-way merge. The element types and shapes of `dst` and `tmp` must be consistent. The element types of all `src` must be consistent with `dst`/`tmp` and be `f16` or `f32`. All tiles must be rank-2 with `rows == 1`. `tmp.cols >= dst.cols`. `executed` must be of type `vector<4xi16>`.

**Example:**

```mlir
// format2: 2-way merge sort
pto.tmrgsort ins(%src0, %src1, %tmp2 {exhausted = false} :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128,
                     v_row=1, v_col=128, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128,
                     v_row=1, v_col=128, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=256,
                     v_row=1, v_col=256, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst2, %ex :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=256,
                     v_row=1, v_col=256, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 vector<4xi16>)
```

---

### `pto.thistogram`: Row-Wise Histogram Accumulation

```mlir
pto.thistogram ins(<src>, <idx> : <src_type>, <idx_type>)
               outs(<dst> : <dst_type>)
```

**Semantics:**

```text
For each row i:
    bin = select_bin(src[i, :], idx[i, 0], isMSB)
    dst[i, bin] += 1
// Perform 256-bin histogram accumulation on the source tile row by row for the bit segment determined by the index.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source data tile (`ui16`). |
| `idx` | `pto.tile_buf` | Index tile that specifies bit selection (`ui8`, single column). |
| `dst` | `pto.tile_buf` | Destination histogram tile (`ui32`, 256 columns). |

**Returns:** None. Writes to `dst` in DPS form.

**Attributes:**

- `isMSB`: Indicates whether to select the high-order byte. The default value is `true`.
  - `true`: Selects the high 8 bits.
  - `false`: Selects the low 8 bits.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products)**
  - Not supported; available only on Ascend 950PR & 950DT products.

- **Implementation check (Ascend 950PR & 950DT products)**
  - The `src` element type must be `ui16`.
  - The `idx` element type must be `ui8`, and it must be a single column (cols=1).
  - The `dst` element type must be `ui32`, and the number of columns must be 256.
  - All tiles must use `loc=vec`.

**Example:**

```mlir
pto.thistogram
    ins(%src, %idx :
        !pto.tile_buf<loc=vec, dtype=ui16, rows=32, cols=32,
                      v_row=8, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui8, rows=32, cols=1,
                      v_row=8, v_col=1, blayout=col_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=32, cols=256,
                              v_row=8, v_col=256, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```
