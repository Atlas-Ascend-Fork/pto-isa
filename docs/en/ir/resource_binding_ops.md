# Resource Binding Operations

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:49:57.550Z pushedAt=2026-11-02T03:00:52.137Z nl=2 -->

This chapter describes the resource binding operations in the PTO ISA, which are responsible for the logical allocation of tile buffers.

---

## Contents

- [`pto.alloc_tile`: Logical Tile Buffer Allocation](#ptoalloc_tile-logical-tile-buffer-allocation)

---

## Operation Details

### `pto.alloc_tile`: Logical Tile Buffer Allocation

```mlir
pto.alloc_tile (valid_row = <vr>)? (valid_col = <vc>)?
               : <result_type>
```

**Semantics:**

```text
result = allocate_tile_buffer(valid_row?, valid_col?)
// Create a logical tile buffer handle, optionally specifying dynamic valid dimensions.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `valid_row` | `index` (optional) | Runtime valid row count. Must be provided when `v_row` of the result type is dynamic (`?`). |
| `valid_col` | `index` (optional) | Runtime valid column count. Must be provided when `v_col` of the result type is dynamic (`?`). |

**Return value:** `!pto.tile_buf<...>`, the allocated tile buffer handle.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The result tile_buf must have a rank-2 validShape.
  - When `v_row` of the result type is dynamic (`?`), the `valid_row` operand must be provided; when `v_row` is static, it must not be provided.
  - When `v_col` of the result type is dynamic (`?`), the `valid_col` operand must be provided; when `v_col` is static, it must not be provided.
  - The layout constraints of the result tile_buf must be valid (verified by `verifyTileBufLayoutConstraints`).

**Example:**

```mlir
// Static valid shape, no valid_row/valid_col required.
%t0 = pto.alloc_tile
    : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major,
                    slayout=none_box, fractal=512, pad=0>

// Dynamic valid shape, valid_row and valid_col required.
%t1 = pto.alloc_tile
                     valid_row = %c16 valid_col = %c256
    : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=256,
                    v_row=?, v_col=?, blayout=col_major,
                    slayout=row_major, fractal=1024, pad=0>
```
