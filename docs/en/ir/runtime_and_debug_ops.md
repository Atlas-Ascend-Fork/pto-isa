# Runtime and Debug Operations

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:58:27.875Z pushedAt=2026-11-02T03:00:52.139Z nl=2 -->

This section describes the operation families in the PTO ISA related to runtime environment queries, debug output, and special control. These operations are grouped by function as follows:

- **Block parameter query**: Gets the index of the current block or subblock, the total block count, or the total subblock count.
- **Scalar access**: Reads or writes a single element from the tile buffer.
- **Debug output**: Formats and prints a scalar value or the entire tile buffer content for diagnostics during program execution.
- **Execution termination**: Terminates execution unconditionally, used for fast failure on assertion failure or during debugging.

---

## Contents

[**Block Parameter Query**](#block-parameter-query)

- [`pto.get_block_idx`: Obtaining the Current Block Index](#ptoget_block_idx-obtaining-the-current-block-index)
- [`pto.get_subblock_idx`: Obtaining the Current Subblock Index](#ptoget_subblock_idx-obtaining-the-current-subblock-index)
- [`pto.get_block_num`: Obtaining the Total Block Count](#ptoget_block_num-obtaining-the-total-block-count)
- [`pto.get_subblock_num`: Obtaining the Total Subblock Count](#ptoget_subblock_num-obtaining-the-total-subblock-count)

[**Scalar Access**](#scalar-access)

- [`pto.tsetval`: Writing a Single Tile Element](#ptotsetval-writing-a-single-tile-element)
- [`pto.tgetval`: Reading a Single Tile Element](#ptotgetval-reading-a-single-tile-element)

[**Debug Output**](#debug-output)

- [`pto.print`: Formatted Scalar Printing](#ptoprint-formatted-scalar-printing)
- [`pto.tprint`: Tile Printing](#ptotprint-tile-printing)

[**Execution Termination**](#execution-termination)

- [`pto.trap`: Trapping/Terminating Execution](#ptotrap-trappingterminating-execution)

---

## Operation Details

## Block Parameter Query

All operations in this group are pure operations with no side effects and no additional constraints.

### `pto.get_block_idx`: Obtaining the Current Block Index

```mlir
%idx = pto.get_block_idx
```

**Semantics:**

```text
result = block_idx()
```

**Parameters:** None

**Returns:** A value of type `i64`, representing the index of the current executing block, in the range `[0, BlockNum-1]`.

**Example:**

```mlir
%idx = pto.get_block_idx
```

---

### `pto.get_subblock_idx`: Obtaining the Current Subblock Index

```mlir
%idx = pto.get_subblock_idx
```

**Semantics:**

```text
result = subblock_idx()
```

**Parameters:** None

**Returns:** A value of type `i64`, representing the index of the current executing subblock (vector core).

**Example:**

```mlir
%idx = pto.get_subblock_idx
```

---

### `pto.get_block_num`: Obtaining the Total Block Count

```mlir
%num = pto.get_block_num
```

**Semantics:**

```text
result = block_num()
```

**Parameters:** None

**Returns:** A value of type `i64` that represents the total number of blocks.

**Example:**

```mlir
%num = pto.get_block_num
```

---

### `pto.get_subblock_num`: Obtaining the Total Subblock Count

```mlir
%num = pto.get_subblock_num
```

**Semantics:**

```text
result = subblock_num()
```

**Parameters:** None

**Returns:** A value of type `i64`, representing the total number of subblocks (vector cores).

**Example:**

```mlir
%num = pto.get_subblock_num
```

---

## Scalar Access

### `pto.tsetval`: Writing a Single Tile Element

```mlir
pto.tsetval ins(<offset>, <val> : index, <scalar_type>)
            outs(<dst> : !pto.tile_buf<...>)
```

**Semantics:**

```text
dst[offset] = val  // Write a single scalar value at the linear offset.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `offset` | `index` | Linear offset in the tile buffer (starting from 0) |
| `val` | `ScalarType` | Scalar value to be written |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Returns:** None. Writes to `dst` in DPS form.

**Constraints:**

- **Type matching**: The scalar type of `val` must exactly match the element type of `dst` (for example, `f16` with `f16`).
- **Offset range**: `offset` must be within the range `[0, rows*cols)`.
- **Validity**: The operation takes effect only within the valid region of `dst`.
- **Hardware pipe**: The operation executes on the scalar pipe (`PIPE_S`), therefore there is a sequential constraint between operations on the same pipe and dependent pipes.

**Example:**

```mlir
%offset = ... : index
%val = ... : f16
pto.tsetval ins(%offset, %val : index, f16)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.tgetval`: Reading a Single Tile Element

```mlir
%val = pto.tgetval ins(<src>, <offset> : !pto.tile_buf<...>, index) outs : <scalar_type>
```

**Semantics:**

```text
result = src[offset]  // Reads a single scalar value from the linear offset.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | Source tile buffer |
| `offset` | `index` | Linear offset in the tile buffer (starting from 0) |

**Returns:** A scalar value of the same type as the element type of `src`.

**Constraints:**

- **Location requirement**: `src` must use `loc=vec`. `loc=mat` is not accepted.
- **Offset range**: `offset` must be within the range `[0, rows*cols)`.
- **Element type**: The `dst` type must match the element type of `src`.
- **Hardware pipe**: The operation executes on the scalar pipe (`PIPE_S`), therefore there is a sequential constraint between operations on the same pipe and dependent pipes.

**Example:**

```mlir
%src = ... : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>
%offset = ... : index
%val = pto.tgetval ins(%src, %offset : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>, index) outs : f16
```

---

## Debug Output

**General constraints:**

- These operations have a memory write side effect (`MemWrite`) to mark debug output.
- The output is visible on the host side.
- Whether valid output is generated is determined by the specific implementation and the target platform.

### `pto.print`: Formatted Scalar Printing

```mlir
pto.print ins(<format_str>, <scalar> : <scalar_type>)
```

**Semantics:**

```text
printf(format_str, scalar)
```

Here `format_str` is determined at compile time, and `scalar` is a runtime value.

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `format_str` | `StrAttr` | Compile-time string attribute containing format control specifiers (such as `%+08.3f`) |
| `scalar` | `ScalarType` (signless integer / float) | Runtime scalar value whose type must be an integer or floating-point number |

**Returns:** None. This operation is for debug output and produces no return value.

**Constraints:**

- **Format requirement**: `format_str` must be an MLIR string attribute (StrAttr).
- **Scalar type**: `scalar` must be one of the following types: `i32`, `i64`, `f32`, `f64`, or another numeric type. `tile_buf` or complex types are not supported.

**Example:**

```mlir
%val = ... : f32
pto.print ins("Value: %+08.3f", %val : f32)
```

---

### `pto.tprint`: Tile Printing

```mlir
pto.tprint ins(<src> : !pto.tile_buf<...>)

// Print the Mat/Acc tile format through a temporary global buffer.
pto.tprint ins(<src>, <tmp> : !pto.tile_buf<...>, !pto.partition_tensor_view<...>)
          {printFormat = <fmt>}
```

**Semantics:**

```text
print(src)  // Print the contents of the entire tile buffer.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` / `pto.partition_tensor_view` | Tile buffer or global view to be printed; it must be `pto.tile_buf` when `tmp` is present |
| `tmp` | `pto.partition_tensor_view` (optional) | Temporary global buffer view used when printing Mat/Acc tiles, **not** `pto.tile_buf` |

**Returns:** None. This operation is for debug output and produces no return value.

**Attributes:**

- `printFormat`: TPRINT output format. This attribute is optional and unspecified by default.
  - `#pto<print_format width8_precision4>`: width 8, precision 4
  - `#pto<print_format width8_precision2>`: width 8, precision 2
  - `#pto<print_format width10_precision6>`: width 10, precision 6

**Constraints:**

The two forms of `src` follow different validation paths and have different constraints.

- **src is a tile (`pto.tile_buf`)**
  - **Element type**: must be one of the following: `f16`, `f32`, `i8`, `i16`, `i32`.
  - **Without `tmp`**: `src` must be located at `loc=vec`; other locations are rejected.
  - **With `tmp`**: `src` can be `vec`/`mat`/`acc`; on Ascend 950PR&950DT products, printing a `mat` tile is not allowed (only Atlas A2 products/Atlas A3 products support it).
- **src is a global view (`pto.partition_tensor_view`)**
  - **Without `tmp`**: printed directly, with no `loc` requirement (the view type does not carry a local location).
  - **`tmp` is not allowed**: the verifier checks `expects tmp only when src is a tile_buf`; when `src` is not a `tile_buf`, `tmp` must not be written.
- **`tmp` type**: allowed only when `src` is a tile, and `tmp` must be `!pto.partition_tensor_view<...>` (the verifier checks `expects mem to be !pto.partition_tensor_view`; writing it as a tile_buf is rejected); its element type must match the element type of `src`.
- **`tmp` layout**: when the layout of `tmp` is inferable, the ND logical layout must be used.
- **Buffer limit**: the print buffer size is limited, and printing an oversized tile may be truncated.
- **Hardware pipe**: the operation is executed on the vector pipe (`PIPE_V`), therefore there is a sequential constraint with operations on the same pipe and dependent pipes.

**Example:**

```mlir
// src is a tile without tmp: loc=vec is required.
pto.tprint ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
    fractal=512, pad=0>)

// src is a tile with tmp: acc/mat can be printed, and tmp uses partition_tensor_view.
pto.tprint ins(%src, %tmp :
        !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=col_major, slayout=row_major,
            fractal=1024, pad=0>,
        !pto.partition_tensor_view<16x16xf32>)
          {printFormat = #pto<print_format width10_precision6>}

// src is a global view: no loc requirement, and tmp cannot be carried.
pto.tprint ins(%pv : !pto.partition_tensor_view<16x16xf32>)
```

---

## Execution Termination

### `pto.trap`: Trapping/Terminating Execution

```mlir
pto.trap
```

**Semantics:**

```text
trap()  // Unconditionally terminate the current block execution without returning.
```

**Parameters:** None

**Returns:** None. This operation terminates execution unconditionally.

**Constraints:**

- **Behavior**: Unconditionally terminates the execution of the current block and does not return to the caller. It is typically used as a fast-fail path when an assertion fails.
- **Control flow**: Typically appears in a conditional branch (such as one branch of `scf.if`), or as a marker for unreachable code.

**Example:**

```mlir
scf.if %cond {
  pto.trap  // Terminate when the condition is true.
}
```
