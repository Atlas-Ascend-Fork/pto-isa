# Intra-Core CV Pipe Communication Operations

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T03:23:58.987Z pushedAt=2026-11-02T03:00:52.132Z nl=2 -->

This section describes the pipe communication interface in the PTO ISA for FIFO-style data exchange between Cube (AIC) and Vector (AIV) cores. These operations support the MPMD (multiple program, multiple data) execution model, allowing Cube cores and Vector cores to exchange data asynchronously through pipes. A pipe entry can be a local tile buffer or a GlobalTensor-style global memory view descriptor. All pipe communication operations must be bound through the compile-time attribute `id`, which associates `initialize_pipe` with the corresponding `tpush`/`tpop`/`tfree` operations.

The common characteristics of this category of operations include:

- **dir_mask** attribute: controls the pipe communication direction, 1 = C2V (Cube to Vector), 2 = V2C (Vector to Cube), 3 = bidirectional
- **id** attribute: a compile-time integer constant that binds the initialization operation with the production/consumption/release operations
- **slot_size** attribute: the logical total pipe size in bytes
- **slot_num** attribute: optional, controls the FIFO depth (default is 8 when dir_mask=1/2, and 4 when dir_mask=3)
- **split** attribute: compile-time attribute (0=TILE_NO_SPLIT, 1=TILE_UP_DOWN, 2=TILE_LEFT_RIGHT)
- **tpop operation return value**: `tpop_from_aic` and `tpop_from_aiv` are operations with return values.
- **Pipe entry type**: tile entry (`!pto.tile_buf`) or global entry (`!pto.tensor_view`)

---

## Contents

- [`pto.aic_initialize_pipe`: Cube-Side Pipe Initialization](#ptoaic_initialize_pipe-cube-side-pipe-initialization)
- [`pto.aiv_initialize_pipe`: Vector-Side Pipe Initialization](#ptoaiv_initialize_pipe-vector-side-pipe-initialization)
- [`pto.tpush_to_aiv`: C2V Producer Push](#ptotpush_to_aiv-c2v-producer-push)
- [`pto.tpush_to_aic`: V2C Producer Push](#ptotpush_to_aic-v2c-producer-push)
- [`pto.tpop_from_aic`: C2V Consumer Pop](#ptotpop_from_aic-c2v-consumer-pop)
- [`pto.tpop_from_aiv`: V2C Consumer Pop](#ptotpop_from_aiv-v2c-consumer-pop)
- [`pto.tfree_from_aic`: C2V Consumer Release](#ptotfree_from_aic-c2v-consumer-release)
- [`pto.tfree_from_aiv`: V2C Consumer Release](#ptotfree_from_aiv-v2c-consumer-release)
- [`pto.reserve_buffer`: Reserving Local Consumer Buffer](#ptoreserve_buffer-reserving-local-consumer-buffer)
- [`pto.import_reserved_buffer`: Importing Peer Reserved Buffer](#ptoimport_reserved_buffer-importing-peer-reserved-buffer)
- [`pto.section.cube`: Cube Core Code Region](#ptosectioncube-cube-core-code-region)
- [`pto.section.vector`: Vector Core Code Region](#ptosectionvector-vector-core-code-region)
- [`pto.initialize_l2g2l_pipe`: Initializing an L2G2L Pipe](#ptoinitialize_l2g2l_pipe-initializing-an-l2g2l-pipe)
- [`pto.initialize_l2l_pipe`: Initializing an L2L Pipe](#ptoinitialize_l2l_pipe-initializing-an-l2l-pipe)
- [`pto.talloc_to_aiv`: C2V Producer FIFO Allocation](#ptotalloc_to_aiv-c2v-producer-fifo-allocation)
- [`pto.talloc_to_aic`: V2C Producer FIFO Allocation](#ptotalloc_to_aic-v2c-producer-fifo-allocation)

---

## Operation Details

### `pto.aic_initialize_pipe`: Cube-Side Pipe Initialization

```mlir
pto.aic_initialize_pipe {
                        id = <id>,
                        dir_mask = <dir>,
                        slot_size = <size>,
                        slot_num = <num>,
                        local_slot_num = <local_num>,
                        nosplit = <bool>
                      } (
                        gm_slot_buffer = <buf> : !pto.ptr<T>,
                        gm_slot_tensor = <tensor> : !pto.tensor_view<...>,
                        c2v_consumer_buf = <c2v> : i32,
                        v2c_consumer_buf = <v2c> : i32
                      )
```

**Semantics:**

Initializes a pipe channel in the Cube core to establish a shared FIFO buffer for bidirectional or unidirectional data exchange. This operation returns no value and only completes the initialization configuration. For local buffer entries, pipe management is handled by the runtime; for global memory entries, the GM FIFO slot descriptor is bound during initialization.

**Attributes:**

- `id`: pipe identifier. A compile-time integer constant used to bind this initialization operation with subsequent `tpush`/`tpop`/`tfree` operations. It must be unique within the same kernel.

- `dir_mask`: communication direction mask. Determines the communication mode of the pipe:
  - `1`: C2V (Cube to Vector). Only the Cube side is allowed to push and the Vector side consumes.
  - `2`: V2C (Vector to Cube). Only the Vector side is allowed to push and the Cube side consumes.
  - `3`: bidirectional. Both sides can push and consume.

- `slot_size`: required, size of a pipe entry (in bytes). It must be greater than 0.

- `slot_num`: optional, FIFO depth (number of entries in the FIFO). Default value: **8** when `dir_mask` is **1** or **2**, and **4** when it is **3**. It must be greater than 0.

- `local_slot_num`: optional, Atlas A2/A3 products only. FIFO depth of the local tile buffer. It must be greater than 0 and not exceed `slot_num`. This attribute must be omitted on Ascend 950PR&950DT products.

- `nosplit`: optional boolean attribute. If `true`, disables tile splitting and requires the `split` attribute of all bound `tpush`/`tpop`/`tfree` operations to be 0 (`TILE_NO_SPLIT`).

- `gm_slot_buffer`: optional, type `!pto.ptr<T>`. Global memory FIFO slot buffer pointer. It is used for C2V or V2C producer push of GM entries.

- `gm_slot_tensor`: optional, type `!pto.tensor_view<...>`. Tensor view descriptor of the global memory FIFO. It is only for GM-only FIFO (global entries only).

- `c2v_consumer_buf`: optional, type `i32`. C2V consumer count buffer or imported value. It is used to synchronize the consumption progress on the Vector side.

- `v2c_consumer_buf`: optional, type `i32`. V2C consumer count buffer or imported value. It is used to synchronize the consumption progress on the Cube side.

**Constraints:**

- **Implementation check (Atlas A2/A3 products)**
  - It must appear in the Cube kernel (`pto.kernel_kind = #pto.kernel_kind<cube>`).
  - `id` must be unique in the kernel.
  - `slot_num` must be greater than 0.
  - If `local_slot_num` is specified, it must be greater than 0 and `<= slot_num`.
  - `dir_mask` must be 1, 2, or 3.
  - `slot_size` must be greater than 0.
  - If `nosplit` is `true`, the `split` attribute of all `tpush`/`tpop`/`tfree` operations using this `id` must be 0.

- **Implementation check (Ascend 950PR&950DT products)**
  - It must appear in the Cube kernel.
  - `local_slot_num` must be omitted (not supported on Ascend 950PR&950DT products).
  - Other constraints are the same as those for Atlas A2 products/Atlas A3 products.

**Example:**

```mlir
// Atlas A2 products/Atlas A3 products: C2V initialization using GM buffer and local slot count configuration.
pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, slot_num = 2, local_slot_num = 1}
    (gm_slot_buffer = %gm_buf : !pto.ptr<f32>,
     c2v_consumer_buf = %c2v_import : i32,
     v2c_consumer_buf = %c0_i32 : i32)

// Ascend 950PR&950DT products: C2V initialization without local slot configuration.
pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, nosplit = true}
    (c2v_consumer_buf = %c2v_import : i32,
     v2c_consumer_buf = %c0_i32 : i32)
```

---

### `pto.aiv_initialize_pipe`: Vector-Side Pipe Initialization

```mlir
pto.aiv_initialize_pipe {id = <id>, dir_mask = <dir>, slot_size = <size>,
                         slot_num = <num>, local_slot_num = <local_num>,
                         nosplit = <bool>,
                         gm_slot_buffer = <buf>, gm_slot_tensor = <tensor>,
                         c2v_consumer_buf = <c2v>, v2c_consumer_buf = <v2c>}
```

**Semantics:**

Initializes a pipe channel in the Vector core, with a structure identical to that of `pto.aic_initialize_pipe`. This operation configures the FIFO pipe for the Vector side, enabling it to participate in data exchange as a consumer (C2V mode) or a producer (V2C mode).

**Attributes:**

The attribute definitions are identical to those of `pto.aic_initialize_pipe`.

**Constraints:**

- **Implementation check (Atlas A2/A3 products)**
  - It must appear in the Vector kernel (`pto.kernel_kind = #pto.kernel_kind<vector>`).
  - All constraints are the same as those of `pto.aic_initialize_pipe`.

- **Implementation check (Ascend 950PR&950DT products)**
  - It must appear in the Vector kernel.
  - `local_slot_num` must be omitted.

**Example:**

```mlir
// Atlas A2 products/Atlas A3 products: Vector-side C2V consumer initialization.
pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, slot_num = 2, local_slot_num = 1}
    (gm_slot_buffer = %gm_buf : !pto.ptr<f32>,
     c2v_consumer_buf = %c2v_import : i32,
     v2c_consumer_buf = %c0_i32 : i32)

// Ascend 950PR&950DT products: Vector-side initialization.
pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, nosplit = true}
    (c2v_consumer_buf = %c2v_import : i32,
     v2c_consumer_buf = %c0_i32 : i32)
```

---

### `pto.tpush_to_aiv`: C2V Producer Push

```mlir
pto.tpush_to_aiv(<pipe_entry> : <pipe_entry_type>)
    {id = <id>, split = <split>}
```

**Semantics:**

Pushes a C2V pipe entry from the Cube core to the FIFO. For a tile buffer entry, performs tile transfer; for a global memory entry, commits the GM FIFO slot and advances the producer pointer. This operation produces no return value.

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `pipe_entry` | `!pto.tile_buf` or `!pto.tensor_view` | Pipe entry to push, which can be a tile buffer or a global tensor view. |

**Returns:** None. The operation returns after the entry is submitted to the FIFO.

**Attributes:**

- `id`: Pipe identifier. It must match the `id` of `pto.aic_initialize_pipe` in the same Cube kernel.

- `split`: Tile split mode. A compile-time attribute that determines how the entry is split:
  - `0`: `TILE_NO_SPLIT`, no split
  - `1`: `TILE_UP_DOWN`, split by row
  - `2`: `TILE_LEFT_RIGHT`, split by column

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - It must appear in the Cube kernel.
  - `id` must match a `pto.aic_initialize_pipe` operation that uses `dir_mask=1` or `dir_mask=3`.
  - For global memory entries, the operation must be dominated by the corresponding `pto.talloc_to_aiv` (it must come after that allocation).
  - If the `nosplit` of the initialization operation is `true`, `split` must be 0.

**Example:**

```mlir
// Push a tile buffer entry (split by row).
pto.tpush_to_aiv(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>)
    {id = 0, split = 1}

// Push a global memory entry (no split).
pto.tpush_to_aiv(%entry : !pto.tensor_view<16x16xf32>)
    {id = 0, split = 0}
```

---

### `pto.tpush_to_aic`: V2C Producer Push

```mlir
pto.tpush_to_aic(<pipe_entry> : <pipe_entry_type>)
    {id = <id>, split = <split>}
```

**Semantics:**

Pushes a V2C Pipe entry from the Vector core to the FIFO. The structure is the same as `pto.tpush_to_aiv`, but in the opposite direction (Vector produces, Cube consumes).

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `pipe_entry` | `!pto.tile_buf` or `!pto.tensor_view` | Pipe entry to push. |

**Returns:** None. The operation returns after the entry is submitted to the FIFO.

**Attributes:**

- `id`: Pipe identifier. It must match the `id` of `pto.aiv_initialize_pipe` in the same Vector kernel.

- `split`: Tile split mode (defined the same as `pto.tpush_to_aiv`).

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - It must appear in the Vector kernel.
  - `id` must match one use of the `pto.aiv_initialize_pipe` operation with `dir_mask=2` or `dir_mask=3`.
  - Other constraints are the same as `pto.tpush_to_aiv`.

**Example:**

```mlir
// Vector-side V2C push.
pto.tpush_to_aic(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
    v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>)
    {id = 0, split = 1}
```

---

### `pto.tpop_from_aic`: C2V Consumer Pop

```mlir
%entry = pto.tpop_from_aic {id = <id>, split = <split>}
    -> <pipe_entry_type>
```

**Semantics:**

Pops a C2V pipe entry from the FIFO for consumption in the Vector core. This operation is an SSA value-returning operation that returns a tile buffer or tensor view descriptor, which subsequent operations can use for data transfer.

**Returns:** A pipe entry whose type is `!pto.tile_buf<...>` or `!pto.tensor_view<...>`, depending on the initialization configuration.

**Attributes:**

- `id`: Pipe identifier. It must match the `id` of `pto.aiv_initialize_pipe` in the same Vector kernel.

- `split`: Tile split mode (defined the same as `pto.tpush_to_aiv`).

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - It must appear in the Vector kernel.
  - `id` must match a `pto.aiv_initialize_pipe` operation that uses `dir_mask=1` or `dir_mask=3`.
  - The return type must be compatible with the initialization configuration.

**Example:**

```mlir
// Tile buffer return value.
%tile = pto.tpop_from_aic {id = 0, split = 1}
    -> !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                     v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>

// Tensor view return value.
%entry = pto.tpop_from_aic {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf32>
```

---

### `pto.tpop_from_aiv`: V2C Consumer Pop

```mlir
%entry = pto.tpop_from_aiv {id = <id>, split = <split>}
    -> <pipe_entry_type>
```

**Semantics:**

Pops a V2C pipe entry from the FIFO for consumption in the Cube core. The structure is the same as `pto.tpop_from_aic`, but in the opposite direction (the Cube consumes data from the Vector).

**Returns:** A pipe entry whose type is `!pto.tile_buf<...>` or `!pto.tensor_view<...>`.

**Attributes:**

- `id`: Pipe identifier. It must match the `id` of `pto.aic_initialize_pipe` in the same Cube kernel.

- `split`: Tile split mode.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - It must appear in the Cube kernel.
  - `id` must match a `pto.aic_initialize_pipe` operation that uses `dir_mask=2` or `dir_mask=3`.

**Example:**

```mlir
// V2C consumption on the Cube side.
%tile = pto.tpop_from_aiv {id = 0, split = 1}
    -> !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                     v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>
```

---

### `pto.tfree_from_aic`: C2V Consumer Release

```mlir
pto.tfree_from_aic {id = <id>, split = <split>}

// Or (for global memory entries):
pto.tfree_from_aic(<entry> : <pipe_entry_type>)
    {id = <id>, split = <split>}
```

**Semantics:**

Releases the current C2V FIFO consumer slot in the Vector core. For tile buffer entries, use the no-operand form; for global memory entries, use the form that includes the entry descriptor. This operation advances the consumer pointer, making the next entry available.

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `entry` (optional) | `!pto.tensor_view<...>` | Release descriptor for a global memory entry (required only when releasing a GM entry). |

**Returns:** None. The operation performs the release and returns.

**Attributes:**

- `id`: Pipe identifier. It must match the `id` of `pto.aiv_initialize_pipe` in the same Vector kernel.

- `split`: Tile split mode.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - It must appear in the Vector kernel.
  - `id` must match one `pto.aiv_initialize_pipe` operation using `dir_mask=1` or `dir_mask=3`.
  - Tile buffer release uses the no-operand form; global memory release must provide an entry operand.

**Example:**

```mlir
// Release a tile buffer entry (no operand).
pto.tfree_from_aic {id = 0, split = 1}

// Release a global memory entry (with an entry descriptor).
pto.tfree_from_aic(%entry : !pto.tensor_view<16x16xf32>)
    {id = 0, split = 0}
```

---

### `pto.tfree_from_aiv`: V2C Consumer Release

```mlir
pto.tfree_from_aiv {id = <id>, split = <split>}

// Or (for global memory entries):
pto.tfree_from_aiv(<entry> : <pipe_entry_type>)
    {id = <id>, split = <split>}
```

**Semantics:**

Releases the current V2C FIFO consumer slot in the Cube core. The structure is the same as `pto.tfree_from_aic`, but in the opposite direction.

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `entry` (optional) | `!pto.tensor_view<...>` | Release descriptor for global memory entries. |

**Returns:** None. The operation performs the release and returns.

**Attributes:**

- `id`: Pipe identifier. It must match the `id` of `pto.aic_initialize_pipe` in the same Cube kernel.

- `split`: Tile split mode.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - It must appear in the Cube kernel.
  - `id` must match a `pto.aic_initialize_pipe` operation that uses `dir_mask=2` or `dir_mask=3`.

**Example:**

```mlir
// Cube-side V2C release (no operand).
pto.tfree_from_aiv {id = 0, split = 1}

// Cube-side V2C release (global memory entry).
pto.tfree_from_aiv(%entry : !pto.tensor_view<16x16xf32>)
    {id = 0, split = 0}
```

---

### `pto.reserve_buffer`: Reserving Local Consumer Buffer

```mlir
pto.reserve_buffer {name = <name>, size = <size>,
                    location = <location>, auto = <autoAlloc>} -> i32
```

**Semantics:**

```text
addr = reserve_local_buffer(name, size, location, autoAlloc)
// Reserve a local consumer slot buffer in the current function,
// and return its address for CV Pipe initialization or peer import.
```

**Parameters:** No operands.

**Returns:** `i32`: the address of the reserved buffer.

**Attributes:**

- `name`: buffer name (string), which must be unique within the function.
- `size`: buffer size (`i32`), which must be greater than 0.
- `location`: address space, which must be `#pto.address_space<vec>` or `#pto.address_space<mat>`.
- `autoAlloc`: whether to allocate automatically (`bool`). It must be `true`, and the base address is allocated by the compiler during the memory planning phase.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - It must be nested inside `func.func`.
  - `size` must be greater than 0.
  - `location` must be `vec` or `mat`.
  - `name` must be unique within the enclosing function.

**Example:**

```mlir
%buf = pto.reserve_buffer
    {name = "c2v_slot_buffer", size = 131072,
     location = #pto.address_space<vec>,
     auto = true} -> i32
```

---

### `pto.import_reserved_buffer`: Importing Peer Reserved Buffer

```mlir
pto.import_reserved_buffer {name = <name>, peer_func = <peer_func>} -> i32
```

**Semantics:**

```text
addr = import_peer_buffer(name, peer_func)
// Import the buffer address reserved by the peer function through pto.reserve_buffer.
// Used for both sides to share the same CV Pipe-related local buffer.
```

**Parameters:** No operands.

**Returns:** `i32`, the address of the peer reserved buffer.

**Attributes:**

- `name`: buffer name (string), which must match the `pto.reserve_buffer` name in the peer function.
- `peer_func`: peer function reference (`FlatSymbolRefAttr`), which must point to an existing `func.func`.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - It must be nested in `func.func`.
  - The function referenced by `peer_func` must exist.
  - A `pto.reserve_buffer` with a matching name must exist in the peer function.
  - The `(name, peer_func)` combination must be unique within the enclosing function.

**Example:**

```mlir
%import_buf = pto.import_reserved_buffer
    {name = "c2v_slot_buffer",
     peer_func = @vector_kernel} -> i32
```

---

## Complete End-to-End Example

### C2V: Cube Core Production, Vector Core Consumption

The following example shows a complete C2V Pipe communication process:

```mlir
// Cube core side (C2V producer).
func.func @cube_kernel(%gm_slot_buffer : !pto.ptr<f32>,
                       %src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                            v_row=16, v_col=16, blayout=row_major,
                                            slayout=none_box, fractal=1024, pad=0>)
    attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
  %c0 = pto.constant 0 : index
  %c1 = pto.constant 1 : index
  %c16 = pto.constant 16 : index

  // Create a tensor view of the global memory pipe.
  %gm_slots = pto.make_tensor_view %gm_slot_buffer,
    shape = [%c16, %c16], strides = [%c16, %c1]
    : !pto.tensor_view<16x16xf32>

  // Initialize the pipe (C2V mode).
  pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
    (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

  // Allocate a global memory entry for the push.
  %entry = pto.talloc_to_aiv {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf32>

  // Obtain the tile view of the entry and perform the data transfer.
  %entry_partition = pto.partition_view %entry,
    offsets = [%c0, %c0], sizes = [%c16, %c16]
    : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>

  pto.tstore ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                      v_row=16, v_col=16, blayout=row_major,
                                      slayout=none_box, fractal=1024, pad=0>)
             outs(%entry_partition : !pto.partition_tensor_view<16x16xf32>)

  // Push the entry to the Vector core.
  pto.tpush_to_aiv(%entry : !pto.tensor_view<16x16xf32>)
    {id = 0, split = 0}

  func.return
}

// Vector core side (C2V consumer).
func.func @vector_kernel(%gm_slot_buffer : !pto.ptr<f32>,
                         %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                              v_row=16, v_col=16, blayout=row_major,
                                              slayout=none_box, fractal=1024, pad=0>)
    attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  %c0 = pto.constant 0 : index
  %c1 = pto.constant 1 : index
  %c16 = pto.constant 16 : index

  // Create a tensor view of the global memory pipe.
  %gm_slots = pto.make_tensor_view %gm_slot_buffer,
    shape = [%c16, %c16], strides = [%c16, %c1]
    : !pto.tensor_view<16x16xf32>

  // Initialize the pipe (C2V consumer side).
  pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
    (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

  // Pop the entry from the pipe.
  %entry = pto.tpop_from_aic {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf32>

  // Obtain the tile view of the entry and perform data transfer.
  %entry_partition = pto.partition_view %entry,
    offsets = [%c0, %c0], sizes = [%c16, %c16]
    : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>

  pto.tload ins(%entry_partition : !pto.partition_tensor_view<16x16xf32>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                      v_row=16, v_col=16, blayout=row_major,
                                      slayout=none_box, fractal=1024, pad=0>)

  // Release the consumer slot.
  pto.tfree_from_aic(%entry : !pto.tensor_view<16x16xf32>)
    {id = 0, split = 0}

  func.return
}
```

---

### V2C: Vector Core Production, Cube Core Consumption

The following example shows a complete V2C pipe communication process:

```mlir
  // Vector core side (V2C producer).
  func.func @vector_kernel(%gm_slot_buffer : !pto.ptr<f32>,
                           %src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                                v_row=16, v_col=16, blayout=row_major,
                                                slayout=none_box, fractal=1024, pad=0>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = pto.constant 0 : index
    %c1 = pto.constant 1 : index
    %c16 = pto.constant 16 : index

    %gm_slots = pto.make_tensor_view %gm_slot_buffer,
      shape = [%c16, %c16], strides = [%c16, %c1]
      : !pto.tensor_view<16x16xf32>

    pto.aiv_initialize_pipe {id = 0, dir_mask = 2, slot_size = 1024}
      (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

    %entry = pto.talloc_to_aic {id = 0, split = 0}
      -> !pto.tensor_view<16x16xf32>

    %entry_partition = pto.partition_view %entry,
      offsets = [%c0, %c0], sizes = [%c16, %c16]
      : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>

    pto.tstore ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                        v_row=16, v_col=16, blayout=row_major,
                                        slayout=none_box, fractal=1024, pad=0>)
               outs(%entry_partition : !pto.partition_tensor_view<16x16xf32>)

    pto.tpush_to_aic(%entry : !pto.tensor_view<16x16xf32>)
      {id = 0, split = 0}

    func.return
  }

  // Cube core side (V2C consumer).
  func.func @cube_kernel(%gm_slot_buffer : !pto.ptr<f32>,
                         %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                              v_row=16, v_col=16, blayout=row_major,
                                              slayout=none_box, fractal=1024, pad=0>)
      attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %c0 = pto.constant 0 : index
    %c1 = pto.constant 1 : index
    %c16 = pto.constant 16 : index

    %gm_slots = pto.make_tensor_view %gm_slot_buffer,
      shape = [%c16, %c16], strides = [%c16, %c1]
      : !pto.tensor_view<16x16xf32>

    pto.aic_initialize_pipe {id = 0, dir_mask = 2, slot_size = 1024}
      (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

    %entry = pto.tpop_from_aiv {id = 0, split = 0}
      -> !pto.tensor_view<16x16xf32>

    %entry_partition = pto.partition_view %entry,
      offsets = [%c0, %c0], sizes = [%c16, %c16]
      : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>

    pto.tload ins(%entry_partition : !pto.partition_tensor_view<16x16xf32>)
              outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                                        v_row=16, v_col=16, blayout=row_major,
                                        slayout=none_box, fractal=1024, pad=0>)

    pto.tfree_from_aiv(%entry : !pto.tensor_view<16x16xf32>)
      {id = 0, split = 0}

    func.return
  }
```

---

### Local FIFO Form: `reserve_buffer` / `import_reserved_buffer` (Ascend 950PR&950DT Products Only)

The following example shows the producer side using `import_reserved_buffer` to bind the same-named FIFO reserved by the consumer side's `reserve_buffer`:

```mlir
  // Producer side (Cube): Use import_reserved_buffer to bind the FIFO reserved by the peer.
  func.func @cube_kernel() attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %c0_i32 = arith.constant 0 : i32
    %c2v_import = pto.import_reserved_buffer {
      name = "c2v_fifo",
      peer_func = @vector_kernel
    } -> i32

    pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
      (c2v_consumer_buf = %c2v_import : i32,
       v2c_consumer_buf = %c0_i32 : i32)

    %tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=i32, rows=16, cols=16,
        v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    pto.tpush_to_aiv(%tile : !pto.tile_buf<loc=acc, dtype=i32, rows=16, cols=16,
        v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) {id = 0, split = 0}
    return
  }

  // Consumer side (Vector): Use reserve_buffer to reserve the local FIFO.
  func.func @vector_kernel() attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0_i32 = arith.constant 0 : i32
    %c2v_local = pto.reserve_buffer {
      name = "c2v_fifo",
      size = 4096,
      location = #pto.address_space<vec>,
      auto = true
    } -> i32

    pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
      (c2v_consumer_buf = %c2v_local : i32,
       v2c_consumer_buf = %c0_i32 : i32)

    %recv = pto.tpop_from_aic {id = 0, split = 0}
      -> !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
           v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tfree_from_aic {id = 0, split = 0}
    return
  }
```

---

### `pto.section.cube`: Cube Core Code Region

```mlir
pto.section.cube {
  <body>
}
```

**Semantics:**

```text
#if defined(MACRO_CUBE)
  <body>
#endif
// Lowered to a code region guarded by conditional compilation macros, compiled and executed only on the Cube core.
```

**Parameters:** None. Contains a single-block region (body).

**Returns:** None.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The region is single-block and has no terminator.
  - Operations in the body are valid only in the Cube core (AIC) context.

**Example:**

```mlir
pto.section.cube {
  %tile = pto.alloc_tile
      : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=256,
                      v_row=16, v_col=256, blayout=col_major,
                      slayout=row_major, fractal=1024, pad=0>
  pto.tmatmul ins(%a, %b : ...) outs(%tile : ...)
}
```

---

### `pto.section.vector`: Vector Core Code Region

```mlir
pto.section.vector {
  <body>
}
```

**Semantics:**

```text
#if defined(MACRO_VECTOR)
  <body>
#endif
// Lowered to a code region guarded by a conditional compilation macro, compiled and executed only on the Vector core.
```

**Parameters:** None. Contains a single-block region (body).

**Returns:** None.

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - The region is single-block and has no terminator.
  - The operations in the body are valid only in the Vector core (AIV) context.

**Example:**

```mlir
pto.section.vector {
  pto.tload ins(%view : !pto.partition_tensor_view<16x16xf16>)
            outs(%dst : !pto.tile_buf<...>)
  pto.tadd ins(%a, %b : ...) outs(%c : ...)
}
```

---

### `pto.initialize_l2g2l_pipe`: Initializing an L2G2L Pipe

```mlir
pto.initialize_l2g2l_pipe {dir_mask = <N>, slot_size = <S>, slot_num = <M>
                           (, local_slot_num = <L>)? (, flag_base = <F>)?
                           (, nosplit = <B>)?}
                          (<gm_addr> : <gm_type> (, <local_addr> : <l_type>)?
                           (, <peer_local_addr> : <p_type>)?)
                          -> !pto.pipe
```

**Semantics:**

```text
pipe = init_l2g2l_pipe(dir_mask, slot_size, slot_num, gm_addr, ...)
// Initialize a Local→Global→Local pipe handle for cross-core data transfer relayed through GM.
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `gm_addr` | `!pto.ptr<...>` | Relay buffer address in GM space. |
| `local_addr` | Optional, local buffer address | Local DMA buffer (optimized path). |
| `peer_local_addr` | Optional, peer local buffer address | Local buffer of the peer core. |

**Returns:** `!pto.pipe`, pipe handle.

**Attributes:**

- `dir_mask`: direction mask (`i8`). Specifies the C2V/V2C direction.
- `slot_size`: size of each FIFO slot (`i32`, in bytes).
- `slot_num`: number of FIFO slots (`i32`).
- `local_slot_num`: Optional, number of local slots (`i32`).
- `flag_base`: Optional, base address of the synchronization flag (`i32`).
- `nosplit`: Optional, disables split mode (`bool`).

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `slot_size` and `slot_num` must be greater than 0.
  - `gm_addr` must be a valid GM space type.

**Example:**

```mlir
%pipe = pto.initialize_l2g2l_pipe
    {dir_mask = 1, slot_size = 16384, slot_num = 2}
    (%gm_ptr : !pto.ptr<f32>) -> !pto.pipe
```

---

### `pto.initialize_l2l_pipe`: Initializing an L2L Pipe

```mlir
pto.initialize_l2l_pipe {dir_mask = <N>, slot_size = <S>, slot_num = <M>
                         (, flag_base = <F>)? (, nosplit = <B>)?}
                        (<local_addr> : <l_type>
                         (, <peer_local_addr> : <p_type>)?)
                        -> !pto.pipe
```

**Semantics:**

```text
pipe = init_l2l_pipe(dir_mask, slot_size, slot_num, local_addr, ...)
// Initialize a Local→Local pipe handle for cross-core data transfer directly through local memory (without GM relay).
```

**Parameters:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `local_addr` | Local buffer address | Local DMA buffer. |
| `peer_local_addr` | Optional, peer local buffer address | Local buffer of the peer core. |

**Returns:** `!pto.pipe`, pipe handle.

**Attributes:**

- `dir_mask`: direction mask (`i8`).
- `slot_size`: size of each FIFO slot (`i32`, in bytes).
- `slot_num`: number of FIFO slots (`i32`).
- `flag_base`: optional, synchronization flag base address (`i32`).
- `nosplit`: optional, disables split mode (`bool`).

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - `slot_size` and `slot_num` must be greater than 0.
  - `local_addr` must be a valid local space type.

**Example:**

```mlir
%pipe = pto.initialize_l2l_pipe
    {dir_mask = 1, slot_size = 8192, slot_num = 2, flag_base = 0}
    (%local_buf : i32) -> !pto.pipe
```

---

### `pto.talloc_to_aiv`: C2V Producer FIFO Allocation

```mlir
pto.talloc_to_aiv {(id = <N>,)? split = <S>} -> <entry_type>
```

**Semantics:**

```text
entry = alloc_c2v_producer_slot(id, split)
// Allocate a GlobalTensor FIFO producer entry for the C2V direction on the Cube side.
```

**Parameters:** No operands.

**Returns:** `!pto.tensor_view<...>`, FIFO entry descriptor.

**Attributes:**

- `id`: Pipe ID (`i32`), with a default value of `0`. It is used to distinguish between multiple pipes.
- `split`: Split factor (`i8`).

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - It must be located inside `pto.section.cube` or a Cube kernel function.
  - It must have a corresponding `pto.aic_initialize_pipe` and the `dir_mask` must contain the C2V bit.

**Example:**

```mlir
%entry = pto.talloc_to_aiv {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf32>
```

---

### `pto.talloc_to_aic`: V2C Producer FIFO Allocation

```mlir
pto.talloc_to_aic {(id = <N>,)? split = <S>} -> <entry_type>
```

**Semantics:**

```text
entry = alloc_v2c_producer_slot(id, split)
// Allocate a GlobalTensor FIFO producer entry for the V2C direction on the Vector side.
```

**Parameters:** No operands.

**Returns:** `!pto.tensor_view<...>`: FIFO entry descriptor.

**Attributes:**

- `id`: Pipe ID (`i32`), default value `0`.
- `split`: Split factor (`i8`).

**Constraints:**

- **Implementation check (Atlas A2 products/Atlas A3 products/Ascend 950PR&950DT products)**
  - It must be inside `pto.section.vector` or a Vector kernel function.
  - It must have a corresponding `pto.aiv_initialize_pipe` and the `dir_mask` must contain the V2C bit.

**Example:**

```mlir
%entry = pto.talloc_to_aic {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf16>
```
