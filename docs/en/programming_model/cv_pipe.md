# CV Pipe

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T06:27:51.394Z pushedAt=2026-11-02T03:00:52.268Z nl=2 -->

## Overview

CV Pipe describes the intra-core data channel between Cube (AIC) and Vector (AIV).

Only the directly writable APIs need to be considered here, namely:

- How to declare the same pipe on both sides
- How the producer performs `push`
- How the consumer performs `pop`
- How to perform `free` after consumption is complete

This chapter does not discuss underlying implementation APIs.

## CV Pipe Focuses

CV Pipe mainly describes the following aspects:

- Which side is the producer and which side is the consumer
- Whether data flows in the C2V or V2C direction
- How the two sides of the same pipe are aligned through `id`
- How an entry goes through `push -> pop -> free`
- Whether an entry passes a local tile or a tensor view descriptor

From the user's perspective, the core value of CV Pipe is:

- To establish a structured data exchange path between Cube cores and Vector cores
- To express production, consumption, and recycling through a fixed set of API operations
- To avoid requiring users to handle excessive underlying details

## APIs

CV Pipe provides the same set of semantic operations in both API forms, with different naming conventions:

| Semantics | C++ Built-in APIs | IR Operation |
| --- | --- | --- |
| Initialize pipe | (Declared by the `TPipe` template parameters) | `pto.aic_initialize_pipe` / `pto.aiv_initialize_pipe` |
| Producer push | `TPUSH(Pipe&, ...)` | `pto.tpush_to_aiv` / `pto.tpush_to_aic` |
| Consumer pop | `TPOP(Pipe&, ...)` | `pto.tpop_from_aic` / `pto.tpop_from_aiv` |
| Free after consumption | `TFREE(Pipe&, ...)` | `pto.tfree_from_aic` / `pto.tfree_from_aiv` |
| Allocate global slot | `TALLOC(Pipe&, GlobalData&)` | - |

The following IR-side operation names use a "who sends to whom" convention, making the direction more intuitive:

- `tpush_to_aiv`: The Cube side pushes entries to the Vector side.
- `tpush_to_aic`: The Vector side pushes entries to the Cube side.
- `tpop_from_aic`: The Vector side pops entries from the Cube side.
- `tpop_from_aiv`: The Cube side pops entries from the Vector side.

## Basic Programming Model

A minimal CV Pipe usage flow typically includes four steps:

1. Initialize the pipe with the same `id` on both the Cube side and the Vector side.
2. Perform `push` on the producer side.
3. Perform `pop` on the consumer side.
4. After consumption is complete, perform the corresponding `free`.

Where:

- Initialization determines the pipe direction, entry size, and slot configuration.
- `push` indicates producing an entry.
- `pop` indicates borrowing a consumable entry.
- `free` indicates returning the entry after consumption is complete.

## Constraints That Users Must Follow

### `id`s on Both Sides Must Be Consistent

The Cube side and Vector side of the same logical pipe must use the same `id`; otherwise, they cannot be paired correctly.

### Directions on Both Sides Must Be Consistent

If the pipe is declared as C2V:

- The Cube side is responsible for production.
- The Vector side is responsible for consumption.

If it is declared as V2C, the directions are reversed.

### Entry Types Must Be Consistent

Both sides must have a consistent understanding of the entries in the same pipe, for example:

- Both are local tile objects of the same type.
- Or both correspond to compatible global tensor views.

Otherwise, subsequent validation will fail.

### `pop` Must Be Followed with a Matching `free`

`pop` obtains a pipe entry that is being consumed. After consumption is complete, you must explicitly call `free`; otherwise, the entry will not be recycled correctly.

### `split`/`nosplit` Semantics Must Be Consistent

If the initialization phase restricts the use to non-split mode, subsequent `push`/`pop`/`free` operations must also satisfy the same constraint.

## Application Scenario

CV Pipe is suitable for the following scenarios:

- The Cube side produces intermediate results, which the Vector side continues to process.
- The Vector side produces data, which is then handed over to the Cube side for further computation.
- A stable C2V/V2C data path needs to be established within the core.

For ordinary global memory reads/writes, local tile computation, or conventional `tload`/`tstore` flows, CV Pipe is usually not required.
