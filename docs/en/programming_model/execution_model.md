# PTO Execution Model

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T06:28:14.286Z pushedAt=2026-11-02T03:00:52.269Z nl=2 -->

## Overview

The PTO execution model is not a direct mapping of a "single-thread sequential program"; instead, it organizes functions, execution domains, data dependencies, and synchronization relationships into an analyzable concurrent execution model. PTO only guarantees the preservation of the observable semantics of a program and does not hard-code any specific scheduling details into the C++/IR semantics.

## Execution Unit Hierarchy

### Compilation Unit

The top-level compilation unit is responsible for carrying functions, symbols, and top-level attributes. In C++, it corresponds to a compilation unit (source file/translation unit); in IR, it corresponds to `module`.

### Function Unit

A function is the most direct compilable procedure body (a regular function in C++ and `func.func` in IR), which carries:

- Input parameters
- tile/view/pipe/event objects
- Control flow and sequential structures

### Execution Domain

Many PTO programs do not execute on "undifferentiated general-purpose cores"; instead, they distinguish different execution domains (for example, the cube path handles matrix computation, while the vector path handles element-wise operations and data movement).

The two API forms handle execution domains differently:

- **C++ built-in APIs**: The execution domain is automatically inferred by the compiler based on the operations used (for example, using `TMATMUL` takes the cube path, while using `TADD` takes the vector path), so users do not need to declare it explicitly.
- **IR operations**: The execution domain must be explicitly marked, entered via `pto.section.cube` / `pto.section.vector` or the equivalent `kernel_kind` function attribute.

## SPMD and MPMD

PTO supports two common work mapping approaches:

- **SPMD**: Single Program, Multiple Data
- **MPMD**: Multiple Program, Multiple Data

These two approaches describe how work is mapped to execution resources.

### SPMD

In SPMD, all blocks participating in execution run the same kernel entry, but each block processes a different data region based on its runtime identity.

This is typically expressed as:

- Obtaining the block index through `get_block_idx()` (`pto.get_block_idx` in IR)
- Combining it with `get_subblockid()` (`pto.get_subblock_idx` in IR) when necessary
- Computing the view offset or tile offset that the current block is responsible for

For example:

C++ form:

```cpp
auto bid = get_block_idx();
auto base = bid * tile_rows;
```

IR form:

```mlir
%bid_i64 = pto.get_block_idx
%bid = pto.index_cast %bid_i64 : i64 to index
%base = pto.muli %bid, %tile_rows : index
```

This approach is suitable for:

- Regular tensor tiling
- Computations with consistent structure across blocks
- Most matrix computations, element-wise computations, and regular data movement paths

### Subblock Participation

If a block contains finer-grained execution participants, they can be further combined:

- Use `get_subblockid()` / `get_subblockdim()` (in IR, `pto.get_subblock_idx` / `pto.get_subblock_num`)

to construct finer-grained logical execution IDs.

This approach is commonly used to:

- Further split row/column work within a block.
- Distinguish the input subregions handled by different subblocks.
- Compute the local offset or GM partition corresponding to each subblock.

### MPMD

In MPMD, different execution resources can execute different program stages within the same overall task.

For PTO users, this mode can generally be understood as:

- Different blocks are responsible for different stages.
- Or the host side separately submits multiple kernels with different responsibilities.
- Or in higher-level task organization, different execution stages are assigned to different entry functions.

MPMD is more suitable for:

- Producer/consumer staged execution
- Multi-stage programs that clearly distinguish different computational responsibilities
- Scenarios where a large task needs to be split into different kernel responsibility units

### How to Choose

If all blocks only process different data slices but execute the same logic, SPMD should be preferred.

If different execution units assume different program responsibilities rather than being repeated instances of the same program on different data, it is closer to MPMD.

## Two Programming Styles

The PTO execution model does not force all users to organize kernels in the same way. For users writing PTO ISA, the two most common styles are:

### Auto Style

The auto style places more emphasis on:

- First expressing tiles, views, and input/output relationships clearly
- Minimizing manual writing of low-level addresses and synchronization details
- Letting the compiler take on more work in layout, addressing, and synchronization

This style is more suitable for:

- Getting started with writing PTO kernels
- Prioritizing correctness and portability
- First stably expressing the algorithm, then gradually refining control

### Manual Style

The manual style places more emphasis on:

- Explicitly controlling tile placement and local addresses
- Explicitly controlling events, synchronization, and execution order
- Explicitly specifying which resources to reuse and which boundaries must not be left to compiler inference

This style is more suitable for:

- Performance tuning of critical paths
- Kernels that require stable control over addresses and synchronization details
- Programs that need to strictly align with specific execution structures

### Relationship Between the Two Styles and the Execution Model

These two styles are not two different languages, but two ways of using the same execution model:

- The auto style relies more on the compiler to complete execution details.
- The manual style relies more on the user to write execution details into the program.

Regardless of the style used, the following must ultimately be satisfied:

- Clear data dependency
- Clear resource location
- Clear synchronization boundary

## Program Order and Observable Order

A PTO program has at least three types of "order":

### Text Order

The order in which operations are written in the program is the most direct program order.

### Data Dependency Order

Value usage relationships, buffer read/write relationships, and view/tile connection relationships form a data dependency order.

### Synchronization Order

Even if certain operations are textually adjacent, the absence of a synchronization relationship does not necessarily mean they must be executed in a fully serial manner; once explicit synchronization exists, a stronger ordering boundary is formed.

What PTO actually guarantees are:

- Data dependencies cannot be violated.
- Event dependencies cannot be violated.
- The architecturally observable order defined by explicit synchronization cannot be violated.

## Parallelism in the Execution Model

PTO programs allow different resource domains, different pipes, and different on-chip objects to advance in parallel when dependencies permit. Precisely because of this:

- Not all tile operations can be understood as ordinary serial operators.
- Synchronization is a first-class component of the execution model.
- Memory side effects and address spaces affect scheduling analysis.

Combined with the PTO machine abstraction, the sources of parallelism can be summarized as follows:

- Different logical execution units can advance in parallel.
- Data flows on different local storage domains can interleave.
- Synchronization is established only at true dependency boundaries, rather than being globally serial by default.

## Requirements for Program Writing

From the perspective of the programming model, the execution model requires the program to clearly express:

- Which objects are true data dependencies
- Which objects are merely logical aliases or reinterpretation relationships
- Which stages must have explicit synchronization between them
- Which functions or execution domains are marked as running in different execution domains

If these boundaries are not clearly expressed, subsequent synchronization derivation, memory planning, and transformation will all lose a reliable foundation.

From the perspective of machine abstraction, it is also necessary to clarify:

- Whether the logical storage domain to which a tile belongs is correct
- Whether the data movement path matches the consumption pattern of the target execution unit
- Whether the synchronization object covers the real producer/consumer boundary

If these constraints do not hold, the program may fail to form a stable compilable execution model even if it is syntactically valid.

## Execution Model and Control Flow

PTO allows loops and branches (native control flow in C++, expressed through standard dialects in IR), so the execution model is not limited to straight-line code. Loops and branches further affect:

- Lifecycle analysis
- Synchronization derivation
- Tile reuse and memory layout

This is also why more conservative synchronization or memory strategies are usually required under complex control flow.
