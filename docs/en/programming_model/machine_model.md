# PTO Machine Model

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T06:28:19.663Z pushedAt=2026-11-02T03:00:52.269Z nl=2 -->

## Overview

The PTO machine model defines the abstract execution environment that a PTO program faces on the device side. It does not directly expose all microarchitectural details of a specific hardware generation; instead, it organizes cores, execution units, on-chip storage, global storage, and synchronization resources into a stable set of logical objects for use by the **PTO ISA**, the type system, and the operational semantics.

## Machine Hierarchy

The PTO machine model can be divided into three conceptual layers:

- **Host**: responsible for compilation, parameter preparation, task submission, and result collection
- **Device**: responsible for mapping tile-level work to available execution resources and maintaining the observable order
- **Core**: responsible for executing a single tile instruction sequence and is the smallest logical execution agent

The **Device** and **Core** layers are the most frequently encountered, because they directly affect:

- Work division of block/subblock
- Which type of local resource is the tile data placed in
- Which operations can run in parallel
- Where synchronization is required

However, from an end-to-end perspective of PTO, **host** is not merely a backdrop but an essential component in the task construction and submission pipe.

## Host Perspective

### Host Responsibilities

In the PTO machine model, the host is responsible for preparing and submitting work to devices. It typically performs the following responsibilities:

- Constructs or loads PTO programs.
- Triggers compilation, caching, or reuse of compiled results.
- Prepares kernel parameters and global memory objects.
- Organizes tile graphs, task graphs, or higher-level scheduling units.
- Submits tasks to one or more device instances and waits for completion.

Therefore, the host is not the party that directly executes tile computation, but it determines:

- Which program is sent for execution
- With what parameters and data range the program executes
- How to organize the execution order among multiple device tasks

### What the Host Does Not Directly Promise

The host perspective in the PTO machine model does not require the documentation to define a unified host API for each platform. Instead, it emphasizes the boundaries of responsibilities:

- The host is responsible for preparation and submission.
- The device is responsible for scheduling and execution.
- The core is responsible for the actual tile instruction stream.

Therefore, the focus of **PTO ISA** is not to define the details of the host-side API, but to define a sufficiently stable program object model so that the host can reliably hand off work to the subsequent compilation and execution pipe.

## Core View

### Scalar Control

In the PTO machine model, a core not only executes tile operators but also provides scalar control capabilities for the following purposes:

- Control flow
- Address computation
- Event and synchronization coordination

Therefore, scalar values and tile objects in a PTO program are not two unrelated worlds; the former typically drives the data selection, view offset, and execution order of the latter.

### Logical Execution Units

From the programming model perspective, a core typically contains multiple types of logical execution units:

- **cube**: the main path for matrix computation
- **vector**: the path for element-wise operations, reduction, and data movement
- **scalar/fix**: the path for control, auxiliary computation, and partial synchronization
- **memory**: the path for data movement between global memory and on-chip memory

These names are not required to correspond one-to-one with the names of underlying hardware units, but in PTO they correspond to clear semantic boundaries.

## Device View

### Tile Block

In the PTO machine model, device execution is not "a single kernel running sequentially in a single thread"; instead, the work is split into multiple tile blocks and then dispatched to the device for scheduling.

Each tile block:

- Processes a portion of the input/output region.
- Has its own block identifier.
- Can execute in parallel with other independent blocks.

### Block and Subblock

Under the hybrid execution model, a logical block is often not a single execution entity but is further distinguished as follows:

- `block_idx`
- `subblock_idx`

Where:

- `block_idx` usually indicates the number of the current logical block.
- `subblock_idx` usually indicates the number of a finer-grained execution participant within the block.

This is also why runtime query APIs such as `get_block_idx()` and `get_subblockid()` are directly incorporated into the PTO ISA.

### Parallelism of Independent Blocks

If there is no explicit dependency between two tile blocks, the device can:

- Execute them in parallel.
- Execute them out of order.
- Map them to different cluster/core resources.

What PTO truly promises to users is not a "fixed scheduling order", but rather that "observable semantics are not violated".

## Storage Hierarchy

The storage in the PTO machine model is not a unified memory, but a set of named logical domains. The most core objects in the current document include:

- **GM**
- **MAT**
- **LEFT**
- **RIGHT**
- **ACC**
- **BIAS**
- **VEC**

These logical domains jointly determine:

- Which resource level the data currently resides in
- Which transfer paths are legal
- Which compute ops can consume the object
- Which read/write combinations may introduce synchronization requirements

Therefore, a tile in PTO is not an abstract value detached from its resource context, but is always bound to a position in the machine model.

## Data Flow Paths

From the perspective of the machine model, a PTO program typically consists of three types of activities:

- Bringing in data from **GM**
- Moving, rearranging, or retaining data between on-chip local domains
- Consuming tiles on specific execution units and generating results

Typical paths include:

- `GM -> VEC`
- `GM -> MAT -> LEFT/RIGHT/BIAS`
- `ACC -> GM`
- `VEC -> GM`

This is also why data movement operations, local tile types, and layout attributes in PTO must all be explicitly modeled.

## Synchronization Resources

The PTO machine model treats synchronization resources as first-class objects rather than as an add-on mechanism. Common synchronization boundaries include:

- Ordering constraints between pipes
- Dependencies between different execution units within a block
- Pipe coordination between producer/consumer
- Wait relationships defined by explicit events/flags

Therefore, whether the device can advance two pieces of work in parallel depends not only on the textual order, but also on:

- Whether they access the same type of resource
- Whether they form a real data dependency
- Whether a synchronization protocol that must be satisfied exists between them
