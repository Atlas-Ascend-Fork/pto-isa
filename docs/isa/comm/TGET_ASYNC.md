# TGET_ASYNC

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:05:29.936Z pushedAt=2026-11-02T03:00:52.142Z nl=2 -->

## Introduction

`TGET_ASYNC` is an asynchronous remote read primitive. It initiates a transfer from remote GM to local GM and returns an `AsyncEvent` immediately.

Data flow:

`srcGlobalData (remote GM)` → DMA engine → `dstGlobalData (local GM)`

## Template Parameters

- `engine`:
    - `DmaEngine::SDMA` (default)
    - `DmaEngine::URMA` (Ascend 950PR/Ascend 950DT, NPU_ARCH 3510 only)
    - `DmaEngine::RDMA` (Ascend 950PR/Ascend 950DT, NPU_ARCH 3510 only; the current NIC platform supports only HNS1825.)

> **Note (SDMA path)**
> `TGET_ASYNC` with `DmaEngine::SDMA` currently **only supports flat, contiguous, logical 1D tensors**.
> The current SDMA asynchronous implementation does not support non-1D or non-contiguous layouts.

## C++ Built-in APIs

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TGET_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);
```

`AsyncSession` is an engine-agnostic session object. Build it once using `BuildAsyncSession<engine>()` and pass it to all asynchronous calls and event waits. The template parameter `engine` selects the DMA backend at compile time, keeping the code forward-compatible with future engines (such as CCU).

## AsyncSession Construction

Use `BuildAsyncSession` in `include/pto/comm/async_common/async_event_impl.hpp`.
This function has two overloads – one for SDMA and one for URMA, with different parameter lists.

### SDMA Construction (Default)

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    const sdma::SdmaBaseConfig &baseConfig = {sdma::kDefaultSdmaBlockBytes, 0, 1},
                                    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

| Parameter | Default Value | Description |
|---|---|---|
| `scratchTile` |: | UB scratch tile used for SDMA control metadata (see [Role of scratchTile](#role-of-scratchtile)). |
| `workspace` |: | GM pointer allocated by the host-side `SdmaWorkspaceManager`. |
| `session` |: | Output `AsyncSession` object. |
| `syncId` | `0` | MTE3/MTE2 pipe synchronization event ID (0-7). If the kernel uses other pipe barriers on the same ID, this value must be overridden. |
| `baseConfig` | `{kDefaultSdmaBlockBytes, 0, 1}` | `{block_bytes, comm_block_offset, queue_num}`. It applies to most single-queue transfer scenarios. |
| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA channel group index. By default, `get_block_idx()` is used internally to map to the current AI Core. This value must be overridden in multi-block, concurrent, or custom channel mapping scenarios. |

### URMA Construction (NPU_ARCH 3510 Only)

> User-level RDMA Memory Access (URMA) is a hardware-accelerated RDMA transfer engine on Ascend 950PR/Ascend 950DT (NPU_ARCH 3510).
> URMA requires CANN Toolkit **>= 9.1.0**.

```cpp
#ifdef PTO_URMA_SUPPORTED
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    uint32_t destRankId,
                                    AsyncSession &session);
#endif
```

| Parameter | Description |
|---|---|
| `workspace` | GM pointer allocated by the host-side `UrmaWorkspaceManager`. |
| `destRankId` | Remote PE rank ID for this session's communication. For `TGET_ASYNC`, this is the source rank from which data originates. |
| `session` | Output `AsyncSession` object. |

URMA does not require `scratchTile` – polling is performed directly through the `ld_dev`/`st_dev` hardware primitives.

### RDMA Construction (NPU_ARCH 3510 Only)

RDMA targets classic cross-machine networking scenarios, and the current NIC platform supports only HNS1825.

```cpp
#ifdef PTO_RDMA_SUPPORTED
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    uint32_t myPe,
                                    AsyncSession &session,
                                    uint32_t syncId = 0);
#endif
```

| Parameter | Description |
|---|---|
| `scratchTile` | UB/Vec tile used for WQE/CQE control data, at least 64 bytes. |
| `workspace` | GM pointer returned by the host-side RDMA initialization process. |
| `myPe` | Local rank ID, used to select the registered local memory region. |
| `session` | Output `AsyncSession` object. |
| `syncId` | MTE/Scalar synchronization event ID, ranging from 0 to 7. |

Construct a session that is not bound to a peer once, and then select the source rank through the explicit `peer` parameter:

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::RDMA>(scratchTile, rdmaWorkspace, myPe, session, syncId)) {
    auto event = comm::TGET_ASYNC<comm::DmaEngine::RDMA>(dstG, srcG, session, peer);
    (void)event.Wait(session);
}
```

## Constraints

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- Both the SDMA and URMA paths require the source tensor to be **flat, contiguous, logical 1D**.
- The SDMA workspace must be a valid GM pointer allocated by the host-side `SdmaWorkspaceManager`.
- The URMA workspace must be a valid GM pointer allocated by the host-side `UrmaWorkspaceManager`.
- The lifetime of the session and workspace must cover the completion phase of the related events.
- URMA is available only on NPU_ARCH 3510 (Ascend 950PR/Ascend 950DT).
- URMA requires CANN Toolkit **>= 9.1.0**.
- The symmetric data buffer passed to `UrmaWorkspaceManager::Init()` must be backed by huge page memory (allocated with `ACL_MEM_MALLOC_HUGE_ONLY`). The underlying MR registration requires a huge page backing. `ACL_MEM_MALLOC_HUGE_FIRST` may silently fall back to 4 KB small pages for small allocations, causing registration to fail.
- RDMA is available only on Ascend 950PR/Ascend 950DT (NPU_ARCH 3510), and the current NIC platform supports only HNS1825.
- The RDMA source and destination tensors must be flat, contiguous, logical one-dimensional, and the complete local and remote transfer ranges must lie within the memory regions registered during the host initialization phase.
- A single RDMA transfer cannot exceed `0x7fffffff` bytes.

If the one-dimensional contiguous requirement is not met, the current implementation returns an invalid async event (`handle == 0`).

## Role of scratchTile

`scratchTile` is **not** a staging buffer for transferring the data payload. It is converted to `TmpBuffer` and used as a temporary UB workspace for:

- Writing/reading SDMA control words (flag, sq_tail, channel_info)
- Polling the event completion flag
- Committing the queue tail upon completion

The actual data path is remote GM → DMA engine → local GM; `scratchTile` is used only for controlling and synchronizing metadata.

## scratchTile Type and Size Constraints

- Must be of the `pto::Tile` type.
- Must be a UB/Vec tile (`ScratchTile::Loc == TileType::Vec`).
- Must have at least `sizeof(uint64_t)` (8 bytes) of available bytes.

Recommended: `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>` (256 bytes).

## Completion Semantics (Quiet Semantics)

The underlying completion mechanisms differ across engines, but the quiet semantics behavior on the user side is consistent:

- **SDMA**: Each `TGET_ASYNC` submits a data transfer SQE and a flag SQE used to mark the completion of this operation. When calling `Wait` or `Test` on the returned event, the corresponding flag is polled to determine whether this `TGET_ASYNC` has completed; after completion, it is also guaranteed that all previously submitted SDMA operations in the same session have completed.
- **URMA**: `TGET_ASYNC` immediately submits the RDMA READ WQE and rings the doorbell. `Wait` polls the Completion Queue (CQ) until all expected CQEs are consumed.
- **RDMA**: `TGET_ASYNC` submits an RDMA READ WQE to the queue selected by `peer`; completion status is tracked separately for different peers/queues.

- `event.Wait(session)` – blocks until **all asynchronous operations issued since the last Wait** are complete.

This means that after multiple `TGET_ASYNC` calls, you only need to call `Wait` once on the last returned `AsyncEvent` to wait for all pending operations to complete (similar to quiet semantics of shmem).

When RDMA operations involve different peers, the last event of each peer must be waited for separately.

A single session can have at most 64 uncompleted operations; submissions beyond this may cause backpressure.

After `Wait` succeeds, all data read into the issued `dstGlobalData` is fully ready.

## SDMA Concurrency and Session Ownership

- The same session cannot be used concurrently by multiple execution streams.
- Operations that share the same channel group must share the same session.
- Concurrent kernels or multiple independent sessions within a kernel must use isolated channel groups.
- Before rebuilding a session or reusing a channel group, all previous events must be completed first.

## Examples

### Single Transfer

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimpleGet(__gm__ T *localDst, __gm__ T *remoteSrc,
                                 __gm__ uint8_t *sdmaWorkspace)
{
    using ShapeDyn  = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT        = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(localDst,  shape, stride);
    GT srcG(remoteSrc, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    auto event = comm::TGET_ASYNC<comm::DmaEngine::SDMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```

### Batch Transfer (Quiet Semantics)

```cpp
template <typename T>
__global__ AICORE void BatchGet(__gm__ T *localDstBase, __gm__ T *remoteSrcBase,
                                __gm__ uint8_t *sdmaWorkspace, int nranks)
{
    using ShapeDyn  = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT        = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    comm::AsyncEvent lastEvent;
    for (int rank = 0; rank < nranks; ++rank) {
        GT dstG(localDstBase + rank * 1024, shape, stride);
        GT srcG(remoteSrcBase + rank * 1024, shape, stride);
        lastEvent = comm::TGET_ASYNC(dstG, srcG, session);
    }
    (void)lastEvent.Wait(session);  // Wait once for all pending operations.
}
```

### URMA Example (NPU_ARCH 3510)

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimpleGetUrma(__gm__ T *localDst, __gm__ T *remoteSrc,
                                     __gm__ uint8_t *urmaWorkspace, uint32_t srcRankId)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(localDst, shape, stride);
    GT srcG(remoteSrc, shape, stride);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(urmaWorkspace, srcRankId, session)) {
        return;
    }

    auto event = comm::TGET_ASYNC<comm::DmaEngine::URMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```
