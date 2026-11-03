# TPUT_ASYNC_NOTIFY

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:12:55.987Z pushedAt=2026-11-02T03:00:52.144Z nl=2 -->

## Introduction

`TPUT_ASYNC_NOTIFY` is an asynchronous remote write primitive with notification. It first transfers a non-empty payload from local GM to remote GM, and then updates a remote 32-bit signal. The returned `AsyncEvent` represents the completion status of both the payload transfer and the signal update.

Data flow:

`srcGlobalData (local GM)` → DMA engine → `dstGlobalData (remote GM)` → update `dstSignalData (remote GM)`

Both `dstGlobalData` and `dstSignalData` are remote addresses passed in by the caller. When only the signal is updated without transferring a payload, use `TNOTIFY`.

## Template Parameters

`engine` selects the DMA backend at compile time:

| Engine | Platform Restriction | Supported `NotifyOp` |
|---|---|---|
| `DmaEngine::SDMA` (default) | A2/A3 and A5 | `Set`, `AtomicAdd` |
| `DmaEngine::URMA` | Ascend 950PR/Ascend 950DT, NPU_ARCH 3510 only; requires CANN Toolkit >= 9.1.0 | `Set`, `AtomicAdd` |
| `DmaEngine::RDMA` | Ascend 950PR/Ascend 950DT, NPU_ARCH 3510 only; currently supports HNS1825 RoCE | `Set` only |

## C++ Built-in APIs

Declared in `include/pto/comm/pto_comm_inst.hpp`:

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData,
          typename GlobalSrcData,
          typename GlobalSignalData,
          typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC_NOTIFY(GlobalDstData &dstGlobalData,
                                      GlobalSrcData &srcGlobalData,
                                      GlobalSignalData &dstSignalData,
                                      int32_t signalValue,
                                      NotifyOp notifyOp,
                                      const AsyncSession &session,
                                      uint32_t peer,
                                      WaitEvents &... events);
```

For URMA and RDMA, `peer` is used to select the communication queue and remote memory information corresponding to the target rank, and both `dstGlobalData` and `dstSignalData` must belong to that peer. SDMA does not use `peer`; the remote address is determined by the GlobalTensor.

`events` is a nullable variadic list. Each event must provide a parameterless `Wait()`; the interface waits for these events in sequence before initiating the payload transfer. The wait interface of `AsyncEvent` is `Wait(session)`, so it cannot be passed as the `events` parameter and should be explicitly waited on using the corresponding Session before the call.

## Parameters

| Parameter | Description |
|---|---|
| `dstGlobalData` | Destination GlobalTensor of the payload in remote GM. |
| `srcGlobalData` | Source GlobalTensor of the payload in local GM. |
| `dstSignalData` | 32-bit signal in remote GM; the data type must be `int32_t`. |
| `signalValue` | Value written by `Set`, or the increment used by `AtomicAdd`. |
| `notifyOp` | Signal update operation: `NotifyOp::Set` or `NotifyOp::AtomicAdd`. |
| `session` | `AsyncSession` built for the template parameter `engine`. |
| `peer` | Target rank for URMA and RDMA; SDMA does not use this parameter. |
| `events` | Zero or more preceding PTO pipe events. |

The return value is an `AsyncEvent`. The completion scope of this Event covers both the payload transfer and the subsequent signal update.

### Signal and `signalValue`

`comm::Signal` is an alias of GlobalTensor corresponding to a single `int32_t` signal:

```cpp
using Signal = GlobalTensor<int32_t,
                            Shape<1, 1, 1, 1, 1>,
                            Stride<1, 1, 1, 1, 1>,
                            Layout::ND>;
```

Constructing a `Signal` only wraps the GM address provided by the caller and does not allocate or initialize the underlying memory. The caller should allocate the signal in advance and set the initial value according to the communication protocol. `Set` updates the signal to `signalValue`; `AtomicAdd` treats `signalValue` as a signed increment.

## Operation Semantics

A single call is executed in the following order:

1. Wait for all `events`.
2. Transfer the complete payload from `srcGlobalData` to `dstGlobalData`.
3. After the payload transfer is complete, update `dstSignalData` according to `notifyOp`.

The semantics of `NotifyOp::Set` are:

$$
\mathrm{signal}^{\mathrm{remote}} = \mathrm{signalValue}
$$

The semantics of `NotifyOp::AtomicAdd` are:

$$
\mathrm{signal}^{\mathrm{remote}} \mathrel{+}= \mathrm{signalValue} \quad (\text{ atomic operation})
$$

The `AtomicAdd` update to the signal is atomic. Multiple producers can concurrently update the same signal, and the final increment is the sum of all `signalValue` values. This atomicity does not apply to payload writes; when multiple producers concurrently transfer payloads, the destination address ranges must not overlap.

When multiple producers concurrently `Set` the same signal, the interface does not guarantee the final value. In the absence of application-level synchronization, `Set`, `AtomicAdd`, or ordinary store operations must not be concurrently mixed on the same signal.

`DmaEngine::RDMA` does not support `NotifyOp::AtomicAdd`. The above ordering of payload before signal applies only to the same invocation; the ordering between different Sessions or different execution streams is not defined.

## AsyncSession Construction

Use `BuildAsyncSession` in `include/pto/comm/async_common/async_event_impl.hpp`. This function provides different build interfaces based on the engine. It returns `false` when the build fails; only a successfully built Session can be used for asynchronous instructions and Event waiting.

### SDMA Construction (Default)

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(
    ScratchTile &scratchTile,
    __gm__ uint8_t *workspace,
    AsyncSession &session,
    uint32_t syncId = 0,
    const sdma::SdmaBaseConfig &baseConfig = {
        sdma::kDefaultSdmaBlockBytes, 0, 1},
    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

| Parameter | Default Value | Description |
|---|---|---|
| `scratchTile` |: | UB scratch tile used for SDMA control metadata. |
| `workspace` |: | GM pointer allocated by the host-side `SdmaWorkspaceManager`. |
| `session` |: | Output `AsyncSession`. |
| `syncId` | `0` | MTE3/MTE2 pipe synchronization event ID, ranging from 0 to 7. |
| `baseConfig` | `{kDefaultSdmaBlockBytes, 0, 1}` | SDMA block bytes, communication block offset, and queue count. |
| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA channel group index; selected based on `get_block_idx()` by default. |

On A2/A3, concurrent AIVs must build sessions separately and use different channel groups. The default `kAutoChannelGroupIdx` selects the group based on `get_block_idx()`; when the caller explicitly specifies a group, each concurrent AIV must also exclusively occupy one group. `queue_num` can be greater than 1, but the payload and signal of a single `TPUT_ASYNC_NOTIFY` use only queue 0 in that group to ensure payload precedes signal; increasing `queue_num` does not split this notify payload in parallel.

### URMA Construction (NPU_ARCH 3510 Only)

```cpp
#ifdef PTO_URMA_SUPPORTED
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    AsyncSession &session);
#endif
```

| Parameter | Description |
|---|---|
| `workspace` | GM pointer allocated by the host-side `UrmaWorkspaceManager`. |
| `session` | Output `AsyncSession`. |

URMA does not require `scratchTile` and requires CANN Toolkit >= 9.1.0. The Session is not bound to a target rank; the target is specified by the `peer` parameter of the instruction.

### RDMA Construction (NPU_ARCH 3510 Only)

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
| `scratchTile` | UB/Vec scratch tile used by RDMA, at least 64 bytes. |
| `workspace` | GM pointer returned by the Host-side RDMA initialization process. |
| `myPe` | Local rank id. |
| `session` | Output `AsyncSession`. |
| `syncId` | MTE/Scalar synchronization event ID, ranging from 0 to 7. |

An RDMA session is not bound to a target peer. The target rank is selected through `peer` when the instruction is invoked:

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::RDMA>(
        scratchTile, rdmaWorkspace, myPe, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::RDMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::Set, session, peer);
    (void)event.Wait(session);
}
```

### scratchTile

`scratchTile` is a temporary UB workspace used by SDMA and RDMA sessions, and is not used to store user payload. It must be a `pto::Tile` in UB/Vec memory and remain valid until the related events complete.

- SDMA requires at least 8 bytes of available space. A commonly used type is
  `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>` (256 bytes).
- RDMA requires at least 64 bytes of available space.

## Constraints

- `GlobalSrcData::RawDType` must be equal to `GlobalDstData::RawDType`.
- `GlobalSrcData::layout` must be equal to `GlobalDstData::layout`.
- The source and destination payload tensors must be flat, contiguous, logical one-dimensional tensors.
- The element capacity of the destination tensor must not be less than the number of elements in the source tensor.
- The payload size must be greater than 0. To update only the signal, use `TNOTIFY`.
- `dstSignalData` must contain exactly one `int32_t` in remote GM, with a non-null address aligned to 4 bytes. The caller is responsible for allocation and initialization.
- The payload destination address range must not overlap with `dstSignalData`.
- For URMA and RDMA, the payload destination address and the signal must belong to the same target peer; the complete address range of the local payload, remote payload, and remote signal must be located within the memory region registered during the host initialization phase.
- URMA payload has no single-transfer size limit: payloads exceeding 256 MB (the single WQE transfer limit) are automatically split into multiple WQEs by the implementation, transparent to the caller, and the signal is still guaranteed to complete last after all payload fragments. A single RDMA payload must not exceed `0x7fffffff` bytes.
- RDMA supports only `NotifyOp::Set`; `NotifyOp::AtomicAdd` must not be used.
- The SDMA workspace must be initialized by the host-side `SdmaWorkspaceManager`; the URMA workspace must be initialized by the host-side `UrmaWorkspaceManager`.
- The symmetric data buffer passed to `UrmaWorkspaceManager::Init()` must be device memory that can be registered by HCCL; the allocation method follows the current CANN/HCCL runtime requirements for registered memory.
- The lifetime of the session, workspace, and scratch tile must cover the completion phase of the related event.

## Completion Semantics

- `event.Wait(session)` blocks until the Event completes.
- `event.Test(session)` non-blockingly checks whether the Event has completed.
- Both APIs must use the same Session that initiated the operation.
- Successful completion covers both the full payload transfer and the subsequent signal update.

Asynchronous operations in the same Session and the same backend queue complete in submission order. After submitting multiple operations consecutively, waiting on the Event of the last operation in that queue also covers any earlier operations that have not yet completed. When URMA or RDMA accesses different peers, the queue of each peer is independent, and the caller must wait on the last Event of each peer separately; there is no need to pass the peer again when calling `Wait` or `Test`.

When the receiver observes the signal update, the corresponding payload has already been transferred to the remote GM, but this API does not guarantee that the receiver's existing cached copy of the payload is updated synchronously. Before reading the payload, the caller is responsible for ensuring payload visibility according to the memory consistency rules of the target platform and runtime.

## Concurrency and Session Ownership

- A2/A3 SDMA: When different AIVs use independent Sessions and Channel Groups, they can concurrently access the same rank or different ranks.
- URMA: Different AIVs can run concurrently when accessing different peers; they must be serialized when accessing the same peer.
- Concurrent payload ranges must not overlap; a shared signal should use `AtomicAdd`, and `Set` should use an independent signal.

- The same session cannot be used concurrently by multiple execution streams.
- For A2/A3 SDMA, each concurrent AIV must use an independent Session and an independent Channel Group; multiple AIVs must not submit to the same
  Group simultaneously. When `queue_num` is `N`, the maximum number of valid Groups is `kSdmaMaxChannelGroups / N`.
- When multiple AIVs run `Set` concurrently, each producer should use an independent signal; when using the same signal for completion counting,
  `AtomicAdd` should be used. In either mode, the payload destination address ranges of the AIVs must not overlap.
- For URMA and RDMA, even if the caller builds different Sessions based on the same workspace, submissions to the same peer/QP must be serialized;
  Different peers use independent queues.
- Before rebuilding a Session or reusing backend queues, all previous Events must be completed.
- When using URMA, before releasing communication resources, the last Event of each active peer/QP must be completed, and all host streams that use the
  `CommContext` must be synchronized. If an external HCCL communicator is passed in through `existingComm`, you must also ensure that other users have stopped, destroy the communicator first to release Channel/MR, and then call `DestroyComm`, `Reset`, call `BuildComm` again, or destroy the `CommContext`.

## Examples

The following example assumes that the Host communication runtime has completed remote address translation and initialized the workspace of the corresponding engine.

### SDMA Set

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void PutAndNotifySdma(__gm__ T *remoteDst,
                                       __gm__ T *localSrc,
                                       __gm__ int32_t *remoteSignalPtr,
                                       __gm__ uint8_t *sdmaWorkspace,
                                       uint32_t peer)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile =
        Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstGlobalData(remoteDst, shape, stride);
    GT srcGlobalData(localSrc, shape, stride);
    comm::Signal remoteSignal(remoteSignalPtr);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(
            scratchTile, sdmaWorkspace, session)) {
        return;
    }

    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::SDMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::Set, session, peer);
    (void)event.Wait(session);
}
```

### SDMA AtomicAdd

The Session and tensor construction method for SDMA `AtomicAdd` is the same as in the previous example:

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::SDMA>(
        scratchTile, sdmaWorkspace, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::SDMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::AtomicAdd, session, peer);
    (void)event.Wait(session);
}
```

### URMA Set

A URMA Session is not bound to a target rank. When invoking the instruction, specify the target through `peer`:

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::URMA>(
        urmaWorkspace, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::URMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::Set, session, peer);
    (void)event.Wait(session);
}
```

### URMA AtomicAdd

URMA `AtomicAdd` uses the same Session construction method:

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::URMA>(
        urmaWorkspace, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::URMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::AtomicAdd, session, peer);
    (void)event.Wait(session);
}
```

### RDMA Set

RDMA supports only `Set` and selects the target rank through `peer`:

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::RDMA>(
        scratchTile, rdmaWorkspace, myPe, session)) {
    auto event = comm::TPUT_ASYNC_NOTIFY<comm::DmaEngine::RDMA>(
        dstGlobalData, srcGlobalData, remoteSignal, 1,
        comm::NotifyOp::Set, session, peer);
    (void)event.Wait(session);
}
```

### Receiver

```cpp
comm::Signal ready(localSignalPtr);
comm::TWAIT(ready, 1, comm::WaitCmp::EQ);

// Read the payload after ensuring its visibility according to the memory consistency rules of the target platform and runtime.
```
