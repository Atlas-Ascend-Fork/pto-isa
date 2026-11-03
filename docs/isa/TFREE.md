# TFREE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:57:51.388Z pushedAt=2026-11-02T03:00:52.182Z nl=2 -->

## Introduction

Releases the slot space in the FIFO.

For the TileData `TPOP` flow, on Atlas A2/A3 training products/Atlas A2/A3 inference products, `TPOP` already performs the free-space notification step internally. Therefore, the `TFREE(Pipe &pipe)` API for TileData is currently a null operation, and exists only to maintain API symmetry with the `GlobalData` flow. On Ascend 950PR/Ascend 950DT, TFREE releases the FIFO slot space used by TPOP.

For the `GlobalData` flow, `TFREE(Pipe&, GlobalData&)` releases the FIFO slot view returned by `TPOP(Pipe&, GlobalData&)`.

In CPU_SIM, the TileData form participates in the host-side FIFO release protocol when the pipe's free-status policy requires release, so the CPU TileData form is not a null operation in the A2A3 path. For `DIR_BOTH`, it releases the ring queue in the direction recorded by the corresponding `TPOP`. When multiple unreleased pops exist, each call releases the consumer's earliest pop and preserves the pop order and direction. CPU_SIM currently does not support GlobalData overloading.

## Operation Semantics

For the TileData flow:

1. `TPUSH(Pipe&, TileData&, Split)` stores the producer tile into the current FIFO slot and records the data-ready synchronization for the consumer. The producer tile index is incremented after the slot address is computed.
2. `TPOP(Pipe&, TileData&, Split)` waits for the producer's data-ready synchronization and loads the current FIFO slot into the consumer tile. The consumer tile index is incremented after the slot address is computed.
3. `TFREE(Pipe&, Split)` releases the slot space in the FIFO. On Atlas A2/A3 training products/Atlas A2/A3 inference products, this API is a null operation (`TPOP` already performs the free-space notification internally). On Ascend 950PR/Ascend 950DT, it releases the FIFO slot space used by `TPOP`.

For the GlobalData flow:

1. `TALLOC(Pipe&, GlobalData&)` allocates a producer FIFO slot from `TPipe` and exposes it as a `GlobalTensor` view. The producer can write data to this slot through instructions such as `TSTORE`.
2. `TPUSH(Pipe&, GlobalData&)` records the data-ready synchronization for the slot already allocated by `TALLOC` and submits the FIFO slot to the consumer. It does not store tile data itself.
3. `TPOP(Pipe&, GlobalData&)` waits for data readiness, assigns `gmTensor` to the current FIFO slot address, and increments the consumer tile index. It does not load data into a local tile, nor does it release the slot. The consumer can read data from the slot through instructions such as `TLOAD`.
4. `TFREE(Pipe&, GlobalData&)` releases the FIFO slot view returned by `TPOP(Pipe&, GlobalData&)`, notifying the producer that the slot space is free.

## C++ Intrinsic

Declaration location: `include/pto/common/pto_instr.hpp`:

```cpp
template <typename Pipe, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, WaitEvents &... events);

template <typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

The Atlas A2/A3 training products/Atlas A2/A3 inference products TileData implementation in `include/pto/npu/a2a3/TFree.hpp` intentionally keeps this overload empty (on Ascend 950PR/Ascend 950DT, the implementation is located in `include/pto/npu/a5/TFree.hpp` and performs the actual free-space notification):

```cpp
template <typename Pipe, TileSplitAxis Split>
PTO_INTERNAL void TFREE_IMPL(Pipe &pipe)
{
    return;
}
```

## Constraints

- **TileData flow**:
    - When the data in the popped FIFO slot is no longer needed, use `TFREE(Pipe&)`.
    - Use TPUSH/TPOP/TFREE together to implement inter-core synchronization and data transfer. During data transfer, the size ratio between the pushed tileshape and the popped tileshape is 1:1 or 1:2.

- **GlobalData flow**:
    - Use `TFREE(Pipe&, GlobalData&)` when the data in the popped FIFO slot is no longer needed.
    - `gmTensor` is only used to select the overload; the implementation does not read or write the tensor content.
    - Free-space notification is sparse and controlled by `Pipe::SyncPeriod`.
    - If the relationship is not 1:1 or 1:2, that is, if subtile data transfer exists, TALLOC/TPUSH/TPOP/TFREE must be used together to implement inter-core synchronization and data transfer.

## Examples

### TileData Flow

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_tiledata(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(T), FifoDepth>;
    using VecTile = Tile<TileType::Vec, T, M / 2, N, BLayout::RowMajor, M / 2, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    VecTile tile;

    TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, tile);
    ...  // final use of VecTile
    TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN>(pipe);
}
```

### GlobalData Slot Release

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_globaldata(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(T), FifoDepth>;
    using SlotGlobal = GlobalTensor<T, Shape<1, 1, 1, M / 2, N>, Stride<1, 1, 1, N, 1>>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    SlotGlobal slot;

    TPOP<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot); // TPOP reassigns slot.
    // Load or consume data from slot here.
    TFREE<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
}
```
