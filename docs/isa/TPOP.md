# TPOP

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T05:31:38.882Z pushedAt=2026-11-02T03:00:52.212Z nl=2 -->

## Introduction

Pops a consumer tile from the `TPipe` FIFO for Cube-Vector communication.

This instruction supports the TileData overloads that explicitly specify `TileSplitAxis`, explicitly specify the subblock ID, and reverse parameters without specifying `Split`, as well as the `GlobalData` overload.

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
template <typename Pipe, typename TileCons, TileSplitAxis Split,
          std::enable_if_t<is_tile_data_v<TileCons>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, TileCons &tile, WaitEvents &... events);

template <typename Pipe, typename TileCons, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, TileCons &tile, int32_t subBlockId,
                          WaitEvents &... events);

template <typename TileData, typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TPOP(TileData &tile, Pipe &pipe, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);
```

`Pipe` is usually the `TPipe` declared in the selected backend `TPush.hpp`, including `include/pto/cpu/TPush.hpp`,
`include/pto/npu/a2a3/TPush.hpp`, or `include/pto/npu/a5/TPush.hpp`:

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## Constraints

- **TileData consumer**:
    - `TileCons::Loc` must be `TileType::Vec`, `TileType::Mat`, or `TileType::Ctrl`.
    - `Direction::DIR_C2V`: vector consumes data produced by cube.
    - `Direction::DIR_V2C`: cube consumes data produced by vector.
    - `Direction::DIR_BOTH`: the same pipe type supports both C2V and V2C consumers.
- **Local consumer buffer**:
    - For C2V vector consumers, `TPipe` allocates the tile to `C2V_CONSUMER_BUF` and uses local FIFO rotation.
    - For a V2C matrix consumer, `TPipe` allocates the tile to `V2C_CONSUMER_BUF` and uses local FIFO rotation.
- **Split behavior**:
    - `TileSplitAxis::TILE_NO_SPLIT`: No split is performed. On Atlas A2/A3 training products/Atlas A2/A3 inference products, AIV0/AIV1 companion core inter-core synchronization is required.
    - `TileSplitAxis::TILE_UP_DOWN`: Consumes the upper and lower row halves.
    - `TileSplitAxis::TILE_LEFT_RIGHT`: Consumes the left and right column halves.
- **NPU synchronization**:
    - Each `TPOP` performs a data-ready wait.
    - Free-space notification is sparse and controlled by `Pipe::SyncPeriod`.
- **GlobalData slot view**:
    - `gmTensor` is assigned the FIFO slot base address selected by `pipe.cons.tileIndex`.
    - For the C2V split mode, the vector sub-block offset is applied based on `Split`.
    - After all loads are completed from the slot view, the caller must call `TFREE(Pipe&, GlobalData&)`.
- **CPU_SIM FIFO model**:
    - `TPOP` uses the host FIFO mutex and condition variable to wait for the producer to submit a slot.
    - TileData data is loaded from the storage owned by the host FIFO state; even if `TPipe` carries a non-empty NPU GM workspace, the CPU_SIM TileData flow does not access that workspace.
    - The split mode selects the lane based on the current subblock context. For a C2V pipe with `IsNoSplit` enabled, `TILE_NO_SPLIT` coordinates one or two vector consumer subblocks according to the runtime subblock count; in other no-split vector scenarios, non-zero non-participating lanes are zero-filled as needed.
    - CPU_SIM currently does not implement the overload that explicitly takes `int32_t subBlockId`; the split lane should be selected through the simulated subblock execution context.
    - For `DIR_BOTH`, the consumer waits in an independent ring queue for its own direction. No-split and V2C consumers take the earliest submitted slot by submission sequence number, maintaining FIFO order even when delayed release leaves gaps in the queue. Overlapping pops reserve different slots; `TFREE` releases, in the corresponding direction, the earliest pop of the calling consumer that has not yet been released.
    - Each enabled `TFREE` releases one occupied concrete slot in CPU_SIM; CPU_SIM does not simulate the NPU's `SyncPeriod` notification cadence.
    - TileData is read back according to the declared shape of the consumer Tile, so this shape must match the window pushed by the producer (under `TILE_NO_SPLIT`, this is its effective shape).
    - CPU_SIM copies TileData to the destination layout by logical coordinates. The `TPOP` of the A5 local FIFO binds the Tile to the FIFO address and does not perform layout conversion. To make V2C behavior consistent with the NPU, use matching producer and consumer layouts, for example NZ Vec to NZ Mat, and complete the required layout conversion before `TPUSH`.
    - CPU_SIM currently does not support the GlobalData overload.

## Examples

### C2V Vector Pop

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_c2v(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;
    constexpr uint32_t LocalBase = 0x0;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(T), FifoDepth>;
    using VecTile = Tile<TileType::Vec, T, M / 2, N, BLayout::RowMajor, M / 2, N>;

    Pipe pipe(fifoMem, LocalBase, 0x0); // C2V mode: the second parameter is the C2V consumer buffer base address, the third parameter is the V2C consumer buffer base address (0 here).
    VecTile tile;

    TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, tile);
}
```

### V2C Matrix Pop

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_v2c(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;
    constexpr uint32_t LocalBase = 0x0;

    using Pipe = TPipe<FlagID, Direction::DIR_V2C, M * N * sizeof(T), FifoDepth>;
    using MatTile = Tile<TileType::Mat, T, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;

    Pipe pipe(fifoMem, 0x0, LocalBase);
    MatTile tile;

    TPOP<Pipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}
```

### GlobalData Slot Pop

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
    using VecTile = Tile<TileType::Vec, T, M / 2, N, BLayout::RowMajor, M / 2, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    SlotGlobal slot;
    VecTile tile;
    TASSIGN(tile, 0x0);

    TPOP<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
    TLOAD(tile, slot);
    TFREE<Pipe, SlotGlobal, TileSplitAxis::TILE_UP_DOWN>(pipe, slot);
}
```
