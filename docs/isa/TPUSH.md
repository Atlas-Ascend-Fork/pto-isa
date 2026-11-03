# TPUSH

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T05:34:32.560Z pushedAt=2026-11-02T03:00:52.214Z nl=2 -->

## Introduction

Pushes a producer tile into the FIFO for data transfer between Cube and Vector and for inter-core synchronization.

This instruction supports the **TileData** overload that explicitly specifies `TileSplitAxis`, the **TileData** overload that explicitly specifies the subblock ID, the **TileData** overload with parameter inversion and no `Split` specified, as well as the `GlobalData` and `TConfig` overloads.

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

For the `TConfig` overload `TPUSH<Pipe, TileProd, TConfig>(pipe, tile)`, the `TConfig` template parameter is used to configure the FixPipe conversion and layout. The NPU backend applies it to the L0C->GM/UB path, while CPU_SIM writes the conversion result to the host FIFO slot.

## C++ Intrinsic

Declaration location: `include/pto/common/pto_instr.hpp`:

```cpp
template <typename Pipe, typename TileProd, TileSplitAxis Split,
          std::enable_if_t<is_tile_data_v<TileProd>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &... events);

template <typename Pipe, typename TileProd, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, int32_t subBlockId,
                           WaitEvents &... events);

template <typename TileData, typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(TileData &tile, Pipe &pipe, WaitEvents &... events);

template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, GlobalData &gmTensor, WaitEvents &... events);

template <typename Pipe, typename TileProd, typename TConfig, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe &pipe, TileProd &tile, WaitEvents &... events);
```

`Pipe` is usually the `TPipe` type declared in `TPush.hpp`:

```cpp
template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t LocalSlotNum = 2, bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe;
```

## Constraints

- **TileData type producer**:
    - `TileProd::Loc` must be `TileType::Acc`, `TileType::Vec`, or `TileType::Ctrl`.
    - `Direction::DIR_C2V`: Cube produces an accumulator tile for vector consumption.
    - `Direction::DIR_V2C`: Vector produces a vector tile for cube consumption.
    - `Direction::DIR_BOTH`: The same pipe type supports both C2V and V2C producers.
- **FIFO slots**:
    - `SlotSize` must be large enough to hold one logical FIFO entry.
    - `SlotNum >= 1`.
- **Ascend 950PR/Ascend 950DT synchronization flags**:
    - `FlagID + 1` must not exceed `MAX_SYC_ID`.
    - `DIR_BOTH` and `DIR_BOTH_GM` use `FlagID + 3`; when both directions are used simultaneously, `FlagID + 3` must not exceed `MAX_SYC_ID`.
- **Split behavior for Atlas A2/A3 training products/Atlas A2/A3 inference products**:
    - `TileSplitAxis::TILE_NO_SPLIT`: No split is performed. To enable this split mode on Atlas A2/A3 training products/Atlas A2/A3 inference products, AIV0 and AIV1 must perform accompanying synchronization operations.
    - `TileSplitAxis::TILE_UP_DOWN`: The vector sub-block is mapped to the upper and lower row halves.
    - `TileSplitAxis::TILE_LEFT_RIGHT`: The vector sub-block is mapped to the left and right column halves.
- **Ascend 950PR/Ascend 950DT split behavior**:
    - `TileSplitAxis::TILE_NO_SPLIT`: No split is performed.
    - `TileSplitAxis::TILE_UP_DOWN`: Splits the data into upper and lower halves. In the Cube->Vector direction over the L0C->UB path, this split mode supports only the b32 data type, and the validRows of srcTile must be an integer multiple of 2. In the Vector->Cube direction over the UB->L1 path, the vector subblocks are inserted into the upper and lower row regions respectively.
    - `TileSplitAxis::TILE_LEFT_RIGHT`: Splits the data into left and right column halves. In the Cube->Vector direction over the L0C->UB path, this split mode supports only the b32 data type, and the validCols of srcTile must be an integer multiple of 32. In the Vector->Cube direction over the UB->L1 path, validCols must be an integer multiple of 32 bytes in this split mode.
- **Simplified TileData API**:
    - `TPUSH(TileData&, Pipe&)` internally uses `TileSplitAxis::TILE_NO_SPLIT` semantics.
    - `TileData::Loc` must be `TileType::Acc` or `TileType::Vec`.
- **TConfig API**:
    - `TConfig` is the configuration type that determines the push behavior (implementation-defined).
    - `TileProd::Loc` must be `TileType::Acc`, `TileType::Vec`, or `TileType::Ctrl`.
- **NPU synchronization**:
    - Free-space waiting is sparse and controlled by `Pipe::SyncPeriod`.
    - Each `TPUSH` issues a data-ready record.
- **GlobalData type producer**:
    - `gmTensor` must be a FIFO slot view returned by `TALLOC`.
    - Before calling `TPUSH(Pipe&, GlobalData&)`, data must have been written to `gmTensor`.
    - `TPUSH(Pipe&, GlobalData&)` ignores the tensor content and only submits the FIFO slot to the consumer.
- **CPU_SIM FIFO model**:
    - The FIFO state is shared by host threads. `TPUSH` waits for a free slot and submits data through a mutex and a condition variable.
    - TileData data uses the storage owned by the host FIFO state. Even if a non-empty NPU GM workspace is passed in when constructing `TPipe`, the CPU_SIM TileData flow does not access that workspace, so that the data lifetime and the slot state are protected by the same set of synchronization.
    - The TileData producer supports `DIR_C2V`, `DIR_V2C`, and `DIR_BOTH`. A `DIR_BOTH` pipe maintains independent ring queues for C2V and V2C, each direction having `SlotNum` slots, along with independent data storage, cursors, and synchronization state. A full direction does not consume the capacity of the other direction; submission sequence numbers are used to preserve FIFO order within each direction.
    - CPU_SIM waits for the specific slot selected by the corresponding direction's producer cursor to become free, and does not simulate the NPU's `SyncPeriod` credit pacing.
    - The split mode selects a lane based on the current subblock context. `TILE_NO_SPLIT` uses one producer lane. For a C2V pipe with `IsNoSplit` enabled, one producer slot can coordinate one or two vector consumer subblocks depending on the runtime subblock count.
    - CPU_SIM currently does not implement the overload that explicitly passes `int32_t subBlockId`; the split lane should be selected through the simulated subblock execution context.
    - The simplified interface uses `TILE_NO_SPLIT`; the `TConfig` interface also supports the CPU fixpipe path.
    - Under `TILE_NO_SPLIT`, the slot payload is laid out according to the shape of the actual transfer window (that is, the valid shape of the pushed tile), so when pushing a narrow view of a wide tile, the rows remain aligned with the tile popped by the consumer; the split axis lays out the payload according to the declared shape of the producer tile.
    - CPU_SIM currently does not support the GlobalData overload.
- **Tile type support**:
    - **Tile types supported by TPUSH/TPOP**:
        - `TileType::Acc` (accumulator tile): used by the Cube core for C2V direction communication.
        - `TileType::Vec` (vector tile): used by the Vector core for V2C direction communication.
        - `TileType::Ctrl` (control tile): used by the Vector core for control signal communication in the V2C_CTRL direction.

## Defining TConfig

The `TConfig` template parameter in the `TPUSH<Pipe, TileProd, TConfig>(pipe, tile)` overload is a configuration structure used to control the fixpipe behavior during the push process. PTO provides the `FixpipeParams` structure to implement this functionality.

Declared in `include/pto/common/fixpipe.hpp`:

```cpp
template <LayoutMode_t layoutMode = LayoutMode_t::NZ2ND,
          QuantMode_t quantMode = QuantMode_t::NoQuant,
          ReluPreMode reluMode = ReluPreMode::NoRelu,
          STPhase phase = STPhase::Unspecified,
          uint8_t subBlockId = 0,
          AtomicType atomicT = AtomicType::AtomicNone,
          ClipReluMode_t clipReluMode = ClipReluMode_t::NOCLIP_RELU,
          bool isChannelSplit = false>
struct FixpipeParams {
    static constexpr LayoutMode_t LayoutMode = layoutMode;
    static constexpr QuantMode_t QuantPre = quantMode;
    static constexpr ReluPreMode ReluMode = reluMode;
    static constexpr STPhase Phase = phase;
    static constexpr uint8_t SubBlockId = subBlockId;
    static constexpr AtomicType AtomicT = atomicT;
    static constexpr ClipReluMode_t ClipReluMode = clipReluMode;
    static constexpr bool IsChannelSplit = isChannelSplit;
};
```

### TConfig Field Description

| Field | Type | Description |
|------|------|------|
| `LayoutMode` | `LayoutMode_t` | Output data layout: `NZ2NZ` (NZ→NZ), `NZ2ND` (NZ→row-major), `NZ2DN` (NZ→column-major). Default value: `NZ2ND`. |
| `QuantPre` | `QuantMode_t` | Quantization/dequantization mode (defined by CANN). Controls the data type conversion during the fixpipe push process. Default value: `NoQuant`. |
| `ReluMode` | `ReluPreMode` | ReLU activation mode: `NoRelu` or `NormalRelu`. Default value: `NoRelu`. |
| `Phase` | `STPhase` | Storage phase (for the unit-flag path): `Unspecified`, `Partial`, or `Final`. Default value: `Unspecified`. |
| `SubBlockId` | `uint8_t` | Subblock identifier, used for accumulator-to-vector transfer mode mapping (Ascend 950PR/Ascend 950DT only). Default value: `0`. |
| `AtomicT` | `AtomicType` | Atomic operation type for GM writes: `AtomicNone` or `AtomicAdd`. Default value: `AtomicNone`. |
| `ClipReluMode` | `ClipReluMode_t` | Clip ReLU mode: `NOCLIP_RELU` or `CLIP_RELU`. Default value: `NOCLIP_RELU`. |
| `IsChannelSplit` | `bool` | Whether channel split is enabled. Default value: `false`. |

### TConfig Usage Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T>
AICORE void example_tconfig_push(__gm__ void *fifoMem)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t FlagID = 0;
    constexpr uint32_t FifoDepth = 2;

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(T), FifoDepth>;
    using AccTile = TileAcc<float, M, N, M, N>;

    // Define TConfig: NZ→row-major layout, dequantize to half, enable ReLU.
    using MyConfig = FixpipeParams<LayoutMode_t::NZ2ND, QuantMode_t::DEQF16, ReluPreMode::NormalRelu>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    AccTile acc;
    TASSIGN(acc, 0x0);

    TPUSH<Pipe, AccTile, MyConfig>(pipe, acc);
}
```

## Examples

### C2V Accumulator Push

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

    using Pipe = TPipe<FlagID, Direction::DIR_C2V, M * N * sizeof(float), FifoDepth>;
    using AccTile = TileAcc<float, M, N, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    AccTile acc;
    TASSIGN(acc, 0x0);

    // Fill acc with a cube computation before pushing.
    TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, acc);
}
```

### V2C Vector Push

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

    using Pipe = TPipe<FlagID, Direction::DIR_V2C, M * N * sizeof(T), FifoDepth>;
    using VecTile = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    VecTile tile;
    TASSIGN(tile, 0x0);

    TPUSH<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}
```

### GlobalData Push Example

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
    using SlotGlobal = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using VecTile = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;

    Pipe pipe(fifoMem, 0x0, 0x0);
    SlotGlobal slot;
    VecTile tile;
    TASSIGN(tile, 0x0);

    TALLOC<Pipe, SlotGlobal, TileSplitAxis::TILE_NO_SPLIT>(pipe, slot);
    TSTORE(slot, tile);
    TPUSH<Pipe, SlotGlobal, TileSplitAxis::TILE_NO_SPLIT>(pipe, slot);
}
```
