# TIMG2COL

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T05:07:23.758Z pushedAt=2026-11-02T03:00:52.190Z nl=2 -->

## Instruction Diagram

![TIMG2COL tile operation](../figures/isa/TIMG2COL.svg)

## Introduction

Performs an image-to-column transformation for convolution-like workloads.

## Mathematical Semantics

Unless otherwise specified, the semantics are defined over the valid region, and target-dependent behavior is marked as implementation-defined.

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent TIMG2COL(TileData &dst, ConvTileData &src, uint16_t posM = 0, uint16_t posK = 0, WaitEvents &... events);
```

## Constraints

- This instruction is target/implementation-specific. For supported tile types/layouts and configuration fields, see `include/pto/npu/*/TImg2col.hpp`.
- **Implementation check (Atlas A2/A3 training products/Atlas A2/A3 inference products)**: `TileData::DType` must be one of the following: `int8_t`, `half`, `bfloat16_t`, `float`.
- **Implementation check (Ascend 950PR/Ascend 950DT)**: `TileData::DType` must be one of the following: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`.

## Examples

See the relevant examples in `docs/isa/` and `docs/coding/tutorials/`.
