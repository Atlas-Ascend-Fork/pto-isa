# TCOLEXPANDADD

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:41:14.907Z pushedAt=2026-11-02T03:00:52.164Z nl=2 -->

## Instruction Diagram

![TCOLEXPANDADD tile operation](../figures/isa/TCOLEXPANDADD.svg)

## Introduction

Column broadcast addition: broadcasts the scalar value of each column to all elements in that column and performs addition.

## Mathematical Semantics

Assume `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Assume that `s_j` is the per-column scalar obtained from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} + s_j
$$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDADD(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

- `TileDataDst::DType`, `TileDataSrc0::DType`, and `TileDataSrc1::DType` must be one of the following: `half`, `float`, `int16`, `int32` (for Atlas A2 training products/Atlas A2 inference products, Atlas A3 training products/Atlas A3 inference products, and Ascend 950PR/Ascend 950DT), `uint16`, `uint32`, `bfloat16_t`, `int8`, `uint8`, `int64`, `uint64` (for Ascend 950PR/Ascend 950DT).
- Tile shape/layout constraints (compile time): `TileDataDst::isRowMajor` must be `true`.
- `src1` is expected to provide **one scalar per column** (that is, its effective shape must cover `C` values).
- The exact layout/fractal constraints are destination-specific; see the backend header files under `include/pto/npu/*/TColExpand*.hpp`.
### 64-Bit Element Types (Ascend 950PR/Ascend 950DT)

`int64` / `uint64` are supported only on Ascend 950PR/Ascend 950DT. This architecture has no native 64-bit vector arithmetic unit, so the instruction is emulated through a pair of 32-bit registers (holding the low 32 bits and high 32 bits of each element, respectively); each column scalar operand is read using the same deinterleaved layout as the full-size operand.

The computed result is an exact 64-bit two's complement value. Tile alignment follows the general rules for 64-bit elements: Row-Major Tiles require `Cols % 4 == 0`.

## Examples

See the relevant examples in `docs/isa/` and `docs/coding/tutorials/`.
