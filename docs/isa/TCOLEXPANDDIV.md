# TCOLEXPANDDIV

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:43:40.649Z pushedAt=2026-11-02T03:00:52.165Z nl=2 -->

## Instruction Diagram

![TCOLEXPANDDIV tile operation](../figures/isa/TCOLEXPANDDIV.svg)

## Introduction

Column broadcast division: divides each column by a per-column scalar vector.

## Mathematical Semantics

Assume `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Assume that `s_j` is the per-column scalar obtained from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{i,j} = \mathrm{src0}_{i,j} / s_j
$$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDDIV(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

- `TileDataDst::DType` and `TileDataSrc1::DType` must be one of the following: `half`, `float` for Atlas A2 training products/Atlas A2 inference products, Atlas A3 training products/Atlas A3 inference products, and Ascend 950PR/Ascend 950DT; `int16`, `int32`, `uint16`, `uint32`, `bfloat16_t`, `int8`, `uint8`, `int64`, `uint64` for Ascend 950PR/Ascend 950DT.
- Tile shape/layout constraint (compile time): `TileDataDst::isRowMajor`.
- `src1` is expected to provide **one scalar per column** (that is, its effective shape must cover `C` values).
- The exact layout/fractal constraints are destination-specific; see the backend header files under `include/pto/npu/*/TColExpand*.hpp`.
### 64-Bit Element Types (Ascend 950PR/Ascend 950DT)

`int64` / `uint64` are supported only on Ascend 950PR/Ascend 950DT. This architecture has no native 64-bit vector arithmetic unit, and the instruction emulates the operation through a pair of 32-bit registers (holding the low 32 bits and high 32 bits of each element, respectively); the per-column scalar operand is read using the same deinterleaved layout as the full-size operand.

Division rounds toward zero, consistent with the behavior of 32-bit integers; division-by-zero behavior is target-defined. Tile alignment follows the general rules for 64-bit elements: Row-Major tiles require `Cols % 4 == 0`.

## Examples

See the relevant examples in `docs/isa/` and `docs/coding/tutorials/`.
