# TCOLEXPANDEXPDIF

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T04:44:34.854Z pushedAt=2026-11-02T03:00:52.166Z nl=2 -->

## Instruction Diagram

![TCOLEXPANDEXPDIF tile operation](../figures/isa/TCOLEXPANDEXPDIF.svg)

## Introduction

Column exponential difference operation: computes exp(src0 - src1), where src1 is a per-column scalar.

## Mathematical Semantics

Assume `R = dst.GetValidRow()` and `C = dst.GetValidCol()`. Assume that `s_j` is the per-column scalar obtained from `src1` (one value per column).

For `0 <= i < R` and `0 <= j < C`:

$$
\mathrm{dst}_{i,j} = \exp(\mathrm{src0}_{i,j} - s_j)
$$

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDEXPDIF(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

- `TileDataDst::DType`, `TileDataSrc0::DType`, and `TileDataSrc1::DType` must be one of the following: `half`, `float`.
- Tile shape/layout constraint (compile time): `TileDataDst::isRowMajor`.
- `src1` is expected to provide **one scalar per column** (that is, its effective shape must cover `C` values).
- The exact layout/fractal constraints are destination-specific; see the backend header files under `include/pto/npu/*/TColExpand*.hpp`.

## Examples

See the relevant examples in `docs/isa/` and `docs/coding/tutorials/`.
