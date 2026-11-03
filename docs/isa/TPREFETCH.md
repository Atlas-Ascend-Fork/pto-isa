# TPREFETCH

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-10-30T05:32:43.720Z pushedAt=2026-11-02T03:00:52.211Z nl=2 -->

## Instruction Diagram

![TPREFETCH tile operation](../figures/isa/TPREFETCH.svg)

## Introduction

Prefetches data from global memory into the tile local cache/buffer (implementation-defined). This is typically used to reduce latency before a subsequent `TLOAD`.

Note: Unlike most PTO instructions, `TPREFETCH` does **not** implicitly call event waits in the C++ wrapper.

## Mathematical Semantics

Unless otherwise specified, the semantics are defined over the valid region, and target-dependent behavior is marked as implementation-defined.

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData &dst, GlobalData &src);
```

## Constraints

- The semantics and cache behavior are target/implementation-defined.
- Some targets may ignore the prefetch and treat it as a hint.

## Examples

See the relevant examples in `docs/isa/` and `docs/coding/tutorials/`.
