# Spec — `std.dense` (typed dim-tracked dense tensors / embeddings, with honest per-op bounds)

| Field | Value |
|---|---|
| **Status** | **Accepted** (2026-06-20, maintainer-ratified per DN-07 — guarantee matrix asserted in tests; open §7/§8 questions are design/scope calls, not contract violations; was *Implemented (Rust-first) — pending ratification* 2026-06-18, Draft/needs-design 2026-06-17) — the Rust-first code landed as `mycelium-std-dense` (M-518, Batch P5-A; guarantee matrix asserted in tests). The Mycelium-lang migration (M-502-gated) remains. |
| **Module / Ring** | `std.dense` · Ring `1` (RFC-0016 §4.2) · Tier `A` |
| **Tracks** | `M-518` (#160) — the Phase-5 task this spec delivers |
| **Scope** | The ergonomic surface over typed, dimension-tracked dense values: `Dense{dim, dtype}` tensors / embeddings, their constructors, elementwise arithmetic, reductions (norms), and similarity ops (dot / cosine). The exported promise is that **`dim` and `dtype` live in the type**, so a shape or precision mismatch is caught at the boundary as a typed error, and every float op carries its ε-tag honestly. |
| **Boundary** | A **representation change** — `Dense{F32} → Dense{BF16}`, or `Dense ↔ VSA` — is **not** here: it is `std.swap` (M-516, RFC-0002 §5), which issues a certificate. A non-representation numeric **cast** (e.g. `F64 → F32` outside a swap regime) is `std.cmp`/`convert` (M-532). The raw ε/δ **bound kernels + shared certificate** are `std.numerics` (M-512, ADR-010); this module *consumes* them, it does not define them. VSA hypervector algebra (bind/bundle/permute) is `std.vsa` (M-513). |
| **Depends on** | RFC-0001 §4.1 (the `Dense{dim, dtype}` descriptor + value model); RFC-0002 §5 (the per-op Exact/Bounded regime vocabulary); RFC-0016 §4.1 (the C1–C6 contract), §4.3 (the `dense` row), §4.5 (the guarantee-matrix obligation) |
| **Grounds on** | the landed typed dim-tracked dense capability crate (M-230, verified `docs/verification/M-230-verified.md`); the ADR-010 verified-numerics `ErrorBound` kernel + shared certificate, surfaced via `std.numerics` (M-512). KC-3: above the kernel — adds no trusted code. |

---

## 1. Summary

`std.dense` is the ergonomic home for typed, dimension-tracked dense values — embeddings and dense tensors typed `Dense{dim, dtype}` (RFC-0001 §4.1), with **`dim` and `dtype` in the type**. It exports constructors, elementwise arithmetic, reductions (norms), and similarity ops (dot / cosine). Its **honesty crux** is two-fold and structural: (1) a **dim or dtype mismatch — or an off-grid / wrong-shape payload — is a *typed error at the boundary*, never a silent broadcast, narrowing, or re-round** (C1/G2); and (2) every **float op carries its ε-bound from the ADR-010 `ErrorBound` kernel**, tagged `Proven` *only* where the rounding-error theorem's side-conditions are checked, and **honestly downgraded** otherwise (C2/VR-5). It is a Ring-1 capability surface (RFC-0016 §4.2): it *consumes* the M-230 dense crate and the ADR-010 numerics certificate and **adds no trusted code** (KC-3).

## 2. Scope & module boundary

- **In scope:** the `Dense{dim, dtype}` value type and its surface — typed constructors (from a literal scalar array, zeros/full, and the *fallible* `from_slice` that checks length against `dim` and grid against `dtype`); elementwise ops (`add`/`sub`/`mul`/`hadamard`, scalar broadcast, `map`); reductions (`sum`, `l2_norm`, `l1_norm`, `max`/`min`); and similarity ops (`dot`, `cosine`). Each op's shape/precision pre-conditions are enforced from the type, and each float op's ε-tag is recorded in `Meta.guarantee` + `Meta.bound` (RFC-0001 §4.3).
- **Out of scope (and who owns it):**
  - representation change (`Dense{F32}→Dense{BF16}`, `Dense↔VSA`) → **`std.swap`** (M-516; RFC-0002 §5 legal-pair table) — it issues a swap certificate; `std.dense` never re-encodes silently.
  - the ε/δ bound algebra + certificate itself → **`std.numerics`** (M-512; ADR-010 `ErrorBound`/shared certificate); `std.dense` is a *consumer*.
  - non-representation scalar conversion → **`std.cmp`/`convert`** (M-532); a lossy cast there is explicit + fallible.
  - VSA hypervector algebra → **`std.vsa`** (M-513).
- **Ring & layering:** Ring 1, Tier A (RFC-0016 §4.2/§4.3). It **wraps** the landed M-230 dense crate (the typed dim-tracked dense value + its ops) and **consumes** the ADR-010 numerics certificate via `std.numerics`; it defines no new representation kind (the four kinds are closed in the kernel, RFC-0001 §4.1) and **enlarges no trusted base** (KC-3). It is a certificate/EXPLAIN consumer, not a producer of trusted bounds.

## 3. Exported-op surface (design sketch)

The surface is value-semantic and immutable-by-default. `dim` and `dtype` are type parameters, so an op over two operands of *different* `dim`/`dtype` does not type-check (the compile-time floor); the *runtime* fallible constructors (`from_slice`) return `Result` for inputs whose shape/grid is only known dynamically. Float ops return the value **and** its recorded bound.

```
// illustrative signatures (not a committed surface)

type Dense<const DIM: Nat, DT: ScalarKind>            // = Value<Dense{dim: DIM, dtype: DT}> (RFC-0001 §4.1)

// --- constructors ---
fn zeros<DIM, DT>() -> Dense<DIM, DT>                 // total; Exact
fn full<DIM, DT>(x: Scalar<DT>) -> Dense<DIM, DT>     // total; Exact
fn from_slice<DIM, DT>(xs: &[Scalar]) -> Result<Dense<DIM, DT>, DenseError>
    // Err if xs.len() != DIM (LenMismatch) or any x off the DT grid (OffGrid) — never truncated/re-rounded

// --- elementwise (shape & dtype from the type; no implicit broadcast across dim) ---
fn add<DIM, DT>(a: Dense<DIM, DT>, b: Dense<DIM, DT>) -> Dense<DIM, DT>     // Exact for integer DT; Bounded(ε) for float DT
fn hadamard<DIM, DT>(a: Dense<DIM, DT>, b: Dense<DIM, DT>) -> Dense<DIM, DT>
fn scale<DIM, DT>(a: Dense<DIM, DT>, s: Scalar<DT>) -> Dense<DIM, DT>
fn map<DIM, DT>(a: Dense<DIM, DT>, f: Fn(Scalar<DT>) -> Scalar<DT>) -> Dense<DIM, DT>  // tag = f's tag (no upgrade)

// --- reductions ---
fn sum<DIM, DT>(a: Dense<DIM, DT>) -> Scalar<DT>                 // Exact (int) | Bounded(ε, ∝ DIM) (float)
fn l2_norm<DIM, DT: Float>(a: Dense<DIM, DT>) -> Scalar<DT>      // Bounded(ε) — sum-of-squares + sqrt
fn l1_norm<DIM, DT>(a: Dense<DIM, DT>) -> Scalar<DT>

// --- similarity ---
fn dot<DIM, DT>(a: Dense<DIM, DT>, b: Dense<DIM, DT>) -> Scalar<DT>          // Exact (int) | Bounded(ε) (float)
fn cosine<DIM, DT: Float>(a: Dense<DIM, DT>, b: Dense<DIM, DT>)
    -> Result<Scalar<DT>, DenseError>   // Bounded(ε); Err if a or b is the zero vector (ZeroNorm) — never NaN/sentinel

enum DenseError { LenMismatch{ expected: Nat, got: Nat }, OffGrid{ dtype: ScalarKind }, ZeroNorm }
```

Two operands of mismatched `dim` or `dtype` are a **static type error** (the type carries the shape) — there is no runtime "mismatch" branch on the typed elementwise/similarity ops, because such a call never elaborates. `DenseError` covers only the **dynamic-input boundary** (`from_slice`) and the **mathematically partial** op (`cosine` on a zero vector). No op silently broadcasts, narrows, or re-rounds.

## 4. Guarantee matrix (the load-bearing deliverable — RFC-0016 §4.5)

Rows = exported ops. Encoded as a checked table (the RFC-0003 §4 template), asserted in tests once code lands — never prose only. "Tag" is the RFC-0001 lattice strength (`Exact ⊐ Proven ⊐ Empirical ⊐ Declared`); float ops carry their ε via the ADR-010 `ErrorBound` kernel (RFC-0001 §4.3 `Bound`). `DT` = `dtype`; **int** = an integer `ScalarKind`, **float** = `F16|BF16|F32|F64`.

| Op | Guarantee tag | Fallibility (explicit error set) | Declared effects | EXPLAIN-able? |
|---|---|---|---|---|
| `zeros` / `full` | `Exact` | total | `none` | n/a |
| `from_slice` | `Exact` (no arithmetic) | `Err DenseError::{LenMismatch, OffGrid}` | `none` | yes (which check failed) |
| `add` / `sub` (int DT) | `Exact` | total (dim/dtype = static type error) | `none` | n/a |
| `add` / `sub` (float DT) | `Proven` (ε) | total (dim/dtype = static type error) | `none` | yes (ε-bound artifact) |
| `hadamard` / `scale` (int DT) | `Exact` | total | `none` | n/a |
| `hadamard` / `scale` (float DT) | `Proven` (ε) | total | `none` | yes (ε-bound artifact) |
| `map` | tag of `f` (meet; never upgraded) | `f`'s error set | `f`'s effects | iff `f` is |
| `sum` (int DT) | `Exact` | total | `none` | n/a |
| `sum` (float DT) | `Empirical` (ε ≈ 2·n·u; `nγ_n` accumulation bound not yet checked) | total | `none` | yes (ε-bound artifact) |
| `l1_norm` (int DT) | `Exact` | total | `none` | n/a |
| `l2_norm` (float DT) | `Empirical` (ε; sqrt side-condition) **[FLAG Q2]** | total | `none` | yes (ε-bound artifact) |
| `dot` (int DT) | `Exact` | total | `none` | n/a |
| `dot` (float DT) | `Empirical` (ε ≈ 2·n·u; `nγ_n` accumulation bound not yet checked) | total | `none` | yes (ε-bound artifact) |
| `cosine` (float DT) | `Empirical` (ε; division + sqrt) **[FLAG Q2]** | `Err DenseError::ZeroNorm` | `none` | yes (ε-bound artifact) |

**Justification of every non-`Exact` tag (VR-5 — downgrade rather than overclaim):**

- **Exact-integer ops.** Integer `add`/`sub`/`hadamard`/`scale`/`sum`/`dot`/`l1_norm` are exact (no rounding); they carry **no** `Bound` (RFC-0001 M-I1) and tag `Exact`. An integer op that *would* overflow `DT` is an explicit error in the kernel arithmetic it wraps, not a wrap-around — but the overflow-policy surface is owned by `std.numerics`/the kernel, FLAGGED below as **Q3**.
- **Float elementwise `add`/`sub`/`hadamard`/`scale` — `Proven` (Q1 finalized, 2026-06-19).** Each carries its per-element round-off via the **ADR-010 `ErrorBound` (ε) kernel** (affine arithmetic; `Bound = ErrorBound{eps, norm}`, basis `ProvenThm{citation: …}`, RFC-0001 §4.3). The instantiated theorem is the per-element IEEE backward-error bound `fl(a∘b) = (a∘b)(1+δ)`, `|δ| ≤ u` (Higham 2002, Thm 2.2); its single side-condition (operand finiteness / no overflow) is **guarded at runtime** (non-finite → the kernel `DenseError::NonFinite`, surfaced as `StdDenseError::Kernel` at the `std.dense` boundary (the illustrative §3 enum elides it — Q4)) and re-validated by the numerics checker (Tier-i certificate re-validation, M-512 delivered). With the checked instantiation in place, the `Proven` tag is finalized (no longer provisional).
- **Float accumulation `sum`/`dot`/`l1_norm` — `Empirical`.** These compose `n` round-offs; the governing bound is the `nγ_n`-style Higham summation/inner-product bound, a **distinct** theorem whose checked instantiation is **not** yet delivered. They are tagged **`Empirical`** (basis `EmpiricalFit`, a conservative `2·n·u` outward-rounded upper bound, RFC-0001 M-I3) — honestly downgraded, never `Proven` on faith. Upgrade only when a checked `nγ_n` accumulation theorem lands (VR-5).
- **`l2_norm` / `cosine`.** These add a **square root** (and `cosine` a **division**) on top of a bounded sum-of-squares / dot. The composed bound's side-conditions (the non-negativity / non-zero-denominator pre-conditions of the sqrt and division error terms) are **not** all checked in this design, so they are honestly tagged **`Empirical`** (basis `EmpiricalFit`, RFC-0001 M-I3) pending a checked composition — FLAGGED **Q2**. `cosine` is additionally *partial*: a zero-norm operand is `Err ZeroNorm`, never `NaN` or a sentinel (C1).
- **`map`.** The tag is the **meet** of the input value's tag and the mapped function's tag — `map` never *upgrades* a value's strength (VR-5). If `f` is `Exact` and the input is `Exact`, the result is `Exact`; otherwise it is the weaker of the two.
- **Strength composes by meet (weakest wins)** across every op, exactly as ADR-010 §Decision-3 and the RFC-0001 §4.7 lattice require: a result is no stronger than its weakest input or its own op tag.

## 5. §4.1 contract conformance (C1–C6)

- **C1 — never-silent (G2):** a dim/dtype mismatch on an elementwise/similarity op is a **static type error** (the shape is in the type, RFC-0001 §4.1) — the call never elaborates, so it cannot silently broadcast or narrow. The *dynamic* boundary (`from_slice`) returns `Result<_, DenseError>` for a length mismatch (`LenMismatch`) or off-grid payload (`OffGrid`) — **never truncated, padded, or re-rounded**. `cosine` on a zero vector is `Err ZeroNorm`, never `NaN`/sentinel. This is the RFC-0002 §5 posture ("pair with no statable bound → type error, not a `Declared` gamble") applied at the op boundary.
- **C2 — honest per-op tag (VR-5):** every row carries an explicit lattice tag (§4). Integer ops are `Exact`; float ops carry their ε from the ADR-010 `ErrorBound` kernel and are `Proven` **only** where the bound's side-conditions are checked, else **downgraded** to `Empirical` (`l2_norm`/`cosine`, and any float op whose Higham-instantiation the checker cannot discharge). No tag is upgraded without a checked basis (RFC-0001 M-I2/M-I3).
- **C3 — no black boxes / EXPLAIN (SC-3/G11):** every float op's ε-bound is the **reified ADR-010 certificate** (`{ε, δ, strength}` + `BoundBasis`), inspectable and surfaced through `std.numerics`; `EXPLAIN` on a float result yields *which* bound theorem (or empirical fit) produced its tag, and `EXPLAIN` on a `from_slice`/`cosine` failure names *which* check failed. Approximation is never hidden.
- **C4 — content-addressed, value-semantic (ADR-003 / RFC-0001):** `Dense{dim, dtype}` values are immutable; every op is a pure function of its inputs (plus, for float ops, the declared rounding mode the ε-bound assumes). Identity is content-addressed over the payload + type (RFC-0001 §4.6); `Meta` (the bound, provenance) is **not** identity.
- **C5 — above the small kernel (KC-3):** the module **consumes** the landed M-230 dense crate and the ADR-010 numerics certificate; it defines no new representation kind (the four kinds are closed, RFC-0001 §4.1) and adds **no** trusted code. No `wild`/FFI (ADR-014) — pure over the value model.
- **C6 — declared, bounded effects (RFC-0014):** every op here is **pure** (`effects: none`) except `map`, whose effects are exactly its function argument's declared effects (it adds none of its own). No op performs IO, time, or randomness; there is no undeclared side effect.

## 6. Grounding

- **The typed `Dense{dim, dtype}` model with dim-in-the-type** — RFC-0001 §4.1 (the `Repr` descriptor: `Dense "{" "dim:" Nat "," "dtype:" ScalarKind "}"`; `dtype` stays in the type because "precision is semantically significant — it bounds embedding error") and §4.2 (the `Value<R>` value model). The landed crate is **M-230** (`docs/verification/M-230-verified.md`: "Typed DenseSpace with dim-in-type; honest Exact/Proven bounds on ops; explicit typed errors for all mismatches"), which grounds on RFC-0001 §4.1 + RFC-0002 §5.
- **The per-op Exact/Bounded regime + "mismatch is a typed error"** — RFC-0002 §5 legal-pair table (`Dense F32→BF16` is `Bounded (ε)` via the ADR-010 `ErrorBound`; "pair with no statable bound → type error, not a `Declared` gamble"). RFC-0016 §4.3 names the `dense` module as "elementwise + similarity ops with per-op tags" grounded in RFC-0001 §4.1 + M-230.
- **The ε-bound machinery + the lattice** — ADR-010 (the `ErrorBound` (ε) kernel = affine arithmetic; the shared `{ε, δ, strength}` certificate; **strength composes by meet**; the Rust certificate-checker trusted base, Tier-i default / Tier-ii verified-binary). RFC-0001 §4.3 (`GuaranteeStrength`, `Bound`/`BoundKind`/`BoundBasis`, invariants M-I1…M-I4) and §4.7 (lattice + bound composition).
- **The contract + matrix obligation** — RFC-0016 §4.1 (C1–C6), §4.2 (ring layering — Ring 1 capability surfaces are certificate/EXPLAIN consumers), §4.5 (every module ships a checked guarantee matrix); `docs/spec/stdlib/README.md` §1–§4.1.
- **House requirements** — VR-5 (honest tags, downgrade not upgrade), G2 (never-silent), SC-3/G11 (EXPLAIN), KC-3 (small kernel), ADR-003 (content-addressed value semantics), RFC-0014 (declared bounded effects).

## 7. Open questions (FLAGGED — resolve before ratification)

- **(Q1 — RESOLVED 2026-06-19; DN-16) Float `Proven` disposition, split by op class.** With M-512 delivered (the ADR-010 checked numerics instantiation), Q1 resolves into two honest verdicts, now reflected in §4 and in the landed crate: **(a) elementwise** `add`/`sub`/`hadamard`/`scale` are **finalized `Proven`** — the per-element IEEE backward-error bound (`fl(a∘b)=(a∘b)(1+δ)`, `|δ|≤u`; Higham 2002 Thm 2.2) with its finiteness side-condition guarded at runtime (the kernel `DenseError::NonFinite`, surfaced as `StdDenseError::Kernel` at the `std.dense` boundary (the illustrative §3 enum elides it — Q4)) and re-validated by the numerics checker; **(b) accumulation** `sum`/`dot`/`l1_norm` are **`Empirical`** — the `nγ_n` summation/inner-product bound is a *distinct* theorem whose checked instantiation is not yet delivered, so they carry a conservative `2·n·u` `EmpiricalFit` and are not `Proven` on faith (VR-5). The exact constants/side-conditions remain owned by ADR-010 / `std.numerics` (cited, not restated). Remaining open: a checked `nγ_n` accumulation theorem would let accumulation upgrade to `Proven` — a future numerics task. (Ties to RFC-0016 §8-Q1, the v0 surface.)
- **(Q2) Are `l2_norm` / `cosine` provable, or do they stay `Empirical`?** Both compose a bounded dot/sum-of-squares with a **sqrt** (and `cosine` a **division**). Whether the composed bound's side-conditions (non-negativity for sqrt, non-zero denominator for division — already guarded by `ZeroNorm`) are *checkable* through the ADR-010 affine/error kernel, and thus earn `Proven`, is open. This design tags them **`Empirical`** to stay honest; an upgrade requires a checked composition from M-512. Disposition: decide with M-512.
- **(Q3) Integer overflow policy.** Integer elementwise/reduction ops are `Exact` *while in range*. The behaviour at the `DT` boundary (explicit `Err`, saturating, or widening) is a **kernel / `std.numerics` arithmetic policy**, not a `std.dense` choice; this spec assumes an explicit error (never silent wrap, per C1) but does not own the decision. Disposition: confirm the kernel's integer-overflow contract; FLAG, do not invent.
- **(Q4) Exact crate API surface (constructor / op names, the `DenseError` variants).** The §3 sketch is illustrative (the template's rule). The committed surface is the M-230 crate's actual API; the names here must be reconciled against it at ratification rather than asserted. Disposition: align §3 + §4 row labels with the landed crate before ratifying. (Ties to RFC-0016 §8-Q2, naming.)
- **(Q5) Ergonomics of the bound-carrying return (tension A).** Float ops return value-plus-bound; how much of the ε-tag/EXPLAIN machinery is implicit-but-inspectable vs. explicit at the call site is the RFC-0016 §8-Q3 (ergonomics-vs-contract) tension, instantiated for dense math. Disposition: a per-ring design pass with M-512; FLAGGED, not settled here.

## Meta — changelog

- **2026-06-17 — Draft (needs-design).** Stands up the `std.dense` (M-518, #160; Ring 1, Tier A) module design spec: the typed dimension-tracked `Dense{dim, dtype}` surface (constructors, elementwise, reductions, similarity), with the **honesty crux** that `dim`/`dtype` live in the *type* — a mismatch is a typed error at the boundary (a static type error for typed ops; an explicit `DenseError` for the dynamic `from_slice`), never a silent broadcast, narrowing, or re-round (C1/G2) — and that every float op carries its **ε from the ADR-010 `ErrorBound` kernel**, tagged **`Proven` only where the rounding-error bound's side-conditions are checked**, else honestly **downgraded** to `Empirical` (C2/VR-5: integer ops `Exact`; `l2_norm`/`cosine` `Empirical` pending a checked sqrt/division composition). Delivers the load-bearing **guarantee matrix** (§4: constructors / elementwise / reductions / similarity rows × tag · fallibility · effects · EXPLAIN), the §4.1 **C1–C6 conformance**, the **grounding** (RFC-0001 §4.1 / §4.3 / §4.7; RFC-0002 §5; ADR-010; M-230; RFC-0016 §4.1/§4.3/§4.5), and **five FLAGGED open questions** (the exact Higham bound + its checked side-conditions; whether `l2_norm`/`cosine` can earn `Proven`; integer overflow policy; the exact M-230 crate API; the bound-carrying-return ergonomics — tied to RFC-0016 §8-Q1/Q2/Q3). Consumes the M-230 dense crate + the ADR-010 numerics certificate (via `std.numerics`, M-512); adds **no** trusted code (KC-3). No code with this draft; ratification is the maintainer's append-only decision. Append-only.

- **2026-06-20 — Accepted (maintainer ratification, DN-07).** The maintainer ratified this Rust-first spec: the §4.5 guarantee matrix is asserted in tests, never-silent fallibility and honest per-op tags hold, and the open §7/§8 questions are design/scope calls, not contract violations. No guarantee tag was upgraded without a checked basis (VR-5). The Proven/Empirical rows were verified/aligned by M-377 (M-512 delivered) — dense elementwise Proven via the ADR-010 IEEE backward-error bound (finiteness side-condition guarded), accumulation rows held at Empirical; nothing upgraded on faith. Status moves *Implemented (Rust-first) — pending ratification → Accepted*. Append-only; no kernel change (KC-3).
