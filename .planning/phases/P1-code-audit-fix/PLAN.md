# P1 — Code audit & fix

**Goal**: Eliminate all discrepancies between the implementation and the (approved) theory in report Chapters 1–4. After P1, the code computes exactly what the report defines.

**Deliverable**: Updated `progetto/code/src/*.py` such that:
1. Objective `f(w) = ½‖Xw-y‖² + λ‖w‖₁` matches report Eq (1.3).
2. IRLS Q-update matches report Eq (2.6) derivation.
3. DSM matches report Algorithm 2 (OLS warm start, β_i = min(β, γ_i)).
4. `make_elm_problem` reference solver mapping is correct.
5. Existing `test_basic.py` still passes after fixes (after adjusting tolerances/expected behaviour where the math changed).

**Out of scope**: New tests (P2), new experiments (P3), report writing (P4–P5).

## Audit findings (full evidence)

| ID | File:line | Current | Should be | Theory ref |
|---|---|---|---|---|
| C1 | `lasso_utils.py:33` | `‖Xw-y‖² + λ‖w‖₁` | `½‖Xw-y‖² + λ‖w‖₁` | Report Eq (1.3) |
| C2 | `lasso_utils.py:51` | `2 Xᵀ(Xw-y)` | `Xᵀ(Xw-y)` | Report § 1.5.2 |
| C3 | `irls.py:109` | `Q[i,i] += 0.5·λ·D[i]` | `Q[i,i] += λ·D[i]` after C1/C2 fix | Report Eq (2.6) → normal eq |
| C4 | `deflected_subgradient.py:104` | `w = np.zeros(n)` | `w = solve_spd(A + ε I, b)` (OLS warm start) | Report § 3.4 |
| C5 | `deflected_subgradient.py:158` | `numerator = beta * (f_curr - target)` | `β_i = min(β, γ_i); numerator = β_i * (f_curr - target)` | Alg 2 line 11, § 3.2 |
| C6 | `data_generation.py:137` | `alpha_sk = lam * m / 2.0` | `alpha_sk = lam / (2.0*m)` (current f) → `lam / m` (report f) | sklearn loss = (1/2m)‖·‖² + α‖·‖₁ |
| C7 | `lasso_utils.py:104` | KKT uses `2 Xᵀ(...)` and threshold `λ` | `Xᵀ(...)` and threshold `λ` after C2 fix | Report Eq (1.4) |

## Tasks

### T1.1 — Update objective and gradient definitions [atomic]
**Files**: `lasso_utils.py`
**Changes**:
- `f_lasso`: return `0.5 * np.dot(residual, residual) + lam * np.sum(np.abs(w))`
- `grad_smooth`: return `X.T @ (X @ w - y)` (factor 2 removed)
- `subgradient_f`: unchanged in logic but verify it now returns `Xᵀ(Xw-y) + λs` (since `grad_smooth` changed)
- `check_optimality`: threshold becomes `|g_smooth_i| ≤ λ` for `w_i = 0`, and `g_smooth_i + λ sign(w_i) = 0` for `w_i ≠ 0` — same form, just applied to the new `grad_smooth`.

**Verify**: numerical gradient check in `test_basic.py` uses central finite differences on `f_lasso`; expected analytic result for the smooth part is now `g_smooth = grad_smooth(...)` directly — adjust the assertion accordingly.

**Commit**: `fix(lasso): align objective and gradient with report definition (½‖Xw-y‖² + λ‖w‖₁)`

### T1.2 — Update IRLS Q-update to match Eq (2.6) [atomic]
**Files**: `irls.py`
**Changes**:
- Line 109: `Q[i,i] += lam * diag_D[i]` (was `0.5 * lam`).
- Update top-of-file docstring: surrogate is `Q(w,w_k) = ½‖Xw-y‖² + (λ/2)‖W_k w‖² + (λ/2)‖w_k‖₁` and normal eq is `(XᵀX + λ W_kᵀW_k) w = Xᵀ y`.

**Derivation (write into commit message for traceability)**:
∇_w Q(w, w_k) = Xᵀ(Xw - y) + λ W_kᵀW_k w = 0
⇒ (XᵀX + λ W_kᵀW_k) w = Xᵀ y, with (W_kᵀW_k)_{ii} = 1/max(|w_{k,i}|, ε_thr).

**Commit**: `fix(irls): correct Q-update coefficient to λ (match report Eq 2.6 normal equation)`

### T1.3 — DSM OLS warm-start [atomic]
**Files**: `deflected_subgradient.py`
**Changes**:
- Lines 99–107: replace `w = np.zeros(n)` default with `w = cholesky_solve(X.T @ X + 1e-12*I, X.T @ y)` when `w0 is None`.
- Add import for `solve_spd`.
- Document in docstring: "default `w0` is the ordinary least-squares solution, identical to IRLS, per report § 3.4."

**Commit**: `fix(dsm): use OLS warm-start as default (match report § 3.4)`

### T1.4 — DSM β_i clipping per Algorithm 2 [atomic]
**Files**: `deflected_subgradient.py`
**Changes**:
- After computing `gamma`, add `beta_i = min(beta, gamma)`.
- Line 158: `numerator = beta_i * (f_curr - target)` (was `beta`).
- Update docstring to reflect that `beta` is the user-supplied β ∈ (0,1] and `β_i` is the clipped per-iteration value.

**Commit**: `fix(dsm): enforce β_i = min(β, γ_i) per Algorithm 2 stepsize-restricted condition`

### T1.5 — Fix `make_elm_problem` sklearn α mapping [atomic]
**Files**: `data_generation.py`
**Changes**:
- Line 137: `alpha_sk = lam / (2.0 * m)` if we keep the post-C1 objective (½‖·‖² + λ‖·‖₁), this becomes `alpha_sk = lam / m`.
  - Derivation: sklearn loss `(1/(2m))‖Xw-y‖² + α‖w‖₁`. Multiply by `m`: `(1/2)‖Xw-y‖² + mα‖w‖₁`. Match `mα = λ` ⇒ `α = λ/m`.
- Same fix in `make_lasso_problem` (line 67): currently `lam / (2m)` for old f; becomes `lam / m` for new f.

**Commit**: `fix(data): correct sklearn α mapping after objective rescaling`

### T1.6 — Update `test_basic.py` expectations [atomic]
**Files**: `test_basic.py`
**Changes**:
- The numerical-gradient check at lines 47–53 currently subtracts `lam * sign(w)` from the finite-difference result. After C1/C2, `grad_smooth` matches the smooth-only finite-difference of `f_lasso` directly — simplify the assertion: `np.allclose(grad_smooth(X,y,w), (f(w+e)-f(w-e))/(2e) - lam*sign(w))`.
- All other tests (IRLS converged, monotonicity, optimality, DSM record non-increasing, gap < 0.1) should still pass with new f. **Run** `python test_basic.py` and confirm.

**Commit**: `test(basic): update finite-difference assertion to match new gradient scaling`

## Verification gate (P1 → P2)

After T1.1–T1.6:
1. `cd progetto/code && python test_basic.py` ⇒ all `[PASS]`.
2. Visual sanity: run a quick IRLS on `make_lasso_problem(n=30, m=100)` and check `f_vals` is monotone non-increasing AND `gap[-1] < 1e-4`.
3. Visual sanity: run DSM with same problem; `f_bar` monotone non-increasing AND `gap[-1] < 1e-2` within 5000 iters.

If any check fails: do not proceed to P2 — diagnose, fix, re-verify.

## Risk register

- **R1**: existing `results/figures/*.pdf` were generated against the old (buggy) objective. They MUST be regenerated in P3. Do not reuse.
- **R2**: report Chapters 1–4 were approved against the **corrected** math (which is the standard math). The old code was the wrong one. After P1 the code matches what the report claims, which is what the prof read. ✓
- **R3**: changing β_i clipping in DSM may slow convergence on some problems; this is expected and matches theory (Cap 3.2, "natural default β = 1 triggers clipping only when greedy γ pushes < 1").
