# PADO-Pauli — Tutorial Notebooks

Tutorial notebooks for the PADO-Pauli high-level tensor API.

The notebooks build circuits with the **`Circuit` builder API** (`from padopauli import Circuit`):
gates are stacked with methods (`qc.ry(0, param_idx=0)`, `qc.cnot(0, 1)`, …), compiled once with
an explicit `qc.compile(observables=..., preset=...)`, and evaluated with `qc.expvals(theta)` —
optionally inspected with `qc.draw()`. `Circuit` is a facade over the functional API
(`PauliRotation` / `CliffordGate` / `PauliSum` + `compile_program`), which remains available.

## Notebook index

1. `01_quickstart.ipynb`
   - End-to-end quickstart: the compile-once / evaluate-many idea, then building a circuit
     with the `Circuit` builder, `qc.draw()`, and `qc.compile(observables=..., preset=...)`.
   - Gradients via `.backward()` (`diff_mode="vjp"` vs `"autograd"`), parameter
     batches, and data embeddings (`embedding_idx=` + `qc.expvals(theta, embedding=x)`).
   - Accuracy vs cost: `max_weight` / `build_thetas` + `build_min_abs` truncation,
     noise channels, and a PennyLane statevector cross-check.

2. `02_pennylane_reference_api.ipynb`
   - How to use the small-qubit PennyLane reference API for expvals and sampling.

3. `03_training_with_compiled_program.ipynb`
   - Explicit PyTorch training loop built on the compiled program's expval path (`Circuit.expvals`).

4. `04_embedding_batched_inputs_basics.ipynb`
   - Embedding + batched input tutorial in three steps:
     1) 1D sin-regression,
     2) 2D linear binary classification,
     3) XOR classification.
   - Demonstrates training only on `thetas` while feeding batched embedding inputs.

5. `05_preset_tuning_gpu_budget.ipynb`
   - How to tune presets with `resolve_preset(...)` + `preset_overrides` under GPU resource constraints.

6. `06_quasi_probability_workflow.ipynb`
   - Z-moments and truncated quasi-probability reconstruction via `build_quasi_sampler`,
     checked against exact PennyLane probabilities at full order.

7. `07_parameter_batch_evaluation.ipynb`
   - Outer-product batched evaluation: feeding batched `embedding` inputs and batched `thetas`
     through a single compiled program, producing the `(e_bs, p_bs, N_obs)` output shape.
   - Cross-checks the batched result against an explicit nested-loop reference.

8. `08_hybrid_classical_quantum.ipynb`
   - Hybrid model: a `torch.nn.Linear` produces the embedding angles of a compiled program,
     and both are trained in one backward pass.
   - Verifies the gradient arriving at the classical weights against central differences,
     trains both parameter groups under one optimizer, and shows `embedding.detach()`
     freezing the upstream layer while the circuit angles keep training.

Application cases (kicked Ising, VQE, MaxCut-QAOA, SAFE ma-QAOA) live one folder up, in
`../examples/`.

## Feature map

| Feature (module) | Where used | What / How it is applied |
|---|---|---|
| `Circuit` (`padopauli`) | 01, 03, 04, 05, 06, 07, 08 | Builder used to define surrogate circuits consistently across benchmarking, training, quasi-probability, and batched evaluation: `qc.rx/ry/rz/rxx/...`, `qc.h/s/x/cnot/cz`, noise channels, and `qc.draw()`. |
| `Circuit.compile(observables=..., preset=...)` (`padopauli`) | 01, 03, 04, 05, 07, 08 | Explicit compile step. Observables are given as `(pauli, qubits[, coeff])` tuples (e.g., per-qubit Z observables); returns the reusable `CompiledProgram`. |
| `Circuit.expvals(...)` (`padopauli`) | 01, 03, 04, 07, 08 | Main evaluation path after compile. In 03 it drives an explicit PyTorch autograd training loop; in 04 and 07 it is used with batched `embedding` / `thetas` inputs (outer product); in 08 its `embedding` argument is the output of a classical layer, so `loss.backward()` reaches that layer's weights. |
| `PauliRotation`, `CliffordGate`, `PauliSum` (`padopauli`) + `compile_program(...)` | — | The low-level functional API `Circuit` wraps. Not used directly by the notebooks; it stays available for external code that works with raw gate lists or prebuilt `PauliSum` objects. |
| `resolve_preset(...)` + `preset_overrides` (`padopauli`) | 05 | Demonstrates preset-based tuning from the built-in presets (`cpu`, `gpu`, `hybrid`) and targeted overrides like `max_weight` / `dtype` for budget-aware execution. |
| `build_quasi_sampler(...)` (`padopauli`) | 02, 06 | Builds the quasi-probability sampler for moment computation and truncated quasi-probability reconstruction; takes the gate list directly (`circuit=qc.gates` from a `Circuit`). In 02 it is paired with reference sampling; 06 is the moment/quasi-probability walkthrough. |
| `Circuit.expvals_reference(...)` (`padopauli`) | 02 | Small-qubit exact expvals via PennyLane statevector simulation, for correctness checks against `Circuit.expvals(...)`. |
| `pennylane_sample(...)` (`padopauli`) | 02 | Exact small-qubit sampling helper used to compare surrogate-side results with reference bitstring samples. |

## Recommended order

Run notebooks in numeric order.
