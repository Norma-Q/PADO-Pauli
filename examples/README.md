# PADO-Pauli — Application Example Notebooks

The application cases of the PADO-Pauli paper, one runnable notebook each. These are end-to-end applications, not API walkthroughs — for the interface
itself start with `../tutorial/` (`README_TUTORIAL.md`).

Every notebook imports the installed `padopauli` package (see the repository README) and writes
its figures into `figures/` next to it.

## Notebook index

| Notebook | Topic | What it does |
|---|---|---|
| `01_kicked_ising_127q.ipynb` | utility-scale example | 127-qubit kicked transverse-field Ising on the IBM Eagle heavy-hex topology, `<Z_62>` vs the RX field angle. |
| `02_vqe_h2_molecule.ipynb` | variational quantum eigensolver | H₂/STO-3G Hamiltonian from PennyLane `qchem` as a `PauliSum`, VQE with `torch.optim`, FCI check, bond-dissociation curve. |
| `03_maxcut_qaoa.ipynb` | MaxCut-QAOA | QAOA angles trained on the surrogate MaxCut objective, calibrated against the brute-force optimum, then sampled for the cut distribution. |
| `04_SAFE_ma-QAOA.ipynb` | SAFE ma-QAOA | SAFE workflow on a 16-qubit Erdos-Renyi MaxCut instance: LWPP (Low-Weight Pauli Propagation) surrogate warmup, parameter distillation, exact fine-tuning against a statevector vs an exact-only baseline. |

`kicked_ising_127q/` holds the batch scripts, recorded results and figure for the
kicked-Ising example; see `kicked_ising_127q/README.md` for the setup and re-run commands.

## Running a notebook

```bash
pip install matplotlib jupyter   # notebook-only extras (padopauli itself pulls in torch/numpy/scipy)
cd examples
jupyter lab            # or: jupyter nbconvert --to notebook --execute --inplace 02_vqe_h2_molecule.ipynb
```
