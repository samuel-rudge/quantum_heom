# Quantum HEOM Repository (Fully Quantum Vibronic Dynamics)

This repository implements a **fully quantum hierarchical equations of motion (HEOM)** framework for simulating nonadiabatic molecular dynamics of molecules interacting with surfaces and leads.

All **electronic and vibrational degrees of freedom are treated quantum mechanically on equal footing**, forming a unified vibronic Fock space. The code propagates the full reduced molecular density matrix using a sparse HEOM Liouvillian representation.

---

# Physical Scope

The framework is designed for:

- Molecular transport through metallic or molecular leads  
- Vibronic effects in electron–phonon coupled systems  
- Fully quantum nonadiabatic dynamics in open systems  
- Time-dependent quantum transport and relaxation phenomena  

The central object is the **full vibronic density matrix**, evolved under an open-system HEOM generator.

---

# Example Model: Fully Quantum Holstein System

A prototypical application is the **fully quantum Holstein model in an open system setting**.

The Hamiltonian takes the form:
\[
H = H_{\text{el}} + H_{\text{vib}} + H_{\text{el-vib}} + H_{\text{leads}} + H_{\text{mol-leads}}
\]

with:

**Electronic subsystem**
\[
H_{\text{mol,el}} = \sum_i \epsilon_i c_i^\dagger c_i + \sum_{ij} t_{ij} c_i^\dagger c_j
\]

**Vibrational subsystem**
\[
H_{\text{mol,vib}} = \sum_\nu \omega_\nu b_\nu^\dagger b_\nu
\]

**Electron–vibrational coupling (Holstein form)**
\[
H_{\text{mol,el-vib}} = \sum_{i,\nu} g_{i\nu} c_i^\dagger c_i (b_\nu^\dagger + b_\nu)
\]

**Lead coupling**
\[
H_{\text{coupling}} = \sum_{k\alpha,i} (V_{k\alpha i} c_i^\dagger a_{k\alpha} + \text{h.c.})
\]

---

# System Construction Philosophy

The code structure is modular and consistent across applications:

- `input_parameters.py` defines all physical, numerical, and algorithmic parameters  
- `system.py` constructs the full operator algebra in the combined electronic–vibrational Fock space  
- HEOM hierarchy generation builds auxiliary density operators (ADOs)  
- Sparse Liouvillian construction encodes the full evolution generator  
- Time propagation is performed using sparse linear algebra routines

The defining feature is that:

> The molecular Hamiltonian is constructed directly in a tensor-product vibronic Fock space, without any classical coordinate dependence.

---

## `input_parameters.py`

This module defines all tunable simulation parameters:

### Electronic structure
- Number of electronic orbitals (`Nel`)
- On-site energies and hopping amplitudes (`Single_El_Int`)
- Electron–electron interaction strengths (if included) (`Double_El_Int`)

### Vibrational structure
- Number of vibrational modes (`N_qu_vib_modes`)
- Maximum vibrational occupation (`max_occ_qu_vib_modes`)
- Vibrational frequencies (`freq_vector_qu_vib_modes`)
- Electron–vibrational coupling strengths (`el_vib_int_qu`)

### Environment / bath
- Temperature (`Kelvin_T`)
- Lead spectral densities
- Lead chemical potentials (`muvec`)
- Bath correlation decomposition settings

### HEOM control parameters
- Hierarchy truncation depth (`Nmax`)
- Accuracy of pole decomposition (`tol_fermi_symmetrized_barycentric`)
- Cutoff thresholds and tolerances
- Filtering parameters for ADO pruning

---

## `system.py`

This module constructs the full **vibronic operator algebra**, including:

- Electronic creation/annihilation operators \(c_i^\dagger, c_i\)
- Vibrational bosonic operators \(b_\nu^\dagger, b_\nu\)
- Tensor-product Fock space basis
- Full vibronic Hamiltonian (electronic + vibrational + coupling terms)
- Initial density matrix in the combined Fock space
- Identity and occupation operators
- Log-indexed operator representations for sparse mapping

The output is a complete operator dictionary defining the system in a basis suitable for HEOM propagation.

---

# HEOM Construction

The HEOM hierarchy is constructed via standard open quantum system decomposition:

- Bath correlation functions are expanded into sums of exponentials (Padé or barycentric decomposition)
- Each exponential term defines auxiliary modes in the hierarchy
- ADO index structures are generated up to truncation level `Nmax`
- Coupling rules between hierarchy tiers are encoded in sparse index maps

This produces:

- Hierarchy index tensors  
- ADO connectivity graphs  
- Bath decay rates and prefactors  

---

# Sparse Liouvillian Representation

The full HEOM superoperator is stored in a **sparse format**, where:

- Row/column structure is fixed after initialization  
- Only numerical values depend on system parameters  
- The Liouvillian includes:
  - Vibronic Hamiltonian commutator terms  
  - Lead coupling contributions  
  - Bath-induced dissipative terms via HEOM hierarchy  

This structure is optimized for:
- Sparse matrix–vector propagation  
- Intel MKL CSR sparse BLAS routines  
- OpenMP-parallel execution where applicable  

---

# Fortran Acceleration Layer

Performance-critical components are implemented in Fortran and interfaced via `f2py`.

These include:

- Bath correlation decomposition routines  
- HEOM hierarchy generation and filtering  
- Sparse Liouvillian construction  
- COO → CSR sparse matrix conversion  
- Sparse propagation kernels  

The final sparse Liouvillian is stored in CSR format using Intel MKL.

---

# Build Instructions

See BUILD.md for details. The Fortran modules must be compiled before running any simulation.

## Requirements

- gfortran (recommended)
- Intel MKL (required for sparse BLAS)
- OpenMP support
- Python with NumPy (f2py enabled)

---

## Compilation

Run the build script from the repository root:

```bash
./compile_f2py.sh