# 2.5D ERT Forward Problem with PINN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the paper:

**"A Mesh-Free Forward Solver for 2.5D Electrical Resistivity Tomography using Physics-Informed Neural Networks"**

**Authors:** [Kuzma Tsukanov], [Caner Sakar], [Gordon Osterman], [Ziv Moreno]
**Journal:** Submitted to the journal of *Water Resources Research*

---

## 📌 Overview

This project implements a **Physics-Informed Neural Network (PINN)** to solve the **2.5D Electrical Resistivity Tomography
(ERT) forward problem**. Unlike traditional finite-element methods (FEM) that rely on global matrix assembly and mesh generation,
this framework solves the governing modified Poisson equation directly in the continuous spatial domain.

### Key Features
*   **Mesh-Independent:** Solves potential fields on continuous coordinates $(x, z)$ without grid discretization errors.
*   **2.5D Formulation:** Implements the wavenumber domain transformation ($k_y$) to simulate 3D point sources in 2D geology.
*   **Singularity Removal:** Utilizes a primary/secondary potential decomposition to handle source singularities analytically.
*   **Adaptive Training:** Implements Residual-based Adaptive Refinement (RAR) to automatically enrich collocation points
in high-error regions.
*   **Compact Domain Handling:** Demonstrates superior accuracy at domain boundaries compared to standard numerical solvers
(e.g., pyGIMLi) without mesh padding.

---

## ⚙️ Installation

### Prerequisites
*   Python 3.8+
*   CUDA-enabled GPU (Recommended for training)

### Dependencies
Install the required packages using pip:

```bash
pip install -r requirements.txt
