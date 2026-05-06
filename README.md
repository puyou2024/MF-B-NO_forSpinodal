# MF-B-NO: Multi-Fidelity Bayesian Neural Operator for Spinodal Metamaterials

**A Multi-Fidelity Bayesian Neural Operator for Mechanics of Spinodal Metamaterial**

This repository contains the implementation of the **Multi-Fidelity Bayesian Neural Operator (MF-B-NO)** framework. This framework integrates abundant low-fidelity (LF) finite element simulations with sparse high-fidelity (HF) experimental data to efficiently predict the nonlinear mechanical response of spinodal metamaterials.

## Abstract

Data-driven modeling of nonlinear responses for metamaterials is often constrained by the high cost of experimental data. To address this, we propose a Multi-Fidelity (MF) framework that aggregates:
1.  **Low-Fidelity (LF) Data:** Finite element simulations (ABAQUS) which provide abundant but idealized data.
2.  **High-Fidelity (HF) Data:** In-situ nanomechanical experiments which capture real-world physics but are sparse and expensive.

The framework utilizes a **Bayesian DeepONet** architecture to enable uncertainty quantification. It employs a **Hybrid Active Learning** strategy to efficiently select LF samples by maximizing both epistemic uncertainty and geometric diversity, significantly reducing computational costs. The final model achieves an 84.1% reduction in Mean Squared Error (MSE) compared to high-fidelity baselines using only ~22 LF samples and 20 HF samples.

## Framework Architecture

The framework consists of a two-stage training process:

### Stage 1: Low-Fidelity (LF) Model with Active Learning
The LF model is a Bayesian DeepONet trained on FEM simulation data.
* **Input:** Design angles $u = \{\theta_1, \theta_2, \theta_3\}$ representing the spinodal topology.
* **Output:** Stress-strain response $\sigma_L$.
* **Active Learning:** A hybrid strategy selects informative samples from a pool of 3000 designs to iteratively retrain the model.

### Stage 2: High-Fidelity (HF) Residual Learning
The HF model learns the discrepancy (residual) between the LF predictions and the experimental reality.
* **Branch Net:** A 3D CNN that processes the **Signed Distance Field (SDF)** of the microstructure to capture geometric irregularities.
* **Trunk Net:** Augmented with the LF model's predicted stress ($\sigma_L$) along with standard loading conditions.
* **Prediction:** The final output is the sum of the LF prediction and the learned Bayesian residual.

## Installation

To set up the environment, please install the required dependencies. It is recommended to use a virtual environment.

```bash
# Clone the repository


# Install dependencies
