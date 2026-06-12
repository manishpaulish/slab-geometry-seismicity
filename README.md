# Slab Geometry & Seismicity

**A computational extension of Chau, Bendick, Choi & Mahadevan (2026)**  
arXiv: [2606.02520](https://arxiv.org/abs/2606.02520)

---

## Motivation

Chau et al. (2026) demonstrate that the differential geometry of subduction zone surfaces — specifically, mean curvature *H* and Gaussian curvature *K* computed from Slab2 depth grids — correlates significantly with earthquake productivity (Pearson *r* up to −0.60 for Mean|*K*| vs. *M*_max).  Their closing sentence calls explicitly for *"computational models and predictive frameworks for earthquake risk"* that incorporate this geometric information.

This repository provides that framework. It

1. **Reproduces** the curvature analysis and Table 1 of Chau et al. from scratch, using public data only.
2. **Extends** the linear correlation to nonlinear models: Gaussian Process regression (with LOO-CV uncertainty quantification) and a proof-of-concept Fourier Neural Operator applied to curvature power spectra.
3. **Connects** to seismic wave-propagation modelling via the FNO framework of Li et al. (2021), bridging the source-geometry problem (slab shape → seismicity) to the forward-propagation problem (source → surface ground motion).

---

## Repository structure

See uploaded source files in `src/`, `tests/`, and `requirements.txt`.

---

## Author

Manish Paul  
B.Tech Applied Geology (HPC micro-specialisation), IIT Kharagpur (batch 2028)  
manishpaul.24@kgpian.iitkgp.ac.in
