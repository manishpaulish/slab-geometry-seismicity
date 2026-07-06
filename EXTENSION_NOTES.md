# Extension: assignment method and rupture dynamics

This extends the original curvature-seismicity reproduction in two directions.

## 1. Earthquake-to-slab assignment (`src/depth_projection.py`)

Replaces the rectangular bounding-box assignment with **depth projection**:
each earthquake is kept only if its hypocentral depth lies within a tolerance
of the interpolated Slab2 slab surface at its (lon, lat).

Run: `python run_depth_projection.py` and `python diagnose_projection.py`

Result: all curvature-seismicity correlations strengthen, peaking at a
tolerance of ~50 km before declining (consistent with a seismogenic band of
that thickness). Mean|H| vs M_max moves from -0.43 (bounding box) to -0.49
(50 km projection), closing ~1/3 of the gap to the published -0.59.

## 2. Rate-and-state rupture dynamics (`src/dynamics/`)

A numerically stable rate-and-state friction model in the direction of
Erickson, Birnir & Lavallee (2011, GJI 187, 178-198).

- `rate_state_slider.py` : single-block quasi-dynamic slider. Handles the
  coseismic stiffness via a regularized friction law (valid at V=0) and a
  radiation-damping formulation, with the slip velocity recovered each step
  by a bracketed root find.
- `bk_chain.py` : N-block Burridge-Knopoff chain with a vectorized velocity
  solve. Reproduces the **size-dependent transition to chaos**: small chains
  slip periodically (CV of recurrence intervals ~0), larger chains become
  chaotic (CV > 1). Onset location depends on coil-spring coupling strength.

See `results/bk_periodic_chaotic_transition.png`.

## Next step

Connect the two: whether slab curvature maps onto the control parameters
(effective system size, coupling) that govern the periodic-to-chaotic
transition, turning the geometry-seismicity correlation into a statement
about rupture dynamics.
