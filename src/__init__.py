"""
slab-geometry-seismicity
------------------------
Computational extension of Chau, Bendick, Choi & Mahadevan (2026),
arXiv:2606.02520 — "How geometry of subduction zones correlates with
earthquake dynamics."

Modules
-------
slab_loader          : Download and grid the Slab2 depth data
earthquake_catalog   : Query USGS ComCat for per-zone seismicity
curvature            : Finite-difference H and K computation
correlation          : Pearson and multi-scale analysis (reproduces Table 1)
ml_model             : Gaussian Process and FNO-based extensions
"""

from .slab_loader import (
    ZONES, download_all_zones, load_zone_grid, load_all_zones,
)
from .earthquake_catalog import (
    ZONE_BOUNDS, download_all_catalogs, compute_productivity,
)
from .curvature import (
    compute_curvature, zone_curvature_stats,
)
from .correlation import (
    build_zone_dataframe, reproduce_table1, multiscale_correlation,
)
from .ml_model import fit_gp, compare_models, SlabFNO

__version__ = "0.1.0"
__all__ = [
    "ZONES", "ZONE_BOUNDS",
    "download_all_zones", "load_zone_grid", "load_all_zones",
    "download_all_catalogs", "compute_productivity",
    "compute_curvature", "zone_curvature_stats",
    "build_zone_dataframe", "reproduce_table1", "multiscale_correlation",
    "fit_gp", "compare_models", "SlabFNO",
]
