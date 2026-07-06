"""
dynamics
--------
Rate-and-state friction earthquake models, in the direction of
Erickson, Birnir & Lavallee (2011), GJI 187, 178-198.

  rate_state_slider : single-block quasi-dynamic RSF slider (stable integrator)
  bk_chain          : N-block Burridge-Knopoff chain; reproduces the
                      size-dependent periodic-to-chaotic transition
"""
from .rate_state_slider import RateStateSlider, classify_events
from .bk_chain import BKChain

__all__ = ["RateStateSlider", "classify_events", "BKChain"]
