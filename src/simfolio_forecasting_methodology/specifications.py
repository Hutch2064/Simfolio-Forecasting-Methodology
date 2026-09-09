"""Typed statistical specifications for compositional canonical model IDs.

Many later catalogue rows are not bespoke algorithms. Their immutable source
IDs explicitly encode mean, volatility, innovation, resampling, and tail
components separated by ``|``. Parsing those components into a typed object
makes the research design auditable and lets one implementation serve all
valid combinations without losing historical identity.
"""

from __future__ import annotations

from dataclasses import dataclass

KNOWN_MEAN_MODELS = {
    "bic_auto_arma_mean",
    "expanding_sample_mean",
    "factor_premium_near_zero_alpha_shrinkage",
}
KNOWN_VOLATILITY_MODELS = {
    "constant_sample_volatility",
    "garch_1_1_volatility",
    "gjr_tarch_1_1_volatility",
    "egarch_1_1_volatility",
}
KNOWN_PARAMETRIC_INNOVATIONS = {
    "gaussian_iid_standardized_innovations",
    "student_t_standardized_innovations",
    "skew_t_standardized_innovations",
}
KNOWN_EMPIRICAL_RESAMPLING = {"iid", "stationary_bootstrap"}
KNOWN_EMPIRICAL_TAILS = {"filtered_empirical_tail", "automated_evt_pot_gpd_tail"}


@dataclass(frozen=True)
class CompositionalSpec:
    source_id: str
    mean_model: str
    volatility_model: str
    innovation_model: str
    resampling: str | None
    tail_method: str

    @property
    def is_parametric(self) -> bool:
        return self.resampling is None

    def validate(self) -> None:
        if self.mean_model not in KNOWN_MEAN_MODELS:
            raise ValueError(f"unknown mean model: {self.mean_model}")
        if self.volatility_model not in KNOWN_VOLATILITY_MODELS:
            raise ValueError(f"unknown volatility model: {self.volatility_model}")
        if self.innovation_model == "empirical":
            if self.resampling not in KNOWN_EMPIRICAL_RESAMPLING:
                raise ValueError(f"unknown empirical resampling method: {self.resampling}")
            if self.tail_method not in KNOWN_EMPIRICAL_TAILS:
                raise ValueError(f"unknown empirical tail method: {self.tail_method}")
        else:
            if self.innovation_model not in KNOWN_PARAMETRIC_INNOVATIONS:
                raise ValueError(f"unknown parametric innovation model: {self.innovation_model}")
            if self.resampling is not None or self.tail_method != "parametric":
                raise ValueError("parametric innovations require parametric tail and no resampling")


def parse_compositional_spec(source_id: str) -> CompositionalSpec:
    """Parse an immutable pipe-delimited source ID into statistical components."""

    parts = source_id.split("|")
    if len(parts) == 4:
        mean_model, volatility_model, innovation_model, tail_method = parts
        spec = CompositionalSpec(
            source_id=source_id,
            mean_model=mean_model,
            volatility_model=volatility_model,
            innovation_model=innovation_model,
            resampling=None,
            tail_method=tail_method,
        )
    elif len(parts) == 5:
        mean_model, volatility_model, innovation_model, resampling, tail_method = parts
        spec = CompositionalSpec(
            source_id=source_id,
            mean_model=mean_model,
            volatility_model=volatility_model,
            innovation_model=innovation_model,
            resampling=resampling,
            tail_method=tail_method,
        )
    else:
        raise ValueError(f"not a supported compositional source ID: {source_id}")
    spec.validate()
    return spec
