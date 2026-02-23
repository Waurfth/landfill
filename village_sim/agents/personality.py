"""Personality trait generation with correlated distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from village_sim.core.config import (
    FEMALE_STRENGTH_MEAN,
    FEMALE_STRENGTH_STD,
    MALE_STRENGTH_MEAN,
    MALE_STRENGTH_STD,
    TRAIT_CORRELATIONS,
    TRAIT_MEAN,
    TRAIT_STD,
)


@dataclass
class PersonalityTraits:
    """All traits on 0-100 scale."""

    # Physical
    strength: float = 50.0
    endurance: float = 50.0
    dexterity: float = 50.0

    # Mental / Personality
    intelligence: float = 50.0
    patience: float = 50.0
    risk_tolerance: float = 50.0
    sociability: float = 50.0
    ambition: float = 50.0
    conscientiousness: float = 50.0
    empathy: float = 50.0
    creativity: float = 50.0

    # Emotional baseline
    baseline_optimism: float = 50.0
    emotional_stability: float = 50.0
    loss_aversion: float = 50.0


# Ordered list of correlated traits (excluding strength, which is sex-specific)
_CORRELATED_TRAITS: list[str] = [
    "endurance", "dexterity", "intelligence", "patience", "risk_tolerance",
    "sociability", "ambition", "conscientiousness", "empathy", "creativity",
    "baseline_optimism", "emotional_stability", "loss_aversion",
]


def generate_personality(sex: str, rng: Generator) -> PersonalityTraits:
    """Generate a unique personality profile with correlated traits."""
    n = len(_CORRELATED_TRAITS)

    # Build correlation matrix
    corr = np.eye(n)
    trait_index = {name: i for i, name in enumerate(_CORRELATED_TRAITS)}
    for trait_a, trait_b, r in TRAIT_CORRELATIONS:
        if trait_a in trait_index and trait_b in trait_index:
            ia, ib = trait_index[trait_a], trait_index[trait_b]
            corr[ia, ib] = r
            corr[ib, ia] = r

    # Ensure positive semi-definite
    eigvals = np.linalg.eigvalsh(corr)
    if np.any(eigvals < 0):
        corr += np.eye(n) * (abs(eigvals.min()) + 0.01)
        # Re-normalize diagonal
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)

    # Generate correlated standard normals
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal(n)
    correlated = L @ z

    # Scale to trait values
    values: dict[str, float] = {}
    for i, name in enumerate(_CORRELATED_TRAITS):
        raw = TRAIT_MEAN + correlated[i] * TRAIT_STD
        values[name] = float(np.clip(raw, 1, 99))

    # Strength uses sex-specific distribution
    if sex == "male":
        strength = rng.normal(MALE_STRENGTH_MEAN, MALE_STRENGTH_STD)
    else:
        strength = rng.normal(FEMALE_STRENGTH_MEAN, FEMALE_STRENGTH_STD)
    values["strength"] = float(np.clip(strength, 1, 99))

    return PersonalityTraits(**values)


def inherit_traits(
    parent_a: PersonalityTraits,
    parent_b: PersonalityTraits,
    child_sex: str,
    rng: Generator,
) -> PersonalityTraits:
    """Generate child traits influenced by parents with randomness."""
    values: dict[str, float] = {}
    for name in _CORRELATED_TRAITS:
        pa = getattr(parent_a, name)
        pb = getattr(parent_b, name)
        # Average of parents + random deviation
        midpoint = (pa + pb) / 2.0
        deviation = rng.normal(0, TRAIT_STD * 0.5)
        values[name] = float(np.clip(midpoint + deviation, 1, 99))

    # Strength: sex-specific inheritance
    pa_s = parent_a.strength
    pb_s = parent_b.strength
    mid_s = (pa_s + pb_s) / 2.0
    if child_sex == "male":
        target_mean = MALE_STRENGTH_MEAN
    else:
        target_mean = FEMALE_STRENGTH_MEAN
    # Blend parental midpoint with sex-typical mean
    blended = mid_s * 0.5 + target_mean * 0.5
    values["strength"] = float(np.clip(blended + rng.normal(0, 8), 1, 99))

    return PersonalityTraits(**values)
