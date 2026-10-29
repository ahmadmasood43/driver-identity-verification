"""Biometric error rates.

Convention everywhere: a higher score means more likely genuine (verification) or
bona fide (PAD), and a sample is accepted when score >= threshold.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def _scores(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("expected a non-empty 1-D array of scores")
    return a


def far(impostor: ArrayLike, threshold: float) -> float:
    return float(np.mean(_scores(impostor) >= threshold))


def frr(genuine: ArrayLike, threshold: float) -> float:
    return float(np.mean(_scores(genuine) < threshold))


def threshold_at_far(impostor: ArrayLike, target: float) -> float:
    """Lowest threshold whose FAR does not exceed `target`.

    Tied scores can push the achieved FAR below target; it never goes above.
    """
    if not 0.0 <= target <= 1.0:
        raise ValueError("target must be in [0, 1]")
    s = np.sort(_scores(impostor))[::-1]
    # Epsilon guards against e.g. 0.29 * 100 == 28.999999999999996.
    allowed = math.floor(target * s.size + 1e-9)
    if allowed >= s.size:
        return -math.inf
    return float(np.nextafter(s[allowed], math.inf))


def error_curve(
    genuine: ArrayLike, impostor: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FAR and FRR at every distinct observed score, thresholds ascending."""
    g = np.sort(_scores(genuine))
    i = np.sort(_scores(impostor))
    thresholds = np.unique(np.concatenate([g, i]))
    far_ = 1.0 - np.searchsorted(i, thresholds, side="left") / i.size
    frr_ = np.searchsorted(g, thresholds, side="left") / g.size
    return thresholds, far_, frr_


def eer(genuine: ArrayLike, impostor: ArrayLike) -> tuple[float, float]:
    """(EER, threshold) at the observed threshold where FAR and FRR are closest."""
    t, fa, fr = error_curve(genuine, impostor)
    k = int(np.argmin(np.abs(fa - fr)))
    return float((fa[k] + fr[k]) / 2), float(t[k])


@dataclass(frozen=True)
class PadErrors:
    apcer: float  # worst PAI species, per ISO/IEC 30107-3
    bpcer: float
    acer: float  # not an ISO metric; kept for comparison with CelebA-Spoof papers
    apcer_per_species: dict[str, float]


def pad_errors(
    bona_fide: ArrayLike, attack: ArrayLike, species: ArrayLike, threshold: float
) -> PadErrors:
    attack_s = _scores(attack)
    species_a = np.asarray(species)
    if species_a.shape != attack_s.shape:
        raise ValueError("species must align with attack scores")
    per = {str(sp): far(attack_s[species_a == sp], threshold) for sp in np.unique(species_a)}
    apcer = max(per.values())
    bpcer = frr(bona_fide, threshold)
    return PadErrors(apcer, bpcer, (apcer + bpcer) / 2, per)


def bpcer_at_apcer(
    bona_fide: ArrayLike, attack: ArrayLike, species: ArrayLike, target: float = 0.01
) -> tuple[float, float]:
    """(BPCER, threshold) at the lowest threshold where every species meets `target` APCER."""
    attack_s = _scores(attack)
    species_a = np.asarray(species)
    t = max(threshold_at_far(attack_s[species_a == sp], target) for sp in np.unique(species_a))
    return frr(bona_fide, t), t
