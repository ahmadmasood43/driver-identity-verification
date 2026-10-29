"""Verification pairs: one rendered ID photo per subject against selfies from other videos.

Each subject gets a single ID-source video, so every pair involving that subject's ID uses
the same ID photo, like a real applicant with one document. Impostor selfies come from
other subjects with the same demographic key (gender, age band, Fitzpatrick group), which
is the harder and more honest test than random impostors.
"""

from collections import defaultdict
from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pair:
    id_subject: str
    id_video: str
    selfie_subject: str
    selfie_video: str

    @property
    def genuine(self) -> bool:
        return self.id_subject == self.selfie_subject


def choose_id_videos(videos: Mapping[str, list[str]], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    return {s: sorted(videos[s])[rng.integers(len(videos[s]))] for s in sorted(videos)}


def make_pairs(
    videos: Mapping[str, list[str]],
    match_key: Mapping[str, Hashable],
    *,
    selfies_per_subject: int = 2,
    impostors_per_subject: int = 10,
    seed: int = 0,
) -> list[Pair]:
    id_video = choose_id_videos(videos, seed)
    selfie_pool = {s: sorted(set(videos[s]) - {id_video[s]}) or [id_video[s]] for s in videos}
    by_key: dict[Hashable, list[str]] = defaultdict(list)
    for s in sorted(videos):
        by_key[match_key[s]].append(s)

    rng = np.random.default_rng(seed + 1)
    pairs = []
    for s in sorted(videos):
        others = sorted(set(videos[s]) - {id_video[s]})
        for v in rng.permutation(others)[:selfies_per_subject]:
            pairs.append(Pair(s, id_video[s], s, str(v)))

        rivals = [r for r in by_key[match_key[s]] if r != s]
        for r in rng.permutation(rivals)[:impostors_per_subject]:
            pool = selfie_pool[r]
            pairs.append(Pair(s, id_video[s], str(r), pool[rng.integers(len(pool))]))
    return pairs
