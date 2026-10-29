"""Subgroup bins fixed in eval/protocol.md."""


def age_band(age: int) -> str:
    if age < 18:
        raise ValueError(f"age {age} below 18")
    if age <= 30:
        return "18-30"
    if age <= 45:
        return "31-45"
    return "46+"


_FITZ = {1: "I-II", 2: "I-II", 3: "III-IV", 4: "III-IV", 5: "V-VI", 6: "V-VI"}


def fitzpatrick_group(skin_type: int) -> str:
    try:
        return _FITZ[int(skin_type)]
    except KeyError:
        raise ValueError(f"Fitzpatrick type must be 1-6, got {skin_type}") from None
