"""MIDV-2020 annotations (VIA 2 JSON, one file per document type)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Field:
    name: str
    value: str
    points: np.ndarray  # N x 2, clockwise from top-left for rectangles

    @property
    def box(self) -> tuple[int, int, int, int]:
        """x, y, width, height of the bounding rectangle."""
        x0, y0 = self.points.min(axis=0)
        x1, y1 = self.points.max(axis=0)
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


@dataclass(frozen=True)
class Template:
    doc_type: str
    image: Path
    fields: dict[str, Field]

    @property
    def gender(self) -> str | None:
        """M or F. Bilingual values ("K/M", "Ž / F") carry the English letter last."""
        f = self.fields.get("gender")
        if f is None or not f.value.strip():
            return None
        v = f.value.split("/")[-1].strip().upper()
        return _GENDER.get(v)


# rus_internalpassport writes the Cyrillic abbreviations.
_GENDER = {"M": "M", "F": "F", "МУЖ.": "M", "ЖЕН.": "F"}


def _points(shape: dict) -> np.ndarray:
    if shape["name"] == "rect":
        x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)
    if shape["name"] == "polygon":
        return np.stack([shape["all_points_x"], shape["all_points_y"]], axis=1).astype(float)
    raise ValueError(f"unsupported region shape {shape['name']!r}")


def load_annotations(path: Path, image_dir: Path) -> list[Template]:
    meta = json.loads(path.read_text())["_via_img_metadata"]
    out = []
    for m in meta.values():
        fields = {}
        for r in m["regions"]:
            name = r["region_attributes"]["field_name"]
            fields[name] = Field(
                name, r["region_attributes"].get("value", ""), _points(r["shape_attributes"])
            )
        out.append(Template(path.stem, image_dir / m["filename"], fields))
    return sorted(out, key=lambda t: t.image.name)


def load_templates(root: Path) -> list[Template]:
    """All templates under a MIDV-2020 `templates/` directory."""
    out = []
    for ann in sorted((root / "annotations").glob("*.json")):
        out.extend(load_annotations(ann, root / "images" / ann.stem))
    return out
