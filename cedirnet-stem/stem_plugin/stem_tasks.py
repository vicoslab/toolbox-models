"""Portable task contract for optional STEM particle and semantic branches.

The semantic branch instantiates the published FPN with class logits. The particle
branch retains its stock checkpoint layout; two independent trunks deliberately
avoid silently remapping localization weights into semantic logits.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    nanoparticles: bool = True
    segmentation: bool = False
    classes: tuple = ('Carbon', 'Film', 'Vacuum')
    ignore_index: int = 255

    def __post_init__(self):
        if type(self.nanoparticles) is not bool or type(self.segmentation) is not bool:
            raise ValueError('task flags must be boolean')
        if not (self.nanoparticles or self.segmentation):
            raise ValueError('enable at least one STEM task')
        object.__setattr__(self, 'classes', tuple(self.classes))
        if self.segmentation and (not 2 <= len(self.classes) <= 255 or
                len(set(self.classes)) != len(self.classes) or
                any(not isinstance(x, str) or not x.strip() for x in self.classes)):
            raise ValueError('semantic classes must be 2..255 unique nonempty names')
        if self.ignore_index != 255:
            raise ValueError('STEM mask ignore_index must be 255')

    def to_dict(self):
        return dict(nanoparticles=self.nanoparticles, segmentation=self.segmentation,
                    classes=list(self.classes), ignore_index=self.ignore_index)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def build_semantic_model(config, **kwargs):
    if not config.segmentation:
        return None
    from .semantic_model import build_semantic_fpn
    return build_semantic_fpn(len(config.classes), **kwargs)
