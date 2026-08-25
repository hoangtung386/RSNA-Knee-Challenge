"""Các thành phần model: student, teacher, SAM và bộ sinh báo cáo."""

from knee_mri.models.heads import (
    ClassifierHead,
    ContrastiveProjector,
    KDProjector,
    MetadataEncoder,
    make_metadata_vector,
)
from knee_mri.models.report_generator import GemmaReportGenerator, rule_based_report
from knee_mri.models.sam_masker import SamMaskGenerator
from knee_mri.models.student import ViT3DStudent
from knee_mri.models.teachers import Dino3DTeacher, GemmaGuidanceEncoder

__all__ = [
    "ClassifierHead",
    "ContrastiveProjector",
    "Dino3DTeacher",
    "GemmaGuidanceEncoder",
    "GemmaReportGenerator",
    "KDProjector",
    "MetadataEncoder",
    "SamMaskGenerator",
    "ViT3DStudent",
    "make_metadata_vector",
    "rule_based_report",
]
