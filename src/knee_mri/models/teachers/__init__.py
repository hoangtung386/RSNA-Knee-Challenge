"""Hai teacher của pipeline: 3DINO-ViT cho ảnh, VLM cho tri thức từ report."""

from knee_mri.models.teachers.dino3d import Dino3DTeacher, DinoUnavailableError
from knee_mri.models.teachers.gemma_guidance import GemmaGuidanceEncoder

__all__ = ["Dino3DTeacher", "DinoUnavailableError", "GemmaGuidanceEncoder"]
