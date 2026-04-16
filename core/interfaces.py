from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDataLoader(ABC):
    """واجهة مجردة لقراءة البيانات وتحويلها لصيغة التدريب."""

    @abstractmethod
    def load_raw(self) -> list[dict[str, Any]]:
        """تحميل البيانات الخام من المصدر."""

    @abstractmethod
    def to_chatml(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """تحويل السجلات الخام إلى صيغة ChatML موحدة."""


class BaseModelTrainer(ABC):
    """واجهة مجردة لأي محرك تدريب حتى يبقى اللب معزولاً عن أداة التنفيذ."""

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> dict[str, Any]:
        """تهيئة إعدادات التدريب (مثل LoRA/QLoRA) وإرجاع ملخص التنفيذ."""

    @abstractmethod
    def run(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """تشغيل عملية التدريب وإرجاع نتيجة معيارية."""
