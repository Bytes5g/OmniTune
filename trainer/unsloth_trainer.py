from __future__ import annotations

from typing import Any

from core.interfaces import BaseModelTrainer


class UnslothTrainer(BaseModelTrainer):
    """تنفيذ أولي لمحرك تدريب يعتمد Unsloth ومهيأ لأجهزة 24GB VRAM."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}

    def initialize(self, config: dict[str, Any]) -> dict[str, Any]:
        # اختيار QLoRA افتراضياً يقلل استهلاك الذاكرة ويخدم قيود 24GB بشكل عملي.
        self._config = {
            "model_name": config.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
            "max_seq_length": config.get("max_seq_length", 2048),
            "load_in_4bit": config.get("load_in_4bit", True),
            "r": config.get("r", 16),
            "lora_alpha": config.get("lora_alpha", 32),
            "lora_dropout": config.get("lora_dropout", 0.05),
            "target_modules": config.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            ),
            "bias": config.get("bias", "none"),
        }

        return {
            "trainer": "unsloth",
            "status": "initialized",
            "strategy": "qlora" if self._config["load_in_4bit"] else "lora",
            "config": self._config,
        }

    def run(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        # هذه المرحلة تأسيسية؛ نرجع نتيجة معيارية دون بدء تدريب فعلي ثقيل.
        if not self._config:
            raise RuntimeError("يجب استدعاء initialize قبل run")

        return {
            "trainer": "unsloth",
            "status": "queued_for_training",
            "samples": len(dataset),
            "model_name": self._config.get("model_name"),
        }
