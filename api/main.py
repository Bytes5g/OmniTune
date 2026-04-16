from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field

from data_pipeline.mock_loader import MockDataLoader
from trainer.unsloth_trainer import UnslothTrainer

app = FastAPI(title="OmniTune Engine", version="0.1.0")

# تخزين خفيف داخل الذاكرة لمرحلة التأسيس فقط؛ لاحقاً يُستبدل بقاعدة بيانات أو Redis.
TASKS: dict[str, dict[str, Any]] = {}


class TrainingRequest(BaseModel):
    model_name: str = Field(default="Qwen/Qwen2.5-7B-Instruct")
    train_mode: Literal["lora", "qlora"] = Field(default="qlora")
    max_seq_length: int = Field(default=2048, ge=256, le=8192)
    dataset: list[dict[str, Any]] = Field(default_factory=list)


def _execute_training_task(task_id: str, payload: TrainingRequest) -> None:
    try:
        loader = MockDataLoader(payload.dataset)
        raw_records = loader.load_raw()
        chatml_dataset = loader.to_chatml(raw_records)

        trainer = UnslothTrainer()
        trainer.initialize(
            {
                "model_name": payload.model_name,
                "max_seq_length": payload.max_seq_length,
                "load_in_4bit": payload.train_mode == "qlora",
            }
        )
        result = trainer.run(chatml_dataset)

        TASKS[task_id] = {"status": "completed", "result": result}
    except Exception as exc:  # pragma: no cover
        TASKS[task_id] = {"status": "failed", "error": str(exc)}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def create_training_job(request: TrainingRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    task_id = str(uuid4())
    TASKS[task_id] = {"status": "queued"}
    background_tasks.add_task(_execute_training_task, task_id, request)

    return {
        "task_id": task_id,
        "status": "queued",
        "message": "تم استلام المهمة وتشغيلها في الخلفية.",
    }


@app.get("/train/{task_id}")
def get_training_job(task_id: str) -> dict[str, Any]:
    return TASKS.get(task_id, {"status": "not_found"})
