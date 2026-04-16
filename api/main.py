from __future__ import annotations

from secrets import compare_digest
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field

from data_pipeline.mock_loader import MockDataLoader
from trainer.unsloth_trainer import UnslothTrainer

app = FastAPI(title="OmniTune Engine", version="0.1.0")

# تخزين خفيف داخل الذاكرة لمرحلة التأسيس فقط؛ لاحقاً يُستبدل بقاعدة بيانات أو Redis.
# تنبيه: هذا النمط غير مناسب للإنتاج مع تعدد worker processes لأن كل عملية تملك ذاكرة مستقلة.
# تنبيه إضافي: جميع حالات المهام ستفقد عند إعادة تشغيل الخادم.
TASKS: dict[str, dict[str, Any]] = {}
TASKS_LOCK = Lock()


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

        with TASKS_LOCK:
            task_token = TASKS.get(task_id, {}).get("task_token")
            TASKS[task_id] = {"status": "completed", "result": result, "task_token": task_token}
    # نلتقط أي خطأ هنا لضمان تحويل فشل المهمة إلى حالة يمكن تتبعها عبر API بدلاً من فقدانها بصمت.
    except Exception as exc:  # pragma: no cover
        with TASKS_LOCK:
            task_token = TASKS.get(task_id, {}).get("task_token")
            TASKS[task_id] = {"status": "failed", "error": str(exc), "task_token": task_token}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/train")
def create_training_job(request: TrainingRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    task_id = str(uuid4())
    task_token = str(uuid4())
    with TASKS_LOCK:
        TASKS[task_id] = {"status": "queued", "task_token": task_token}
    background_tasks.add_task(_execute_training_task, task_id, request)

    return {
        "task_id": task_id,
        "task_token": task_token,
        "status": "queued",
        "message": "تم استلام المهمة وتشغيلها في الخلفية.",
    }


@app.get("/train/{task_id}")
def get_training_job(task_id: str, task_token: str) -> dict[str, Any]:
    with TASKS_LOCK:
        task_payload = TASKS.get(task_id)

    if not task_payload:
        return {"status": "not_found"}

    if not compare_digest(task_payload.get("task_token", ""), task_token):
        return {"status": "forbidden"}

    return {k: v for k, v in task_payload.items() if k != "task_token"}
