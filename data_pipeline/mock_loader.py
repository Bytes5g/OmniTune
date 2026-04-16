from __future__ import annotations

from typing import Any

from core.interfaces import BaseDataLoader


class MockDataLoader(BaseDataLoader):
    """محول تجريبي يوضح كيف نفصل مصدر البيانات عن صيغة التدريب."""

    def __init__(self, payload: list[dict[str, Any]]) -> None:
        self._payload = payload

    def load_raw(self) -> list[dict[str, Any]]:
        return self._payload

    def to_chatml(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # نستخدم ChatML كصيغة وسيطة لأنها متوافقة مع معظم خطوط تدريب المحادثة.
        chatml_dataset: list[dict[str, Any]] = []
        for item in records:
            system_prompt = item.get("system", "أنت مساعد عربي متخصص.")
            user_prompt = item.get("user") or item.get("instruction") or ""
            assistant_reply = item.get("assistant") or item.get("output") or ""

            chatml_dataset.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant_reply},
                    ]
                }
            )

        return chatml_dataset
