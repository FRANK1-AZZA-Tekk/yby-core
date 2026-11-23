"""Fila de tarefas noturnas do Córtex AirLLM."""
import json
from pathlib import Path


class TaskQueue:
    """Fila simples baseada em arquivo JSON, processada apenas em lote."""

    def __init__(self, path: str = "./airllm_queue.json"):
        self.path = Path(path)

    def pending(self) -> list[dict]:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def add(self, task_id: str, tipo: str, prompt: str):
        tasks = self.pending()
        tasks.append({"id": task_id, "tipo": tipo, "prompt": prompt})
        self.path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear(self):
        if self.path.exists():
            self.path.unlink()
