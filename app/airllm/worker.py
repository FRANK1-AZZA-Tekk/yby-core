"""Córtex Noturno AirLLM — worker assíncrono em lote.

Roda inferência de modelos grandes em modo streaming por camadas (AirLLM),
apenas durante o modo sono do YBY. Nunca use para chat interativo.
"""
import json
import logging
import os
import time
from pathlib import Path

from .config import AirLLMConfig
from .tasks import TaskQueue

logger = logging.getLogger("yby.airllm")


class AirLLMWorker:
    """Carrega o modelo em camadas e processa tarefas em lote."""

    def __init__(self, config: AirLLMConfig | None = None):
        self.config = config or AirLLMConfig.from_env()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        from airllm import AutoModel

        logger.info("Carregando AirLLM: %s", self.config.model)
        self._model = AutoModel.from_pretrained(
            self.config.model,
            compression=self.config.compression,
            delete_original=self.config.delete_original,
            profiling_mode=self.config.profiling,
            layer_shards_saving_path=self.config.shards_path,
        )
        return self._model

    def run_batch(self, tasks: list[dict], output_path: str | None = None) -> list[dict]:
        """Processa uma lista de tarefas em lote e grava resultados.

        Cada task deve conter: id, tipo e prompt.
        """
        if not tasks:
            return []
        model = self._load_model()
        prompts = [t["prompt"] for t in tasks]
        start = time.time()
        outputs = model.generate(
            prompts,
            max_new_tokens=self.config.max_new_tokens,
        )
        elapsed = time.time() - start
        results = []
        for task, output in zip(tasks, outputs):
            text = output if isinstance(output, str) else output[0]
            results.append({
                "id": task["id"],
                "tipo": task["tipo"],
                "resultado": text,
                "segundos": round(elapsed / max(len(tasks), 1), 2),
            })
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        logger.info("Lote concluído: %d tarefas em %.1fs", len(tasks), elapsed)
        return results

    def unload(self):
        """Libera VRAM e referências do modelo."""
        self._model = None
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("AirLLM descarregado da VRAM.")


def run_night_queue(queue: TaskQueue, output_path: str | None = None):
    """Ponto de entrada: processa a fila noturna inteira."""
    worker = AirLLMWorker()
    try:
        return worker.run_batch(queue.pending(), output_path=output_path)
    finally:
        worker.unload()
