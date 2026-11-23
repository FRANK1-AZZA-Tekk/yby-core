"""Gatilho de modo sono: descarrega Ollama e inicia o Córtex Noturno."""
import logging
import os
import requests

from .tasks import TaskQueue
from .worker import run_night_queue

logger = logging.getLogger("yby.airllm.trigger")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def unload_ollama():
    """Libera VRAM do Ollama antes de iniciar o AirLLM."""
    try:
        models = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
        for m in models.get("models", []):
            requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": m["name"], "keep_alive": 0},
                timeout=10,
            )
        logger.info("Ollama descarregado da VRAM.")
    except Exception as exc:
        logger.warning("Falha ao descarregar Ollama: %s", exc)


def on_sleep_mode(queue: TaskQueue | None = None, output_path: str | None = None):
    """Chamado quando o YBY detecta modo sono. Roda o lote noturno."""
    queue = queue or TaskQueue()
    unload_ollama()
    results = run_night_queue(queue, output_path=output_path)
    queue.clear()
    return results
