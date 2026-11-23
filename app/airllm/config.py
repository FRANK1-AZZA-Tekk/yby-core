"""Configuração do Córtex Noturno AirLLM via variáveis de ambiente."""
import os
from dataclasses import dataclass


@dataclass
class AirLLMConfig:
    model: str
    shards_path: str
    compression: str
    delete_original: bool
    profiling: bool
    max_new_tokens: int

    @classmethod
    def from_env(cls) -> "AirLLMConfig":
        return cls(
            model=os.getenv("AIRLLM_MODEL", "Qwen/Qwen3-8B"),
            shards_path=os.getenv("AIRLLM_SHARDS_PATH", "./airllm_shards"),
            compression=os.getenv("AIRLLM_COMPRESSION", "none"),
            delete_original=os.getenv("AIRLLM_DELETE_ORIGINAL", "true").lower() == "true",
            profiling=os.getenv("AIRLLM_PROFILING", "true").lower() == "true",
            max_new_tokens=int(os.getenv("AIRLLM_MAX_NEW_TOKENS", "512")),
        )
