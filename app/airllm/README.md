# Córtex Noturno AirLLM

Módulo do YBY que roda inferência de modelos grandes (70B+) em GPUs de 4–12 GB
via streaming por camadas (AirLLM), exclusivamente em modo sono (lote assíncrono).

## Uso

1. Configure as variáveis de ambiente:
   - `AIRLLM_MODEL` — ex.: `Qwen/Qwen3-8B`
   - `AIRLLM_SHARDS_PATH` — pasta NVMe para shards
   - `AIRLLM_COMPRESSION` — `none`, `4bit` ou `8bit`
   - `AIRLLM_DELETE_ORIGINAL` — `true` para não duplicar checkpoint
   - `AIRLLM_PROFILING` — `true` para medir tokens/segundo

2. Adicione tarefas à fila:

```python
from app.airllm.tasks import TaskQueue
q = TaskQueue()
q.add("resumo-2026-11-23", "resumo_dia", "Resuma os eventos de hoje...")
```

3. Dispare o modo sono:

```python
from app.airllm.trigger import on_sleep_mode
on_sleep_mode(output_path="./output/night_briefing.json")
```

## Limites

- Latência: segundos a minutos por token (não usar para chat interativo).
- Requisitos: NVMe rápido, RAM suficiente no host, GPU com VRAM livre.
- O Ollama deve estar descarregado antes de iniciar (o trigger faz isso).
