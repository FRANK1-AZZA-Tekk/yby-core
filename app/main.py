from fastapi import FastAPI
from pydantic import BaseModel

from app.graph import yby_graph


app = FastAPI(title="YBY Core", version="0.1.0")


class ChatRequest(BaseModel):
    message: str
    project_path: str = "projects/yby-seed"


class ChatResponse(BaseModel):
    answer: str
    task_type: str
    needs_approval: bool


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    initial_state = {
        "messages": [
            {
                "role": "user",
                "content": request.message,
            }
        ],
        "task_type": "chat",
        "project_path": request.project_path,
        "needs_approval": False,
        "result": "",
    }

    final_state = await yby_graph.ainvoke(initial_state)

    return ChatResponse(
        answer=final_state["result"],
        task_type=final_state["task_type"],
        needs_approval=final_state["needs_approval"],
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "yby-core"}
