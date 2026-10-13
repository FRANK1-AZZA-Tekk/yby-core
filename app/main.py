import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.graph import yby_graph


app = FastAPI(title="YBY Core", version="0.2.0")


class ChatRequest(BaseModel):
    message: str
    project_path: str = "projects/yby-seed"


class ChatResponse(BaseModel):
    answer: str
    task_type: str
    needs_approval: bool


@app.get("/health")
async def health():
    return {"status": "ok", "service": "yby-core"}


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


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
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

    async def event_generator():
        yield sse_event(
            "start",
            {
                "message": request.message,
                "project_path": request.project_path,
            },
        )

        final_answer = ""

        try:
            async for chunk, metadata in yby_graph.astream(
                initial_state,
                stream_mode="messages",
            ):
                content = getattr(chunk, "content", "")

                if not content:
                    continue

                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    )

                if not content:
                    continue

                final_answer += content

                node = metadata.get("langgraph_node", "unknown")

                yield sse_event(
                    "token",
                    {
                        "node": node,
                        "content": content,
                    },
                )

            yield sse_event(
                "done",
                {
                    "answer": final_answer,
                },
            )

        except Exception as error:
            yield sse_event(
                "error",
                {
                    "message": str(error),
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
