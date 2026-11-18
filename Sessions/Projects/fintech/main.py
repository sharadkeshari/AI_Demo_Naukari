import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from app import run_chat_turn, classify_query, run_general_turn, run_policy_run
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input ?? query --> messages --> return --> FE
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # None = new session

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    query_type: str
    tools_called: list[str] = []


@app.post('/chat',response_model=ChatResponse)
def chat(req:ChatRequest):
    session_id = str(uuid.uuid4())
    query_type = classify_query(req.message)

    if query_type == "tool":
        reply, tool_called = run_chat_turn(req.message,req.session_id)
        return ChatResponse(reply=reply, session_id=session_id, tools_called=tool_called,
                            query_type=query_type)
    elif query_type == "policy":
        reply = run_policy_run(req.message,req.session_id)
        return ChatResponse(reply=reply, session_id=session_id, query_type=query_type)
    
    else:
        reply = run_general_turn(req.message,req.session_id)
        return ChatResponse(reply=reply, session_id=session_id, query_type=query_type)

    
    # reply, tool_called = run_chat_turn(req.message,req.session_id)
    # print('tool used: ',tool_called)
    # return ChatResponse(reply=reply, session_id=session_id, tools_called=tool_called)
 

@app.get('/',response_class=HTMLResponse)
def ui():
    with open("home.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
