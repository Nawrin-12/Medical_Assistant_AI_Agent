from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from ingest import page_extract, split_text, text_embedding, search_faiss
from agent import agent_response, get_chat_history, clear_chat_history
from agent import learn_prompt, prompt_context, llm_model, shot_prompting_messages, TEMPLATE, memory, chat_history_store, save_chat_history
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
import os, pickle
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from langchain.schema import Document
import time
from typing import List

from auth import authenticate_user, create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES

# Extracting Text from PDF
pdf_file = os.path.join('PDF_input','Clinical guideline.pdf')
extracted_pdf = "Extracted_pdf.txt"

# Path for storing Embedded data
faiss_index_path = "vectorDB/faiss.index"
Meta_path = "vectorDB/docs.pkl"

app = FastAPI()
class Query(BaseModel):
    question: str
    models: List[str] =["qwen","mistral"]
    temperature: float = 0.0

# chunks = []

class PDFRetriever:
    def __init__(self):
        self.faiss_index_path = faiss_index_path
        self.meta_path = Meta_path

    def get_relevant_documents(self, query):
        try:
            with open(self.meta_path, "rb") as f:
                loaded_chunks = pickle.load(f)

            top_query = min(3, len(loaded_chunks))
            results, scores = search_faiss(query, self.faiss_index_path, loaded_chunks, top_query)

            documents = []
            for text in results:
                documents.append(Document(page_content=text))

            print(f"PDF Retriever found {len(documents)} relevant documents")
            return documents

        except Exception as e:
            print(f"Error in PDF retriever: {e}")
            return []

pdf_retriever = PDFRetriever()
AVAILABLE_MODELS = ["qwen", "mistral"]

def unified_query_handler(question: str, models: List[str], temperature: float = 0.0):
    start_time = time.time()
    valid_models = [model for model in models if model in AVAILABLE_MODELS]

    if not valid_models:
        valid_models = ["qwen","mistral"]

    if len(valid_models) == 1:
        print(f"Single model mode: {valid_models[0]}")
        response = agent_response(
            model_name=valid_models[0],
            question=question,
            pdf_retriever=pdf_retriever,
            temperature=temperature
        )
        return {
            "answers": {valid_models[0]: response},
            "models_used": valid_models,
            "total_time": round(time.time() - start_time, 2),
            "mode": "single_model"
        }

    # Multi-model case
    print(f"Multi-model mode: {valid_models}")
    learn_prompt(question)
    docs = pdf_retriever.get_relevant_documents(question)
    pdf_context = ".\n".join([i.page_content for i in docs]) if docs else ""
    retrieved_prompt_context = prompt_context(question)

    messages = shot_prompting_messages(question, pdf_context, retrieved_prompt_context)

#     chat_history_data = memory.load_memory_variables({})
#     chat_history = chat_history_data.get("chat_history", [])
#
#     if isinstance(chat_history, list):
#         chat_history_str = "\n".join([msg.content if hasattr(msg, 'content') else str(msg) for msg in chat_history])
#     else:
#         chat_history_str = str(chat_history)
#
#     context = pdf_context + "\n\n" + retrieved_prompt_context
#     base_prompt = TEMPLATE.format(
#         chat_history=chat_history_str,
#         context=context,
#         question=question
#     )

    context_time = time.time() - start_time
    print(f"Total time: {context_time:.2f}s")

    responses = {}
    model_times = {}

    for model in valid_models:
        try:
            print(f"🤖 Querying model: {model}")
            model_start = time.time()
            response = llm_model(model, messages, temperature)
            responses[model] = response
            learn_prompt(response)
            model_time = time.time() - model_start
            model_times[model] = round(model_time, 2)
            print(f" {model} completed in {model_time:.2f}s")
        except Exception as e:
            print(f"Error with model {model}: {e}")
            responses[model] = f"Error: {str(e)}"
            model_times[model] = 0

    chat_history_store.append({"role": "user", "content": question})
    combined_response = "Multi-model responses:\n\n"
    for model, response_text in responses.items():
        combined_response += f"--- {model.upper()} ---\n{response_text}\n\n"

    chat_history_store.append({"role": "assistant", "content": combined_response})
    memory.save_context({"input": question}, {"output": combined_response})
    save_chat_history()

    total_time = time.time() - start_time
    print(f"All models completed in {total_time:.2f}s")

    return {
        "answers": responses,
        "models_used": valid_models,
        "model_times": model_times,
        "total_time": round(total_time, 2),
        "mode": "multi_model"
    }

# API endpoints
@app.get("/")
def root():
    return {
        "message": "Server running!",
        "available_models": AVAILABLE_MODELS
        }

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail= "Incorrect user or password")
    access_token_expires = timedelta(minutes = ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(data={"sub":user["username"]}, expires_delta=access_token_expires)
    return {"access_token":token, "token_type": "bearer"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global chunks
    if file is None:
        return {"message":"No file uploaded"}
    path = f"PDF_input/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    page_extract(path)
    chunks = split_text("Extracted_pdf.txt")
    text_embedding(chunks, "vectorDB/faiss.index", "vectorDB/docs.pkl")
    return {"message":f"{file.filename} uploaded Successfully."}

@app.post("/query")
def ask_question(query: Query):
    try:
        start_time = time.time()
        print(f"Received query: {query.question}")

        result = unified_query_handler(
            question=query.question,
            models=query.models,
            temperature=query.temperature
        )

        print(f"Query completed in {result['total_time']}s")
        return result

    except Exception as e:
        print("Error in query:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history")
def get_chat_history_endpoint():
    """Get the complete chat history"""
    try:
        history = get_chat_history()
        return {
            "history": history
        }
    except Exception as e:
        print("Error in /chat-history:", e)
        raise HTTPException(status_code=500, detail="Failed to load chat history")

@app.delete("/clear-history")
def clear_history_endpoint():
    try:
        clear_chat_history()
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        print("Error in /clear-history:", e)
        raise HTTPException(status_code=500, detail="Failed to clear chat history")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)