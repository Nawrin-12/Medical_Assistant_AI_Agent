import ollama
import numpy as np
import os, pickle
from langchain.memory import ConversationBufferMemory
import faiss
from sentence_transformers import SentenceTransformer

current_dir = os.path.dirname(os.path.abspath(__file__))
vectorDB_dir = os.path.join(current_dir, "vectorDB")

faiss_path = os.path.join(vectorDB_dir, "prompt_index.faiss")
meta_path = os.path.join(vectorDB_dir, "prompt_text.pkl")
chat_history_path = os.path.join(vectorDB_dir, "chat_history.pkl")

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
dimension = embedding_model.get_sentence_embedding_dimension()

if os.path.exists(faiss_path) and os.path.exists(meta_path):
    prompt_faiss_index = faiss.read_index(faiss_path)
    with open(meta_path, "rb") as f:
        prompt_text_store = pickle.load(f)
else:
    prompt_faiss_index = faiss.IndexFlatL2(dimension)
    prompt_text_store = []
    faiss.write_index(prompt_faiss_index, faiss_path)
    with open(meta_path, "wb") as f:
        pickle.dump(prompt_text_store, f)

# Buffer memory for llm
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

if os.path.exists(chat_history_path):
    try:
        with open(chat_history_path, "rb") as f:
            chat_history_store = pickle.load(f)
    except Exception as e:
        chat_history_store = []
else:
    chat_history_store = []

if chat_history_store:
    try:
        memory.clear()
        for msg in chat_history_store:
            if msg["role"] == "user":
                memory.chat_memory.add_user_message(msg["content"])
            elif msg["role"] in ["assistant", "ai"]:
                memory.chat_memory.add_ai_message(msg["content"])
    except Exception as e:
        print(f"ERROR initializing memory: {e}")

TEMPLATE = """You are a helpful medical AI assistant.
Answer the following question using only the provided context with corresponding context.
If the question is not relevant to context, say the question is not relevant to the given context"

Conversation so far:
{chat_history}

Context:
{context}

Question:
{question}
"""

# def llm_model(model_name, prompt, temperature):
#     try:
#         response = ollama.chat(
#         model= model_name,
#         messages=[{"role":"user", "content": prompt}]
#         )
#         return response['message']['content']
#     except Exception as e:
#         return f"Error querying {model_name}: {str(e)}"

def llm_model(model_name, prompt, temperature):
    try:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        response = ollama.chat(
            model=model_name,
            messages=messages
        )
        return response['message']['content']
    except Exception as e:
        return f"Error querying {model_name}: {str(e)}"

def learn_prompt(text: str):
    global prompt_text_store, prompt_faiss_index
    if text in prompt_text_store:
        return

    embedding = embedding_model.encode([text])
    prompt_faiss_index.add(np.array(embedding, dtype="float32"))
    prompt_text_store.append(text)

    faiss.write_index(prompt_faiss_index, faiss_path)
    try:
        with open(meta_path, "wb") as f:
            pickle.dump(prompt_text_store, f)
    except Exception as e:
        print(f"Error saving prompt_text.pkl: {e}")

def prompt_context(query: str, k=3):
    if len(prompt_text_store) == 0:
        return ""

    embedding = embedding_model.encode([query])
    embedding = np.array(embedding, dtype="float32")
    D, I = prompt_faiss_index.search(embedding, min(k, len(prompt_text_store)))

    retrieved = []
    for i in I[0]:
        if i < len(prompt_text_store):
            retrieved.append(prompt_text_store[i])
    return "\n".join(retrieved)

def save_chat_history():
    try:
        with open(chat_history_path, "wb") as f:
            pickle.dump(chat_history_store, f)
        print("Chat history saved successfully")
    except Exception as e:
        print(f"Error saving chat history: {e}")

def shot_prompting_messages(question, pdf_context, retrieved_prompt_context):
    messages =[]
    messages.append({
            "role": "system",
            "content": """You are a helpful medical AI assistant. Answer questions using only the provided
    context. If the question is not relevant to context, say the question is not relevant to the given context."""
    })

    messages.append({
            "role": "user",
            "content": """A 50-year-old male patient with a history of diabetes presents with
    persistent anxiety. He reports constant worry about his health, irritability, difficulty
    sleeping with frequent nightmares, and avoidance of social activities. He also complains
    of a dry mouth, hot flashes, and something like a lump in the throat, though no physical
    illness explains these symptoms. What is the likely diagnosis, and how should it be managed?"""
    })

    messages.append({
            "role": "assistant",
            "content": "The patient’s symptoms are most consistent with anxiety disorder.\n\n"
                "Management:\n"
                "- Reassure and identify triggers (stress, depression, PTSD).\n"
                "- Teach relaxation and sleep hygiene.\n\n"
                "Medicines:\n"
                "- Acute severe anxiety: Diazepam 5–10 mg PO (or 10 mg IM), repeat after 1 hour if needed.\n"
                "- Short course (max 2–3 weeks): Diazepam 2.5–5 mg PO twice daily, taper in last days.\n"
                "- Moderate anxiety (>2 weeks): Hydroxyzine 25–50 mg PO twice daily (max 100 mg/day).\n"
                "- If no improvement after 1 week: Diazepam 2.5–5 mg PO twice daily (max 2 weeks).\n"
                "- Generalized anxiety (>2 months): Fluoxetine or Paroxetine 20 mg PO once daily "
                "(continue 2–3 months after symptoms resolve, then taper)."
    })

    context_parts = []
    if pdf_context.strip():
        context_parts.append(f"Medical Context from Documents:\n{pdf_context}")
    if retrieved_prompt_context.strip():
        context_parts.append(f"Additional Medical Context:\n{retrieved_prompt_context}")

    if context_parts:
        full_context = "\n\n".join(context_parts)
        user_content = f"Context:\n{full_context}\n\nQuestion: {question}"
    else:
        user_content = f"Question: {question}\n\nNote: No specific medical context was found for this question."

    messages.append({"role": "user", "content": user_content})
    return messages

def agent_response(model_name, question, pdf_retriever, temperature=0.3):
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
#     prompt = TEMPLATE.format(
#         chat_history = chat_history_str,
#         context = context,
#         question = question
#     )


    response = llm_model(model_name, prompt, temperature)

    memory.save_context({"input": question}, {"output": response})
    chat_history_store.append({"role": "user", "content": question})
    chat_history_store.append({"role": "assistant", "content": response})

    save_chat_history()

    return response

def get_chat_history():
    return chat_history_store

def clear_chat_history():
    global chat_history_store
    chat_history_store = []
    memory.clear()
    try:
        if os.path.exists(chat_history_path):
            os.remove(chat_history_path)
            print("Chat history deleted successfully")
    except Exception as e:
        print(f"Error clearing chat history: {e}")

# def compute_bertscore(prediction, reference, lang ="en"):
#     p, r, F1 = bert_score([prediction], [reference[0]], lang=lang, model_type="microsoft/deberta-xlarge-mnli")
#     return{
#         "precision": p.mean().item(),
#         "recall": r.mean().item(),
#         "f1": F1.mean().item()
#     }
#
# def compute_Rouge_L(prediction, reference):
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     scores = [scorer.score(ref, prediction)['rougeL'].fmeasure for ref in reference]
#     return max(scores)




