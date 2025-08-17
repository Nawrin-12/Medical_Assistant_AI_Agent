import ollama

def llm_model(model_name, prompt):
    try:
        response = ollama.chat(
        model= model_name,
        messages=[{"role":"user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error querying {model_name}: {str(e)}"


