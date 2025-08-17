import PyPDF2
import os, pickle
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
#LLM Models
from agent import llm_model
import time

# Extracting Text from PDF
pdf_file = os.path.join('PDF_input','gigworkeraccountability.pdf')
extracted_pdf = "Extracted_pdf.txt"

# Path for storing Embedded data
faiss_index_path = "vectorDB/faiss.index"
Meta_path = "vectorDB/docs.pkl"

def page_extract(path: str):
    with open(path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pages = len(pdf_reader.pages)
        pdf_text = ""

        # iterate through each page of the pdf
        for i in range(0, pages):
            page = pdf_reader.pages[i]
            pdf_text += page.extract_text()
            pdf_text += "\n"

        with open(extracted_pdf, "w", encoding="utf-8") as file:
            file.write(pdf_text)

# Splitting the texts into smaller chunks for convenience
def split_text(file_path):
    chunks=[]
    max_Chunk_Size= 500
    overlap= 100
    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=max_Chunk_Size,
                                          chunk_overlap= overlap,
                                          length_function = len)
    with open(file_path, "r", encoding="utf-8") as file:
        file_text = file.read()
    chunks = text_splitter.split_text(file_text)
    return chunks

# Embed the text and store embedded data
def text_embedding (chunks, faiss_index_path, meta_path):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, normalize_embeddings = True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings))

    # saving the faiss index
    faiss.write_index(index, faiss_index_path)

    #saving the meta data
    with open(meta_path, "wb") as file:
        pickle.dump(chunks, file)
    return faiss_index_path,meta_path

# re-ranking result for retrieving top relevant contexts
def rerank_chunks(chunks, query, top_query = None):
    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    embedded_query = model.encode([query],normalize_embeddings = True)
    embedded_chunk = model.encode(chunks, normalize_embeddings = True)

    # Computing cosine similarities here
    similarities = cosine_similarity(embedded_query, embedded_chunk)[0]
    ranked_query = np.argsort(similarities)[::-1]

    if top_query is not None:
        ranked_query = ranked_query[:top_query]

    ranked_chunks = [chunks[i] for i in ranked_query]
    scores = [similarities[i] for i in ranked_query]

    return ranked_chunks, scores

# faiss search for retrieving the relevant chunks
def search_faiss(query, faiss_index_path, chunks, top_query):
    index = faiss.read_index(faiss_index_path)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedded_query = model.encode([query], normalize_embeddings=True)
    dist, indices = index.search(np.array(embedded_query),top_query)

    retrieve_chunks = [chunks[i] for i in indices[0]]
    reranked_chunks, scores = rerank_chunks(retrieve_chunks, query, top_query)
    return reranked_chunks, scores


# Main
# Reading and extracting pdf file
if os.path.exists(pdf_file):
    page_extract(pdf_file)
else:
    print(f"PDF file not found at {pdf_file}")
# splitting chunks
chunks = split_text("Extracted_pdf.txt")

# text embedding
if chunks:
    faiss_path, meta_file = text_embedding(chunks, faiss_index_path, Meta_path)
    print(f" FAISS index saved at: {faiss_path}")
else:
    print("No chunks available to embed.")

 # QA
models =["llama3", "mistral", "qwen"]
while True:
    query = input("\nEnter your query: ").strip()
    if query.lower() == "exit":
        print("Goodbye !")
        break
    results, scores = search_faiss(query, faiss_index_path, chunks, top_query=3)
    retrieved_text = "\n".join(results)

    final_prompt = f"""You are a helpful AI assistant.
    Answer the following question using only the provided context.
    If the answer cannot be found in the context, say "The answer
    is nowhere to be found in the provided PDF."

    Context:
    {retrieved_text}

    Question:
    {query}
    """

    for model_name in models:
        print(f"\n{model_name.upper()}")
        start = time.time()
        answer = llm_model(model_name, final_prompt)
        time_lapse= time.time() - start
        print(f"Response time: {time_lapse:.2f}s")
        print("Answer:", answer)
#     answer = llm_model(selected_model, final_prompt)
#     print("Answer:", answer)
