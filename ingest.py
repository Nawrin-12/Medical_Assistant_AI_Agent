import PyPDF2
import os, pickle
import numpy as np
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from agent import llm_model
# from agent import compute_bertscore, compute_Rouge_L
from convert_to_pdf import pdf_generation
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import util
import time


from langchain.document_loaders import UnstructuredPDFLoader
from pathlib import Path

# Extracting Text from PDF
pdf_file = os.path.join('PDF_input','Clinical guideline.pdf')
extracted_pdf = "Extracted_pdf.txt"

# Path for storing Embedded data
faiss_index_path = "vectorDB/faiss.index"
Meta_path = "vectorDB/docs.pkl"


# # path for ground truth
# ground_truth_path = "vectorDB/gt.pkl"
# with open(ground_truth_path, "rb") as f:
#     QA_pairs = pickle.load(f)

# pdf_generation(pdf_file, ground_truth_path)
# print(f"PDF successfully generated at: {pdf_file}")
# print(f"Ground truth successfully stored at: {ground_truth_path}")

# def page_extract(path: str):
#     parser = DoclingPdfParser()
#     pdf_doc: PdfDocument = parser.load(path_or_stream=path)
#
#     with open(extracted_pdf, "w", encoding="utf-8") as file:
#        for _,page in pdf_doc.iterate_pages():
#            for word in page.iterate_cells(unit_type=TextCellUnit.WORD):
#                file.write(word.text+" ")
#            file.write("\n\n")


def page_extract(path: str):
    with open(path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pages = len(pdf_reader.pages)
        pdf_text = ""

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

#     for i, c in enumerate(chunks:
#         chunks.append({"text": c, "source": f"chunk_{i+1}"})
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

# # ground truth search for finding the closest question
# def closest_gt_search(query, qa_dict, threshold=0.7):
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#     embedded_query = model.encode(query, convert_to_tensor=True)
#     gt_queries = list(qa_dict.keys())
#     embedded_gt_queries = model.encode(gt_queries, convert_to_tensor=True)
#
#     similarity = util.cos_sim(embedded_query, embedded_gt_queries)[0]
#     top_indexes = int(np.argmax(similarity))
#     top_scores = float(similarity[top_indexes])
#
#     return gt_queries[top_indexes], top_scores

#
#
# # Main
# # Reading and extracting pdf file
# if os.path.exists(pdf_file):
#     page_extract(pdf_file)
# else:
#     print(f"PDF file not found at {pdf_file}")
# # splitting chunks
# chunks = split_text("Extracted_pdf.txt")
#
# # text embedding
# if chunks:
#     faiss_path, meta_file = text_embedding(chunks, faiss_index_path, Meta_path)
# else:
#     print("No chunks available to embed.")
#
#  # QA
# models =["llama3", "mistral"]
# # qa_dict= {qa["query"]["text"]:[answer["text"] for answer in qa["answers"]] for qa in QA_pairs }
# score_matrix=[]
#
# TEMPLATE= """You are a helpful medical AI assistant.
# Answer the following question using only the provided context with corresponding context.
# If the question is not relevant to context, say the question is not relevant to the given context"
#
# Conversation so far
# {chat_history}
#
# Context:
# {context}
#
# Question:
# {question}
# """
# while True:
#     query = input("\nEnter your query: ").strip()
#     if query.lower() == "exit":
#         memory.clear()
#         print("Goodbye !")
#         break
#     results, scores = search_faiss(query, faiss_index_path, chunks, top_query=3)
#     retrieved_text = "\n".join(results)
#
#     chat_history_text = memory.load_memory_variables({}).get("chat_history", "")
# #     print("\n--- Chat History ---")
# #     print(chat_history_text if chat_history_text else "EMPTY")
#     final_prompt = TEMPLATE.format(
#         chat_history=chat_history_text,
#         context=retrieved_text,
#         question=query
#     )
#
#
#     for model_name in models:
#         print(f"\n{model_name.upper()}")
#         start = time.time()
#         answer = llm_model(model_name, final_prompt, temperature=0.0)
#         time_lapse= time.time() - start
#         print(f"Response time: {time_lapse:.2f}s")
#         print("Answer:", answer)
#
#         memory.save_context({"input": query}, {"output": answer})


#
#         closest_gt, sim_score = closest_gt_search(query, qa_dict)
#         if sim_score>=0.6:
#             reference = qa_dict[closest_gt]
# #              Compute BERTScore
#             bert_scores = compute_bertscore(answer, reference)
#             print(f"BertScore -> Precision: {bert_scores['precision']:.4f},"
#                   f"Recall: {bert_scores['recall']:.4f},"
#                   f"F1: {bert_scores['f1']:.4f}")
#
#             # Compute ROUGH-L
#             rouge_l = compute_Rouge_L(answer, reference)
#             print(f"ROUGE-L F1: {rouge_l:.4f}")
#
#             score_matrix.append([model_name, query,
#                                  bert_scores['precision'],
#                                  bert_scores['recall'],
#                                  bert_scores['f1'],
#                                  rouge_l])
#         else:
#             print("skipping Evaluation")

#     df = pd.DataFrame(score_matrix, columns=["Model","Query","Precision","Recall","F1","ROUGE_L"])
#     print(df)
#
#     # Generating Heatmap
#     heatmap_data_bert = df.pivot(index="Query", columns="Model", values="F1")
#     plt.figure(figsize=(10,6))
#     sns.heatmap(heatmap_data_bert, annot=True, cmap="viridis", fmt=".2f")
#     plt.title("BERTScore (F1) Heatmap")
#     plt.ylabel("Query")
#     plt.xlabel("LLM Model")
#     plt.show()
#
#     heatmap_data_rouge = df.pivot(index="Query", columns="Model", values="ROUGE_L")
#     plt.figure(figsize=(10,6))
#     sns.heatmap(heatmap_data_rouge, annot=True, cmap="magma", fmt=".2f")
#     plt.title("ROUGE-L Heatmap")
#     plt.ylabel("Query")
#     plt.xlabel("LLM Model")
#     plt.show()




