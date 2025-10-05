# Dataset
from datasets import load_dataset
from fpdf import FPDF
import os
import pickle

def pdf_generation(pdf_file, ground_truth_path):
    # Loading data
    ds = load_dataset("deepmind/narrativeqa", split="train")
    summary_text = ds[0] ["document"]
#     print(type(summary_text))
#     print(summary_text.keys())

    story_text = summary_text['text']

    # Converting the story to PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for line in story_text.split('\n'):
        pdf.multi_cell(0, 10, line)

    pdf.output(pdf_file)
    print(f"PDF saved at: {pdf_file}")

    # Storing Ques and ground truth
    QA =[]
    for val in ds:
        if val["document"]["id"] == summary_text["id"]:
            QA.append({"query": val["question"], "answers": val["answers"]})

    with open(ground_truth_path, "wb") as f:
        pickle.dump(QA, f)

    return pdf_file, ground_truth_path