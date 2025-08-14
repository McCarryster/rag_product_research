import os
import fitz

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a single PDF file."""
    text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_texts_from_folder(input_folder: str) -> dict:
    """
    Extracts text from all PDFs in a folder.
    :param input_folder: Path to the folder with PDFs
    :return: dict {filename: text}
    """
    results = {}

    for file in os.listdir(input_folder):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, file)
            print(f"[INFO] Processing: {file}")
            text = extract_text_from_pdf(pdf_path)
            results[file] = text 

    return results
