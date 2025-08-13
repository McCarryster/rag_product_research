# extract_text.py
import os
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """Извлекает текст из одного PDF файла."""
    text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_texts_from_folder(input_folder: str) -> dict:
    """
    Извлекает текст из всех PDF в папке.
    :param input_folder: Путь к папке с PDF
    :return: dict {имя_файла: текст}
    """
    results = {}

    for file in os.listdir(input_folder):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, file)
            print(f"[INFO] Обработка: {file}")
            text = extract_text_from_pdf(pdf_path)
            results[file] = text 

    return results

if __name__ == "__main__":
    # Пример запуска
    input_dir = "/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_product_research/data/pdfs"
    extracted_data = extract_texts_from_folder(input_dir)
    print(f"[INFO] Обработано {len(extracted_data)} PDF.")
    # Пример просмотра первого результата
    for fname, content in extracted_data.items():
        print(f"--- {fname} ---")
        print(content[:500], "...\n")
