import fitz  # PyMuPDF

def load_and_split_pdf(file_path, max_chars=1000):
    doc = fitz.open(file_path)
    chunks = []
    global_index = 0
    for page_num, page in enumerate(doc):
        text = page.get_text()
        for i in range(0, len(text), max_chars):
            chunk = text[i:i+max_chars]
            chunks.append({
                "content": chunk,
                "page": page_num + 1,
                "index": global_index  # index global unique
            })
            global_index += 1
    return chunks
