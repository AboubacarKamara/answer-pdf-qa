import fitz  # PyMuPDF
import re

def split_text_by_paragraph(text):
    # Découpe les paragraphes en utilisant les doubles sauts de ligne ou les lignes vides
    raw_paragraphs = re.split(r'\n\s*\n', text)
    # Nettoie les retours à la ligne internes et espaces inutiles
    paragraphs = [p.strip().replace("\n", " ") for p in raw_paragraphs if p.strip()]
    return paragraphs

def group_small_paragraphs(paragraphs, max_chars=1000):
    grouped = []
    current = ""

    for para in paragraphs:
        # Ajoute les paragraphes tant que la longueur totale ne dépasse pas le seuil
        if len(current) + len(para) + 1 <= max_chars:
            current += para + " "
        else:
            if current:
                grouped.append(current.strip())
            current = para + " "
    if current:
        grouped.append(current.strip())
    return grouped

def load_and_split_pdf(file_path, max_chars=1000):
    doc = fitz.open(file_path)
    chunks = []
    global_index = 0

    for page_num, page in enumerate(doc):
        text = page.get_text()
        paragraphs = split_text_by_paragraph(text)
        smart_paragraphs = group_small_paragraphs(paragraphs, max_chars=max_chars)

        for para in smart_paragraphs:
            chunks.append({
                "content": para,
                "page": page_num + 1,
                "index": global_index  # index global unique
            })
            global_index += 1

    return chunks

