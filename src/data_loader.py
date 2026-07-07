from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from pathlib import Path

## Read all docs inside the directory
def load_all_documents(dir:str):
    """"Process all the pdfs inside the directory and return the list of documents"""
    all_documents=[]
    pdf_dir = Path(dir).resolve()
    print(f"[DEBUG] Data Path: {pdf_dir}")

    #Find all pdfs recursively inside the directory
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} pdf files in the directory {dir}")

    for pdf_file in pdf_files:
        print(f"[DEBUG] Processing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(pdf_file)
            documents = loader.load() # N documents for N pages in the pdf
            print(f"[DEBUG] Loaded: {len(documents)}")
            
            # Add source metadata to each document
            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"]   = "pdf"

            all_documents.extend(documents)
        except Exception as e:
            print(f"[ERROR] failed to load PDF {pdf_file}: {e}")
    print(f"[DEBUG] Processed {len(all_documents)} documents from {len(pdf_files)} pdf files")
    return all_documents