# PDF Semantic Search Assistant
A lightweight, fully local AI-powered tool for semantic search inside PDF documents.  
Upload a PDF → ask a question → the system finds the most relevant sections using modern embedding and vector search techniques.

This project demonstrates an end-to-end Retrieval system (the “R” in RAG) without requiring large language model inference or GPU resources.

## Features
- Semantic search
- PDF text extraction
- Automatic text chunking
- FastEmbed embeddings (no Torch required)
- FAISS vector search
- Streamlit UI



## How It Works
1. Upload a PDF  
2. Text is chunked into ~300-word segments  
3. Each chunk is embedded using FastEmbed  
4. FAISS stores & searches embeddings  
5. User asks a question  
6. System returns the top‑k most relevant text chunks  


## Installation
```bash
git clone https://github.com/<your-username>/pdf-semantic-search-assistant.git
cd pdf-semantic-search-assistant
python3 -m venv venv
source venv/bin/activate
pip install "numpy<2.0.0" streamlit faiss-cpu PyPDF2 fastembed
```

## Usage
```bash
python -m streamlit run app.py
```

## Folder Structure
```
project/
│
├── app.py
├── README.md
└── venv/
```

