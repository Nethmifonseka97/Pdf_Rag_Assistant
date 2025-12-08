import streamlit as st
import faiss
import PyPDF2
import numpy as np
from fastembed import TextEmbedding


# -----------------------------
# Embedding model
# -----------------------------
embedder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------
# PDF text extraction
# -----------------------------
def extract_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# -----------------------------
# Chunk the PDF text
# -----------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# -----------------------------
# Build FAISS index
# -----------------------------
def build_faiss_index(chunks):
    # fastembed returns a generator, so we convert to list first
    vectors = list(embedder.embed(chunks))
    vectors = np.array(vectors, dtype="float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors


# -----------------------------
# Retrieve best-matching context
# -----------------------------
def retrieve_context(question, chunks, index, top_k=3):
    q_vec = list(embedder.embed([question]))
    q_vec = np.array(q_vec, dtype="float32")
    distances, indices = index.search(q_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((dist, chunks[idx]))
    return results


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Simple PDF Semantic Search (RAG Application)")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Reading and indexing your PDF..."):
        text = extract_pdf_text(uploaded_file)
        if not text.strip():
            st.error("Could not extract any text from this PDF.")
        else:
            chunks = chunk_text(text)
            index, _ = build_faiss_index(chunks)
            st.success(f"Indexed {len(chunks)} chunks from the PDF.")

            question = st.text_input("Ask something about the document:")

            if st.button("Search"):
                if not question.strip():
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Searching for relevant content..."):
                        results = retrieve_context(question, chunks, index, top_k=3)

                    st.subheader("Most Relevant Passages:")
                    for i, (dist, ctx) in enumerate(results, start=1):
                        st.markdown(f"**Result {i} (distance: {dist:.4f})**")
                        st.write(ctx)
                        st.markdown("---")