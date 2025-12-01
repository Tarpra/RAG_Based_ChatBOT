# RAG Chatbot – Chat with Your Documents

This is a Streamlit-based RAG (Retrieval-Augmented Generation) chatbot built using **LangChain** and **OpenAI**.  
You can upload a **PDF** or **TXT** document and then ask questions based on its content. The app splits the document into chunks, creates embeddings using OpenAI, stores them in a **FAISS** vector store, and uses a chat model to answer your questions.

## Features

- Upload **PDF** or **TXT** files
- Preview the extracted document text
- Chunking of text using `RecursiveCharacterTextSplitter`
- Embeddings with `OpenAIEmbeddings`
- Vector search with **FAISS**
- Chat-style Q&A with memory (chat history)
- Streamlit UI

## Tech Stack

- Python
- Streamlit
- LangChain (langchain-core, langchain-community, langchain-openai, langchain-text-splitters)
- FAISS
- OpenAI
- PyPDF2
- python-dotenv

## Project Structure

```text
.
├── RAGChatBOT.py        # Main Streamlit app
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .env.example         # Example environment variables (no real keys)


## 🔧 Installation & Setup

1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

2️⃣ Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate     # macOS / Linux
# venv\Scripts\activate      # Windows

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Add your API key

Copy .env.example → .env:

cp .env.example .env            # macOS / Linux
# copy .env.example .env        # Windows

Open .env and add:

OPENAI_API_KEY=your_openai_api_key_here

Run the Streamlit application:
streamlit run RAGChatBOT.py


**💬 How It Works**

Upload a PDF or TXT file

The app extracts and splits the text

Embeddings are created using OpenAIEmbeddings

A FAISS vector store indexes the documents

Your query is compared against the vector store

The retrieved chunks + your question → Sent to an OpenAI chat model

You get an accurate, document-aware response

