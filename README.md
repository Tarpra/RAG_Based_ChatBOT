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

**Setup & Installation**

Clone the repository

git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>


Create and activate a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate   # On macOS/Linux
# venv\Scripts\activate    # On Windows


**Install dependencies
**
pip install -r requirements.txt


Set up environment variables

Copy .env.example to .env:

cp .env.example .env      # macOS/Linux
# copy .env.example .env  # Windows PowerShell / CMD


Open .env and put your actual OpenAI API key:

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx

Run the App Locally

From the project directory, run:

streamlit run RAGChatBOT.py


This will start the Streamlit server and open the app in your browser (usually at http://localhost:8501).

Usage

Upload a PDF or TXT file using the uploader.

Wait for the app to:

Extract the text

Split the text into chunks

Create embeddings

Build the FAISS vector store

Type your question in the input box.

The model will retrieve relevant chunks from the document and answer your question.

Previous Q&A pairs are kept in the chat history section.

**Notes**

The app reads your OpenAI API key from the .env file using python-dotenv.

Make sure never to commit your real .env file or your API key to GitHub.

You can change the underlying model (e.g., gpt-4o-mini) inside RAGChatBOT.py if needed.


---

## 5️⃣ How to upload this to GitHub (UI steps)

Once those files are ready in your local `rag-chatbot` folder:

1. Go to **github.com → New repository**.
2. Enter a **Repository name**, e.g. `rag-chatbot`.
3. Keep it **Public** (or Private if you prefer).
4. You can skip initializing with a README (since you already have one) or let GitHub create it and overwrite later.
5. Click **Create repository**.

### Upload the files

On the new repo page:

1. Click **“Upload files”**.
2. Drag and drop these files from your folder:
   - `RAGChatBOT.py`
   - `requirements.txt`
   - `README.md`
   - `.env.example`
   - (Optional) `.gitignore`
3. Scroll down and click **“Commit changes”**.

If GitHub had created a placeholder file (like `.gitkeep` or its own README):

- Open that unwanted file in GitHub.
- Click the **trash bin / Delete this file** icon.
- Add a short commit message like `Remove placeholder file` and confirm.

---

If you show me a screenshot of your repo after this (like you did for the previous project), I can double-check and then help you with **next step: deploying this Streamlit RAG app (e.g., Streamlit Community Cloud)** whenever you’re ready.

