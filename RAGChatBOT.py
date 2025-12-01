import streamlit as st
from PyPDF2 import PdfReader
from io import StringIO
from dotenv import load_dotenv
import os

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load API key from environment file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Chat with Document", layout="wide")
st.title("Chat with Document")
st.caption("Upload a PDF or TXT and ask questions based on its content")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

# Helper to extract text from uploaded file
def extract_text_from_file(uploaded):
    if uploaded.type == "application/pdf":
        reader = PdfReader(uploaded)
        return "".join([page.extract_text() or "" for page in reader.pages])
    else:
        return StringIO(uploaded.getvalue().decode("utf-8")).read()

# Robust retrieval helper: tries multiple retriever methods and falls back to vector_store
def retrieve_documents(retriever, vector_store, query, k=3):
    """
    Try several common retriever method names (different LangChain versions).
    Returns a list of strings (document texts).
    """
    # 1) try get_relevant_documents (common)
    try:
        docs = retriever.get_relevant_documents(query)
        # docs may be Document objects or strings
        return [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
    except Exception:
        pass

    # 2) try get_relevant_texts (older/other variants)
    try:
        texts = retriever.get_relevant_texts(query)
        return [t if isinstance(t, str) else str(t) for t in texts]
    except Exception:
        pass

    # 3) try retrieve (some retrievers provide retrieve)
    try:
        docs = retriever.retrieve(query)
        return [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
    except Exception:
        pass

    # 4) fallback to vector_store similarity search (works for FAISS and others)
    try:
        docs = vector_store.similarity_search(query, k=k)
        return [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
    except Exception:
        pass

    # final fallback: empty list
    return []

if uploaded_file:
    text = extract_text_from_file(uploaded_file)
    st.subheader("Document preview")
    preview = text[:1500] + "..." if len(text) > 1500 else text
    st.text_area("Preview", preview, height=200)

    with st.spinner("Splitting text and creating embeddings..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        # Use retriever as provided by the vector store; may be VectorStoreRetriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("Ask a question about the document")
    user_query = st.text_input("Enter your question here")

    if user_query:
        with st.spinner("Retrieving context and generating answer..."):
            # Use the robust retrieval helper
            doc_texts = retrieve_documents(retriever, vector_store, user_query, k=3)
            context_text = "\n\n".join(doc_texts).strip()

            if not context_text:
                answer = "I couldn't find relevant information in the document."
            else:
                # prepare memory from chat_history (question/answer pairs)
                # Prepare conversation memory (previous Q&A)
                memory_msgs = []
                for q, a in st.session_state.chat_history:
                    memory_msgs.append(HumanMessage(content=f"User: {q}"))
                    memory_msgs.append(HumanMessage(content=f"Assistant: {a}"))

                # System instruction: allow model to use both document + its own reasoning
                system_prompt = (
                    "You are a helpful assistant. "
                    "Use the uploaded document as the primary source for your answer. "
                    "If the document does not contain the needed information, "
                    "you may use your own general knowledge to fill in the gaps. "
                    "Always make sure the answer is relevant, factual, and clear."
                )

                # Combine document context with user question
                human_prompt = (
                    f"Document context (may be incomplete):\n{context_text}\n\n"
                    f"User question: {user_query}"
                )

                messages = [SystemMessage(content=system_prompt)] + memory_msgs + [HumanMessage(content=human_prompt)]

                # Call the chat model. Many LangChain wrappers provide .invoke; if yours uses a different call,
                # replace llm.invoke(messages) with the appropriate call (llm(messages) or llm.predict_messages(messages)).
                try:
                    response = llm.invoke(messages)
                    answer = response.content
                except Exception:
                    # Try calling llm as a callable (other LangChain versions)
                    try:
                        response = llm(messages)
                        # response may be a ChatResult-like object; try to extract safely
                        # Common patterns: response[0].message.content or response.content
                        if hasattr(response, "generations"):
                            # response.generations[0][0].text is common for LLMResult
                            gen = response.generations
                            # best-effort extraction
                            answer = gen[0][0].text if gen and gen[0] and hasattr(gen[0][0], "text") else str(response)
                        elif isinstance(response, list) and response:
                            answer = str(response[0])
                        else:
                            answer = str(response)
                    except Exception as e:
                        # Last-resort fallback message to user
                        answer = f"Model call failed: {e}. Please check your LangChain/OpenAI integration."

            # store and display
            st.session_state.chat_history.append((user_query, answer))
            st.markdown("### Answer")
            st.write(answer)

    # show chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Chat History")
        for q, a in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(f"**You:** {q}")
            with st.chat_message("assistant"):
                st.markdown(f"**Bot:** {a}")

else:
    st.info("Please upload a document to start.")