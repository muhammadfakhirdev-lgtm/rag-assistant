import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from operator import itemgetter

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# --- 1. Load Environment Variables ---
# Ensures the OPENAI_API_KEY is available for LangChain components
load_dotenv()

# --- 2. Streamlit UI Configuration (Dark Mode for Readability) ---
st.set_page_config(
    page_title="LangChain RAG Assistant",
    layout="centered",
    initial_sidebar_state="expanded" # Changed to expanded to show RAG controls
)

# Custom CSS for enhanced UI/UX (High-contrast Dark Mode)
st.markdown("""
<style>
    /* Dark Mode Theme for High Contrast */
    .stApp {
        background-color: #1f2833; /* Dark Slate Gray */
        color: #c5c6c7; /* Light Gray default text */
        font-family: 'Inter', sans-serif;
    }
    /* Target the main chat area to limit width on large screens */
    .main .block-container {
        max-width: 800px;
        padding-top: 1rem;
        padding-bottom: 5rem;
    }
    /* Style the title (Primary Cyan Color) */
    h1 {
        color: #66fcf1; 
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    /* Ensure general text (like markdown) is light */
    p, .stMarkdown, .stText {
        color: #c5c6c7 !important;
    }
    /* Style for the chat input container */
    .st-emotion-cache-4oy32y {
        border-top: 1px solid #3c4f69; /* Darker border */
        padding-top: 10px;
        background-color: #1f2833; /* Match background */
    }
    /* Style for chat messages (important for contrast) */
    .st-chat-message-container p {
        color: #ffffff !important; /* White text inside bubbles */
    }
    /* Sidebar styling for better contrast */
    [data-testid="stSidebar"] {
        background-color: #2b3a4a; /* Slightly lighter dark background for contrast */
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 AI Document Q&A & Chat")
st.markdown("Upload documents on the left to activate Retrieval-Augmented Generation (RAG).")
st.markdown("---")

# --- 3. LangChain Core Setup & Initialization ---

# Initialize LLM and Embeddings (used for RAG)
try:
    model = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
    st.session_state.embeddings = OpenAIEmbeddings()
except Exception as e:
    st.error(f"Error initializing OpenAI components. Please ensure your OPENAI_API_KEY is set. Error: {e}")
    model = None
    st.session_state.embeddings = None

# --- 4. History Management ---

if 'history_store' not in st.session_state:
    st.session_state.history_store = {}
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! I'm ready for a chat, or upload a document to begin Q&A.")
    ]
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves or creates a ChatMessageHistory object for a given session ID."""
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = ChatMessageHistory()
    return st.session_state.history_store[session_id]

SESSION_ID = "stream_chat_session"
CONFIG = {"configurable": {"session_id": SESSION_ID}}


# --- 5. RAG Processing Logic ---

def process_uploaded_file(uploaded_file):
    """Loads, splits, and indexes the uploaded document, storing the retriever in state."""
    
    if st.session_state.embeddings is None:
        st.error("Embeddings model is not initialized. Cannot process documents.")
        return

    # 1. Save file to temp directory
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # 2. Select appropriate loader
    if uploaded_file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_path)
    elif uploaded_file.name.endswith(('.docx', '.doc')):
        loader = Docx2txtLoader(tmp_path)
    else:
        st.error("Unsupported file type. Please upload PDF or DOCX.")
        os.remove(tmp_path)
        return

    # 3. Load, split, and index
    try:
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            # Create FAISS vector store and retriever
            vectorstore = FAISS.from_documents(chunks, st.session_state.embeddings)
            st.session_state.retriever = vectorstore.as_retriever()
            st.session_state.rag_file_name = uploaded_file.name
            st.success(f"Successfully loaded and indexed **{uploaded_file.name}**.")
            st.session_state.messages.append(AIMessage(content=f"Document '{uploaded_file.name}' loaded! I can now answer questions based on its content."))

    except Exception as e:
        st.error(f"Error processing document: {e}")
        st.session_state.retriever = None
    finally:
        os.remove(tmp_path) # Clean up temp file

# --- 6. Sidebar (RAG Controls) ---
with st.sidebar:
    st.header("Document RAG Setup")
    
    # Display current RAG status
    if st.session_state.retriever:
        st.success(f"RAG Active: {st.session_state.rag_file_name}")
        if st.button("Clear Document & Reset Chat"):
            st.session_state.retriever = None
            st.session_state.rag_file_name = None
            st.session_state.messages = [AIMessage(content="RAG document cleared. Returning to general chat mode.")]
            st.session_state.history_store = {} # Clear chat history as well
            st.rerun()
    else:
        st.info("General Chat Mode (No Document)")

    uploaded_file = st.file_uploader(
        "Upload a PDF or DOCX file:", 
        type=['pdf', 'docx', 'doc'],
        accept_multiple_files=False,
        key="file_uploader_key"
    )

    if uploaded_file and st.session_state.retriever is None:
        process_uploaded_file(uploaded_file)


# --- 7. Chain Definition (Dynamic RAG/Chat) ---

def get_full_chain(retriever):
    """Returns the history-aware RAG or simple chat chain based on retriever availability."""
    
    if retriever:
        # --- RAG Chain Setup ---
        
        # 1. Define Context Retrieval Runnable
        # This function takes the chain input (containing 'messages') and uses the last message as the query
        def retrieve_context(input_dict):
            query = input_dict["messages"][-1].content
docs = retriever.invoke(query)
            return "\n\n---\n\n".join([doc.page_content for doc in docs])

        # Define the map that feeds the context and the messages into the prompt
        context_retriever_runnable = RunnableLambda(retrieve_context)

        # RAG System Prompt
        rag_system_message = (
            "You are an expert document assistant. Use the provided CONTEXT and the conversation history "
            "to answer the user's question. If the question cannot be answered from the CONTEXT, "
            "state clearly: 'I cannot find the answer in the document I have access to.' "
            "Always maintain a conversational tone and use the history for follow-up questions."
            "\n\n--- CONTEXT ---\n{context}\n------------------\n"
        )
        
        # The prompt template structure remains the same, but the system message now includes context placeholder
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", rag_system_message),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        
        # LCEL structure to pass context and messages to the prompt
        rag_chain = {
            "context": context_retriever_runnable,
            "messages": itemgetter("messages") 
        } | rag_prompt | model
        
        return rag_chain

    else:
        # --- Simple Conversational Chain Setup ---
        simple_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful and friendly conversational assistant. Keep your responses concise, engaging, and always remember the user's name if they've told you."),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        return simple_prompt | model

# Wrap the dynamic chain with history management
if model:
    dynamic_chain = get_full_chain(st.session_state.retriever)
    with_message_history = RunnableWithMessageHistory(
        dynamic_chain,
        get_session_history,
        input_messages_key="messages"
    )
else:
    with_message_history = None

# --- 8. Streamlit Chat Interface ---

# Display existing messages
for message in st.session_state.messages:
    message_type = "assistant" if isinstance(message, AIMessage) else "user"
    avatar = "🤖" if message_type == "assistant" else "👤"
    with st.chat_message(message_type, avatar=avatar):
        st.markdown(message.content)

# Handle new user input
if user_input := st.chat_input("Ask me anything..."):
    if not model or not with_message_history:
        st.error("Cannot process request: AI model is not initialized or the chain setup failed.")
    else:
        # 1. Add new user message to the display state
        user_message = HumanMessage(content=user_input)
        st.session_state.messages.append(user_message)

        # 2. Display the user message immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # 3. Stream the AI response for better UX
        with st.chat_message("assistant", avatar="🤖"):
            response_container = st.empty()
            full_response = ""

            # LangChain input only requires the new message for the history runnable
            # NOTE: We pass the WHOLE history here, as the chain handles it internally.
            lc_input = {"messages": st.session_state.messages} 
            
            # Stream the response from the history-aware chain
            stream = with_message_history.stream(
                lc_input,
                config=CONFIG
            )

            for chunk in stream:
                # Concatenate content and display with a typing cursor (▌)
                if chunk.content:
                    full_response += chunk.content
                    response_container.markdown(full_response + "▌")
            
            # Final output without the cursor
            response_container.markdown(full_response)
            
            # 4. Add final AI response to the display state
            st.session_state.messages.append(AIMessage(content=full_response))

# Hint for the user
if len(st.session_state.messages) <= 1:
    st.sidebar.info("The AI remembers context! Try: 'Hi my name is John' and then 'What is my name?'")

# --- End of Streamlit UI ---
