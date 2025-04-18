import streamlit as st
from dotenv import load_dotenv
import os
import sys
import time
import subprocess
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import torch
import psutil
from htmlTemplates import css, bot_template, user_template

# Function to check and install missing packages
def check_and_install_packages():
    required_packages = {
        "PyPDF2": "PyPDF2==3.0.1",
        "langchain": "langchain==0.3.0",
        "langchain_community": "langchain-community==0.3.0",
        "sentence_transformers": "sentence-transformers==3.1.1",
        "faiss_cpu": "faiss-cpu==1.8.0",
        "transformers": "transformers==4.44.2",
        "accelerate": "accelerate==0.33.0",
        "torch": "torch==2.4.1",
        "torchvision": "torchvision==0.19.1",
        "psutil": "psutil==6.0.0",
        "streamlit": "streamlit==1.38.0",
        "python_dotenv": "python-dotenv==1.0.1"
    }
    
    missing_packages = []
    
    try:
        result = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"], text=True)
        installed_packages = {pkg["name"].lower(): pkg["version"] for pkg in json.loads(result)}
    except Exception as e:
        st.error(f"Failed to get installed packages: {str(e)}")
        st.stop()
    
    for pkg_name, pkg_spec in required_packages.items():
        pkg_pip_name = pkg_spec.split("==")[0].lower()
        required_version = pkg_spec.split("==")[1] if "==" in pkg_spec else None
        installed_version = installed_packages.get(pkg_pip_name)
        
        if not installed_version or (required_version and installed_version != required_version):
            missing_packages.append(pkg_spec)
    
    if missing_packages:
        st.error(f"Missing or incorrect version of packages: {', '.join(missing_packages)}")
        st.info("Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                st.success(f"Successfully installed {package}")
            except Exception as e:
                st.error(f"Failed to install {package}: {str(e)}")
                st.stop()
        
        st.info("Please restart the application after the installations.")
        st.stop()

check_and_install_packages()

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForQuestionAnswering
    from transformers import BartForQuestionAnswering, BartTokenizer
except ImportError as e:
    st.error(f"Error importing required modules: {str(e)}")
    try:
        import transformers
        st.info(f"Transformers version: {transformers.__version__}")
    except:
        st.info("Transformers package is not installed.")
    st.info("Please ensure all required packages are installed properly. Run 'pip install -r requirements.txt'.")
    st.stop()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    if not text.strip():
        st.warning("No text could be extracted from the uploaded PDFs.")
    return text

def get_text_chunks(text):
    if not text.strip():
        st.warning("No text available to chunk.")
        return []
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=256,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    if not chunks:
        st.warning("No text chunks were created.")
    return chunks

def get_vectorstore(text_chunks):
    if not text_chunks:
        st.error("Cannot create vector store: No text chunks provided.")
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create vector store: {str(e)}")
        return None

def get_conversation_chain(vectorstore, model_choice):
    if vectorstore is None:
        st.error("Cannot create conversation chain: Invalid vector store.")
        return None

    model_options = {
        "DistilBERT-QA": "distilbert-base-uncased-distilled-squad",
        "BERT-QA": "deepset/bert-base-cased-squad2",
        "MiniLM-QA": "deepset/minilm-uncased-squad2",
        "BART-QA": "valhalla/bart-small-finetuned-squadv1",
        "Tiny-Llama-Chat-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen2-0.5B": "Qwen/Qwen2-0.5B-Instruct",
        "Gemma-2-2b-it": "google/gemma-2-2b-it",
        "LLaMA-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf"
    }
    
    repo_id = model_options.get(model_choice)
    if not repo_id:
        st.error(f"Unknown model choice: {model_choice}")
        return None

    try:
        with st.spinner(f"Loading {model_choice} model..."):
            # Handle question-answering models (DistilBERT, BERT, MiniLM, BART)
            if model_choice in ["DistilBERT-QA", "BERT-QA", "MiniLM-QA", "BART-QA"]:
                if "bart" in repo_id.lower():
                    tokenizer = BartTokenizer.from_pretrained(repo_id)
                    model = BartForQuestionAnswering.from_pretrained(
                        repo_id,
                        torch_dtype=torch.float32
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(repo_id)
                    model = AutoModelForQuestionAnswering.from_pretrained(
                        repo_id,
                        torch_dtype=torch.float32
                    )
                
                # Create a pipeline for question-answering
                qa_pipeline = pipeline(
                    "question-answering",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=512,
                    device=-1  # Ensure CPU usage
                )

                # Custom QA chain to integrate with vectorstore
                class CustomQAChain:
                    def __init__(self, qa_pipeline, retriever, memory):
                        self.qa_pipeline = qa_pipeline
                        self.retriever = retriever
                        self.memory = memory
                    
                    def __call__(self, inputs):
                        question = inputs["question"]
                        # Retrieve relevant documents
                        docs = self.retriever.get_relevant_documents(question)
                        context = " ".join([doc.page_content for doc in docs])
                        
                        # Get chat history
                        chat_history = self.memory.load_memory_variables({})["chat_history"]
                        
                        # Run question-answering pipeline
                        result = self.qa_pipeline(question=question, context=context)
                        answer = result["answer"]
                        
                        # Update memory
                        self.memory.save_context({"question": question}, {"answer": answer})
                        
                        # Return response in LangChain-compatible format
                        return {
                            "question": question,
                            "answer": answer,
                            "chat_history": self.memory.load_memory_variables({})["chat_history"]
                        }

                memory = ConversationBufferMemory(
                    memory_key='chat_history',
                    return_messages=True,
                    max_token_limit=1000
                )
                
                conversation_chain = CustomQAChain(
                    qa_pipeline=qa_pipeline,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
                    memory=memory
                )
            
            # Handle generative models (Tiny-Llama, Qwen2, Gemma-2, LLaMA-2)
            else:
                # Load Hugging Face token for gated models (e.g., LLaMA-2)
                hf_token = os.getenv("HF_TOKEN")
                if model_choice == "LLaMA-2-7b-chat-hf" and not hf_token:
                    st.error("Hugging Face token (HF_TOKEN) not found in .env file for LLaMA-2-7b-chat-hf. Please set it up.")
                    return None
                
                tokenizer = AutoTokenizer.from_pretrained(
                    repo_id,
                    token=hf_token if model_choice == "LLaMA-2-7b-chat-hf" else None
                )
                model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=(model_choice == "Qwen2-0.5B"),
                    token=hf_token if model_choice == "LLaMA-2-7b-chat-hf" else None
                )
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.95,
                    return_full_text=False,
                    device=-1  # Ensure CPU usage
                )
                llm = HuggingFacePipeline(pipeline=pipe)
                
                memory = ConversationBufferMemory(
                    memory_key='chat_history',
                    return_messages=True,
                    max_token_limit=1000
                )
                
                conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
                    memory=memory
                )
            
            return conversation_chain

    except Exception as e:
        st.error(f"Failed to load model {model_choice}: {str(e)}")
        if model_choice == "LLaMA-2-7b-chat-hf":
            st.info("LLaMA-2 requires a Hugging Face token. Ensure you have access and HF_TOKEN is set in .env.")
        elif model_choice == "Gemma-2-2b-it":
            st.info("Gemma-2 may require sufficient memory or access permissions. Try a lighter model like DistilBERT-QA.")
        if model_choice != "DistilBERT-QA":
            st.info("Trying fallback to DistilBERT-QA model...")
            try:
                return get_conversation_chain(vectorstore, "DistilBERT-QA")
            except Exception as fallback_error:
                st.error(f"Fallback model failed: {str(fallback_error)}")
        return None

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please process documents first.")
        return
    if not user_question.strip():
        st.warning("Please enter a valid question.")
        return

    try:
        with st.spinner("Thinking..."):
            start_time = time.time()
            
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            
            processing_time = time.time() - start_time
            st.session_state.response_times.append(processing_time)
        
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        
        if st.session_state.response_times:
            avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
            st.sidebar.info(f"Last response time: {processing_time:.2f}s\nAvg response time: {avg_time:.2f}s")
            
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    load_dotenv()

    st.set_page_config(
        page_title="PDF Knowledge Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "response_times" not in st.session_state:
        st.session_state.response_times = []

    st.header("📚 PDF Knowledge Assistant")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Settings")
        model_choice = st.selectbox(
            "Select AI Model",
            ["DistilBERT-QA", "MiniLM-QA", "BERT-QA", "BART-QA", "Tiny-Llama-Chat-1.1B", "Qwen2-0.5B", "Gemma-2-2b-it", "LLaMA-2-7b-chat-hf"],
            index=0,
            help="Select a model optimized for question-answering or chat. LLaMA-2 requires HF_TOKEN."
        )
        
        st.subheader("Document Processing")
        pdf_docs = st.file_uploader(
            "Upload PDFs and click 'Process'",
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        process_button = st.button("Process Documents")
        
        if process_button:
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        st.error("Failed to extract text from PDFs.")
                        st.stop()
                    
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        st.error("Failed to create text chunks.")
                        st.stop()
                    
                    vectorstore = get_vectorstore(text_chunks)
                    if not vectorstore:
                        st.error("Failed to create vector store.")
                        st.stop()
                    
                    st.session_state.conversation = get_conversation_chain(vectorstore, model_choice)
                    st.session_state.response_times = []
                    
                    if st.session_state.conversation:
                        st.success(f"Documents processed successfully with {model_choice}!")
                        st.balloons()
                    else:
                        st.error(f"Failed to initialize conversation chain with {model_choice}.")
            else:
                st.warning("Please upload at least one PDF document.")
                
        if st.button("Clear Conversation"):
            st.session_state.chat_history = None
            st.session_state.response_times = []
            st.success("Conversation cleared!")

        st.subheader("System Info")
        memory = psutil.virtual_memory()
        st.info(f"RAM: {memory.percent}% used\nAvailable: {memory.available / (1024 * 1024 * 1024):.1f} GB")
    
    with col1:
        st.subheader("Ask questions about your documents")
        user_question = st.text_input("Type your question here:")
        if user_question:
            handle_userinput(user_question)

if __name__ == '__main__':
    main()