# 📚 PDF Knowledge Assistant

> A powerful **Streamlit-based web application** that lets users upload PDFs, process their content, and query them using state-of-the-art **AI models** like BERT, TinyLlama, LLaMA-2, and more.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen)
![LangChain](https://img.shields.io/badge/LangChain-Integrated-orange)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 Features

- 📄 Upload **multiple PDF files** for smart text extraction.
- 🤖 Choose from **multiple AI models** (QA and conversational).
- 💬 Chat with your PDFs in **real-time**, with contextual memory.
- 📊 Monitor **system performance** (RAM usage & response time).
- 🔒 Hugging Face token support for gated models like LLaMA-2.
- ⚙️ **Auto-installs** missing packages for smoother setup.

---

## 📋 Table of Contents

- [🛠 Installation](#-installation)
- [📖 Usage](#-usage)
- [🤖 Supported Models](#-supported-models)
- [🐛 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [🧑‍💻 Author](#-author)
- [📜 License](#-license)

---

## 🛠 Installation


#2. Set Up a Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Or let the app handle it automatically on first run!

4. Add Environment Variables
Create a .env file in the project root with:

env
Copy
Edit
HF_TOKEN=your_huggingface_token
Get your token from huggingface.co/settings/tokens.

5. Launch the App
bash
Copy
Edit
streamlit run app.py
Then open your browser at http://localhost:8501

📖 Usage
🔹 Upload PDFs
Use the sidebar to upload one or more PDF files.

🔹 Choose an AI Model
Select a model like:

DistilBERT-QA for light QA

LLaMA-2-7b-chat-hf for advanced conversations

🔹 Process Documents
Click "Process Documents" to chunk and vectorize the content.

🔹 Ask Questions
Type your questions in the chat box and get AI-powered answers!

🔹 Monitor Performance
See RAM usage and response time in real time from the sidebar.

🔹 Reset Conversation
Use "Clear Conversation" to start a new chat.

🤖 Supported Models

Model Name	Type	Description
DistilBERT-QA	QA	Lightweight model for fact-based answers
BERT-QA	QA	Deep QA model with robust context understanding
MiniLM-QA	QA	Fast, efficient QA with decent accuracy
BART-QA	QA	Generative QA with fluency
TinyLlama-1.1B-Chat	Conversational	Small conversational model by TinyLlama
Qwen2-0.5B	Conversational	Efficient open-source chat model
Gemma-2-2b-it	Conversational	Instruction-tuned model by Google
LLaMA-2-7b-chat-hf	Conversational	High-performance, gated model (Hugging Face Token)
⚠️ Note: Heavier models (e.g., LLaMA-2, Gemma-2) require more memory.

🐛 Troubleshooting

Problem	Solution
❌ Failed to install [package]	Run pip install [package] manually.
❌ Hugging Face token not found	Add HF_TOKEN=your_token in .env.
⚠️ No text extracted from PDFs	Use OCR if your PDFs are scanned images.
💥 App crashes with memory errors	Try lightweight models like DistilBERT-QA.
🕐 Model loading is too slow	Ensure fast internet and enough storage.
For more help: Hugging Face Docs, Streamlit Forum

🤝 Contributing
We welcome contributions! 🛠

Fork the repo

Create a branch: git checkout -b feature/awesome-feature

Commit your changes: git commit -m "Added awesome feature"

Push: git push origin feature/awesome-feature

Submit a Pull Request 🎉

🧑‍💻 Author
Vimarsh Dwivedi
