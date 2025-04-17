# chatbot_web_qna.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ Google Gemini

# Load environment variables
load_dotenv()

# Get Google API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Please add it to your .env file.")

# Initialize Google Gemini LLM
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.3)
# llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-turbo", temperature=0.3)  # Uncomment for different model

# --- Function to extract text from a URL ---
def get_website_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        return f"Error fetching website: {e}"

# --- Load and prepare the document for Q&A ---
def prepare_documents(raw_text):
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)
    docs = [Document(page_content=t) for t in texts]
    return docs

# --- Setup LangChain QA ---
def ask_question(docs, question):
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=question)
    return answer

# --- Streamlit App Interface ---
st.title("🔍 Website Q&A Chatbot (Powered by Google Gemini)")
st.write("Ask anything about a website's content.")

url = st.text_input("Enter a website URL", "https://example.com")

if url:
    raw_text = get_website_text(url)
    if raw_text.startswith("Error"):
        st.error(raw_text)
    else:
        docs = prepare_documents(raw_text)
        question = st.text_input("Ask a question about the website:")
        if question:
            with st.spinner("Thinking..."):
                answer = ask_question(docs, question)
                st.success("Answer:")
                st.write(answer)