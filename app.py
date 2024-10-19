import streamlit as st
import os
import pandas as pd
import pickle
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"

# Paths
DATA_PATH = "data/handcrafted"
REPORTS_JSON_PATH = 'data/reports.json'
VECTOR_DB_NAME = "vector_db.pkl"

# Load Data
df = pd.read_json(REPORTS_JSON_PATH)

# Streamlit App
def show_home_page():
    st.title("Handcrafted Dataset Overview")
    st.write("### Available Companies")
    st.dataframe(df)

def show_company_page(company):
    st.title(f"{company} - Document Query")
    company_dir = os.path.join(DATA_PATH, company)
    db_path = os.path.join(company_dir, VECTOR_DB_NAME)

    if not os.path.exists(db_path):
        st.error(f"No vector database found for {company}. Please generate the database first.")
        return

    with open(db_path, "rb") as f:
        db_bytes = pickle.load(f)
    db = FAISS.deserialize_from_bytes(db_bytes, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # Load the LLM
    llm = ChatOpenAI(model_name=MODEL, temperature=0)

    system_prompt = """
    You are an expert assistant. Use only the following retrieved context to answer the question accurately and concisely. 
    If nothing is mentioned in the context, say "I don't know".
    Context: {context}
    Question: {question}
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"], 
        template=system_prompt
    )

    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    query = st.text_input("Enter your query:")
    if query:
        response = retrieval_chain({"query": query})
        st.write("### Answer")
        st.write(response['result'])

        st.write("### Top Matching Documents")
        for i, doc in enumerate(response['source_documents']):
            st.write(f"**Document {i + 1}:**")
            st.write(doc.page_content)

# Streamlit Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Home", "Company Query"])

if page == "Home":
    show_home_page()
else:
    company = st.sidebar.selectbox("Select a Company:", df['company_name'].unique())
    if company:
        show_company_page(company)
