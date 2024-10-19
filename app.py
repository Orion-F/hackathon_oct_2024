import os
import requests
import PyPDF2
import pandas as pd
import pickle
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import tiktoken

# Paths
DATA_PATH = "data/handcrafted"
CRITERIA_PATH = 'criteria.csv'
REPORTS_JSON_PATH = 'data/reports.json'
VECTOR_DB_NAME = "vectors.db"

# Load Data
criteria_df = pd.read_csv(CRITERIA_PATH)
df = pd.read_json(REPORTS_JSON_PATH)

# Create directory for data if it doesn't exist
os.makedirs(DATA_PATH, exist_ok=True)

# Function to download reports
def download_reports(df: pd.DataFrame, company_name: str, save_dir: str):
    company_dir = os.path.join(save_dir, company_name)
    os.makedirs(company_dir, exist_ok=True)
    
    for url in df['pdf_url']:
        pdf_filename = os.path.basename(url)
        response = requests.get(url)
        with open(os.path.join(company_dir, pdf_filename), 'wb') as file:
            file.write(response.content)
    print(f"Reports for {company_name} downloaded successfully.")

# Function to create vector database from PDF reports
def create_vector_database(files_path: str):
    documents = []
    for file in os.listdir(files_path):
        _, file_extension = os.path.splitext(file)
        text = ""
        if file_extension == ".pdf":
            with open(os.path.join(files_path, file), 'rb') as f:
                reader = PyPDF2.PdfReader(f, strict=False)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            
            if text:
                documents.append(Document(page_content=text, metadata={"source": file}))
            else:
                print(f"WARNING: No text extracted from {file}")
        else:
            print(f"Unused file: {file}")
    
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300, separators=["\n\n", "\n"])
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(texts, embeddings)
        
        tokenizer = tiktoken.get_encoding("cl100k_base")
        build_token_count = sum([len(tokenizer.encode(doc.page_content)) for doc in texts])
        print(f"Token count: {build_token_count}")
        
        db_path = os.path.join(files_path, VECTOR_DB_NAME)
        with open(db_path, "wb") as f:
            pickle.dump(db.serialize_to_bytes(), f)
        print(f"Vector database created and saved at {db_path}")

# Function to rate company based on criteria
def rate_company(company_dir: str, criteria_df: pd.DataFrame):
    ratings = []
    
    vector_db_path = os.path.join(company_dir, VECTOR_DB_NAME)
    if not os.path.exists(vector_db_path):
        print(f"No vector database found for {company_dir}")
        return
    
    with open(vector_db_path, "rb") as f:
        db_bytes = pickle.load(f)
        db = FAISS.deserialize_from_bytes(db_bytes, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    
    for _, crit in criteria_df.iterrows():
        query = crit['criterion']
        retrieved_docs = db.similarity_search(query)
        compliance_score = 0
        for doc in retrieved_docs:
            if query.lower() in doc.page_content.lower():
                compliance_score += crit['weight']
        ratings.append({'criterion': query, 'compliance_score': compliance_score})
    
    ratings_df = pd.DataFrame(ratings)
    ratings_path = os.path.join(company_dir, "ratings.csv")
    ratings_df.to_csv(ratings_path, index=False)
    print(f"Ratings for company saved at {ratings_path}")

# Process each company
for company_name in df['company_name'].unique():
    company_df = df[df['company_name'] == company_name]
    company_dir = os.path.join(DATA_PATH, company_name)
    
    # Step 1: Download reports
    download_reports(company_df, company_name, DATA_PATH)
    
    # Step 2: Create vector database for each company
    create_vector_database(company_dir)
    
    break
    
    # Step 3: Rate the company based on the criteria
    rate_company(company_dir, criteria_df)
    
    

print("All companies processed successfully.")
