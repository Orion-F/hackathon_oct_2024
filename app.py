import streamlit as st
import os
import pandas as pd
import pickle
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# ----------------- Configuration -----------------

# Configure the page
st.set_page_config(layout="wide", page_title="SAM | Nexus Group AI", page_icon="🚀") # SAM = Sustainability Advanced Model

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"

# Paths
DATA_PATH = "data/handcrafted"
REPORTS_JSON_PATH = 'data/reports.json'
VECTOR_DB_NAME = "vector_db.pkl"

# ----------------- Style -----------------

NEXUS_PURPLE = "#523b88"
NEXUS_BLUE = "#5170ff"
# NEXUS_PINK = "#ffa3ff"
NEXUS_PINK = "#B963B9" # improved pink color

COLOR_LOW = "lightgreen"
COLOR_MEDIUM = "gold"
COLOR_HIGH = "tomato"

# ----------------- Data -----------------

# Load Data
df = pd.read_json(REPORTS_JSON_PATH)
# filter df to only show companies in dataset:handcrafted
df = df[df['dataset'] == 'handcrafted']
# sort companies alphabetically
df = df.sort_values(by='company_name')

# ----------------- Sidebar -----------------

# Sidebar Logo
st.sidebar.image("img/logo.png", width=250)

# Streamlit Sidebar for Navigation
st.sidebar.title("Navigation")
company = st.sidebar.selectbox("Select a Company:", df['company_name'].unique())
analysis_type = st.sidebar.radio("Select Analysis Type:", ["ESG Compliance","Report-Based Analysis", "Internet-Based Analysis"])

# ----------------- Main -----------------

st.title("SAM: Sustainability Advanced Model")

def show_esg_compliance(company):
    st.header(f"ESG Compliance: {company}")

def show_report_based_agent(company):
    st.header(f"ESG Agent: {company} (Report-Based Analysis)")
    company_dir = os.path.join(DATA_PATH, company)
    db_path = os.path.join(company_dir, VECTOR_DB_NAME)
    
    if not os.path.exists(db_path):
        st.error(f"No vector database found for {company}. Please generate the database first.")
        return

    with open(db_path, "rb") as f:
        db_bytes = pickle.load(f)
    db = FAISS.deserialize_from_bytes(db_bytes, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # Load the LLM
    llm = ChatOpenAI(model_name=MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

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

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Retrieve relevant documents and generate response
        response = retrieval_chain({"query": prompt})
        assistant_response = response['result']
        
        # get list of unique source document titles
        titles = list(set([os.path.basename(doc.metadata.get("source", "Unknown Document")) for doc in response['source_documents']]))

        # Display relevant document titles
        st.write("Documents used for this response:")
        for title in titles:
            st.write(f":page_facing_up: **{title}**")

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.chat_message("assistant").write(assistant_response)

def show_internet_based_agent(company):
    st.header(f"ESG Agent: {company} (Internet-Based Analysis)")
    
    # Load the LLM
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not OPENAI_API_KEY:
            st.info("Please add your OpenAI API key to continue.")
            return

        llm = ChatOpenAI(model_name=MODEL, openai_api_key=OPENAI_API_KEY, streaming=True)
        search = DuckDuckGoSearchRun(name="Search")
        search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

# Show the company page with the selected analysis type
if company:
    if analysis_type == "ESG Compliance":
        show_esg_compliance(company)
    elif analysis_type == "Report-Based Analysis":
        show_report_based_agent(company)
    elif analysis_type == "Internet-Based Analysis":
        show_internet_based_agent(company)
