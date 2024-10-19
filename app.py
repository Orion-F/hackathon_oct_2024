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
import plotly.graph_objects as go

# ----------------- Configuration -----------------

# Configure the page
st.set_page_config(layout="wide", page_title="SAM | Nexus Group AI", page_icon="🚀") # SAM = Sustainability Advanced Model

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# MODEL = "gpt-3.5-turbo"
MODEL = "gpt-4o"

# Paths
DATA_PATH = "data/handcrafted"
REPORTS_JSON_PATH = 'data/reports.json'
VECTOR_DB_NAME = "vector_db.pkl"

# ----------------- Style -----------------

NEXUS_PURPLE = "#523b88"
NEXUS_BLUE = "#5170ff"
# NEXUS_PINK = "#ffa3ff"
NEXUS_PINK = "#B963B9" # improved pink color

COLOR_HIGH = '#2E7D32'   # Darker Green
COLOR_MEDIUM = '#F9A825' # Darker Golden Yellow
COLOR_LOW = '#B71C1C'    # Darker Red
VALUE_LOW = 0
VALUE_MEDIUM = 50
VALUE_HIGH = 100

# Function to apply the colors to the Compliance column
def compliance_color(val):
    if val == "HIGH":
        return f'background-color: {COLOR_HIGH}'
    elif val == "MEDIUM":
        return f'background-color: {COLOR_MEDIUM}'
    elif val == "LOW":
        return f'background-color: {COLOR_LOW}'
    else:
        return ''

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

# API Key Input
if not openai_api_key:
    st.sidebar.warning("OpenAI API key is required for certain analyses. Please add your key to proceed.")
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")
    if openai_api_key and st.sidebar.button("Submit API Key"):
        st.session_state["openai_api_key"] = openai_api_key
        st.experimental_rerun()
else:
    st.session_state["openai_api_key"] = openai_api_key

# ----------------- Main -----------------

def show_main():
    st.title("SAM: Sustainability Advanced Model")

    # Show the company page with the selected analysis type
    if company:
        if analysis_type == "ESG Compliance":
            show_esg_compliance(company)
        elif analysis_type == "Report-Based Analysis":
            if "openai_api_key" not in st.session_state or not st.session_state["openai_api_key"]:
                st.info("Please add your OpenAI API key to continue with Report-Based Analysis.")
                return
            show_report_based_agent(company)
        elif analysis_type == "Internet-Based Analysis":
            if "openai_api_key" not in st.session_state or not st.session_state["openai_api_key"]:
                st.info("Please add your OpenAI API key to continue with Internet-Based Analysis.")
                return
            show_internet_based_agent(company)  

# Function to create the compliance dial with percentage display
def create_compliance_dial(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",  # Display percentage
        value=value,
        number={'suffix': "%"},  # Display the value as a percentage
        gauge={
            'axis': {'range': [0, 100], 'tickvals': [0, 33, 66, 100], 'ticktext': ['Low', 'Medium', 'High', '']},
            'bar': {'color': "darkblue"},  # Color of the bar that shows the actual value
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},   # Low section
                {'range': [33, 66], 'color': "gold"},        # Medium section
                {'range': [66, 100], 'color': "tomato"}      # High section
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    # Reduce the size of the gauge to fit in the sidebar
    fig.update_layout(
        font={'size': 10},
        height=200, width=200, margin=dict(l=10, r=10, t=10, b=10)
    )
    
    return fig

def show_esg_compliance(company):
    st.header(f"ESG Compliance: {company}")
    company_dir = os.path.join(DATA_PATH, company)

    # Read the results.csv file from the company directory
    results_path = os.path.join(company_dir, "results.csv")
    results_df = pd.read_csv(results_path)  # headers: Goal, Compliance, Explanation, where Compliance is HIGH, MEDIUM, LOW

    # Calculate the average compliance score
    compliance_scores = results_df['Compliance'].map({"HIGH": VALUE_HIGH, "MEDIUM": VALUE_MEDIUM, "LOW": VALUE_LOW})
    average_compliance = compliance_scores.mean()

    st.write(f"Based on the collected reports, the SAM ESG Agent estimates that {company} has the following levels of compliance with the goals of the EU Green Deal.")

    st.header("Overall Compliance")
    st.plotly_chart(create_compliance_dial(average_compliance), use_container_width=True)

    st.header("Detailed Compliance")
    
    results_df = results_df.rename(columns={"Goal": "EU Green Deal Goal"})
    
    # Apply the color function to the dataframe
    styled_df = results_df.style.applymap(compliance_color, subset=['Compliance'])

    # Display the styled dataframe in Streamlit
    st.dataframe(styled_df)


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
    llm = ChatOpenAI(model_name=MODEL, temperature=0, openai_api_key=st.session_state["openai_api_key"])

    system_prompt = """
    You are an expert assistant. Use only the following retrieved context to answer the question accurately and concisely. 
    If nothing is mentioned in the context, say "I am unable to answer that question".
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
        if "openai_api_key" not in st.session_state or not st.session_state["openai_api_key"]:
            st.info("Please add your OpenAI API key to continue.")
            return

        llm = ChatOpenAI(model_name=MODEL, openai_api_key=st.session_state["openai_api_key"], streaming=True)
        search = DuckDuckGoSearchRun(name="Search")
        search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

show_main()
