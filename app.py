import streamlit as st
import openai

import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader

import os

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Extract text from the uploaded PDF
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Query OpenAI for insights from the report
def query_openai(text, query):
    prompt = f"Based on the following ESG report:\n\n{text}\n\nAnswer this query: {query}"
    
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Example Visualization function (adjust for actual data)
def visualize_data():
    years = [2019, 2021, 2023]
    carbon_emissions = [500, 450, 300]

    plt.figure(figsize=(10, 6))
    plt.plot(years, carbon_emissions, marker='o')
    plt.title("Carbon Emissions Over Time")
    plt.xlabel("Year")
    plt.ylabel("Carbon Emissions (tons)")
    st.pyplot(plt)

# Streamlit UI
st.title("ESG Report Analyzer")
st.write("Upload an ESG report and analyze sustainability data.")

# File Upload
uploaded_file = st.file_uploader("Upload an ESG report (PDF)", type="pdf")
if uploaded_file:
    # Extract text from uploaded PDF
    text = extract_text_from_pdf(uploaded_file)
    
    st.subheader("Report Content")
    st.write(text[:1000])  # Display first 1000 characters as a preview

    # Query Section
    query = st.text_input("Ask a question about this report:")
    
    if query:
        # Send the query to OpenAI
        response = query_openai(text, query)
        
        st.subheader("AI Generated Answer")
        st.write(response)

# Load data from reports.json (if applicable)
if st.checkbox("Show Pre-loaded Company Data"):
    df = pd.read_json('data/reports.json')
    st.write(df)

# Sample Visualization
if st.checkbox("Show ESG Trends Visualization"):
    visualize_data()

