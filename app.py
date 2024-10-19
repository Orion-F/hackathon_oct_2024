import streamlit as st
import pandas as pd
import os
import requests
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt

# Helper function to load the dataset from JSON
def load_dataset(dataset_name):
    try:
        df = pd.read_json('data/reports.json')
        return df[df['dataset'] == dataset_name]
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

# Function to download reports based on URLs in the dataset
def download_reports(df, dataset_name):
    os.makedirs(f"data/{dataset_name}/reports", exist_ok=True)
    for index, row in df.iterrows():
        company_name = row["company_name"]
        year = row["year"]
        pdf_url = row["pdf_url"]
        file_name = f"{company_name}_{year}.pdf".replace(" ", "_")
        file_path = f"data/{dataset_name}/reports/{file_name}"

        # Download the file if it doesn't already exist
        if not os.path.exists(file_path):
            try:
                response = requests.get(pdf_url)
                if response.status_code == 200:
                    with open(file_path, "wb") as pdf_file:
                        pdf_file.write(response.content)
                    st.write(f"Downloaded: {file_name}")
                else:
                    st.warning(f"Failed to download {file_name}: {response.status_code}")
            except Exception as e:
                st.error(f"Error downloading {file_name}: {e}")
        else:
            st.write(f"{file_name} already exists.")

# Extract text from a downloaded PDF file
def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Failed to extract text from {file_path}: {e}")
        return ""

# Functionality for interacting with the dataset
def display_summary(df):
    st.write("### Dataset Summary")
    st.write(df.describe())

def filter_by_company(df):
    company_name = st.selectbox("Select a company to filter:", df["company_name"].unique())
    filtered_df = df[df["company_name"] == company_name]
    st.write(f"Showing data for {company_name}")
    st.write(filtered_df)

def filter_by_year(df):
    year = st.slider("Select a year to filter:", int(df["year"].min()), int(df["year"].max()))
    filtered_df = df[df["year"] == year]
    st.write(f"Showing data for the year {year}")
    st.write(filtered_df)

def visualize_trends(df):
    st.write("### Visualize Report Trends Over Time")
    companies = st.multiselect("Select companies to compare:", df["company_name"].unique())
    if companies:
        filtered_df = df[df["company_name"].isin(companies)]
        grouped_data = filtered_df.groupby(['year', 'company_name']).size().reset_index(name='Counts')
        
        plt.figure(figsize=(10, 6))
        for company in companies:
            company_data = grouped_data[grouped_data["company_name"] == company]
            plt.plot(company_data["year"], company_data["Counts"], marker='o', label=company)
        
        plt.title("Number of ESG Reports Over Time")
        plt.xlabel("Year")
        plt.ylabel("Number of Reports")
        plt.legend()
        st.pyplot(plt)

def search_company_info(df):
    query = st.text_input("Search for information by company name:")
    if query:
        search_result = df[df['company_name'].str.contains(query, case=False, na=False)]
        if not search_result.empty:
            st.write(f"Results for '{query}':")
            st.write(search_result)
        else:
            st.write(f"No results found for '{query}'.")

# Streamlit UI
st.title("ESG Report Analyzer")
st.write("Analyze and interact with ESG reports from different companies.")

# Dataset Selection
dataset_name = st.selectbox("Select a dataset to load:", ["handcrafted", "austria", "scraped"])
if st.button("Load Dataset"):
    df = load_dataset(dataset_name)
    if df is not None and not df.empty:
        st.write("### Loaded Dataset")
        st.write(df)

        # Download Reports
        if st.button("Download ESG Reports"):
            download_reports(df, dataset_name)

        # Interaction Options
        interaction_choice = st.selectbox("Choose how you want to interact with the data:", [
            "Summary of Dataset",
            "Filter by Company",
            "Filter by Year",
            "Visualize Trends",
            "Search Company Information",
            "Extract Text from Report"
        ])

        if interaction_choice == "Summary of Dataset":
            display_summary(df)
        elif interaction_choice == "Filter by Company":
            filter_by_company(df)
        elif interaction_choice == "Filter by Year":
            filter_by_year(df)
        elif interaction_choice == "Visualize Trends":
            visualize_trends(df)
        elif interaction_choice == "Search Company Information":
            search_company_info(df)
        elif interaction_choice == "Extract Text from Report":
            report_files = os.listdir(f"data/{dataset_name}/reports")
            report_file = st.selectbox("Select a report to extract text from:", report_files)
            if report_file:
                file_path = f"data/{dataset_name}/reports/{report_file}"
                extracted_text = extract_text_from_pdf(file_path)
                if extracted_text:
                    st.subheader(f"Extracted Text from {report_file}")
                    st.write(extracted_text[:1000])  # Display the first 1000 characters

