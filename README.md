# SAM: Sustainability Advanced Model | Nexus Group AI

SAM (Sustainability Advanced Model) is an AI-driven platform that analyzes and assesses the ESG (Environmental, Social, and Governance) compliance of companies based on their reports and internet-based data. This project leverages LangChain, FAISS, and OpenAI’s GPT models to generate detailed compliance results for selected companies in alignment with the goals of the EU Green Deal.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Setup](#setup)
  - [Running the Application](#running-the-application)
  - [Generating ESG Compliance Results](#generating-esg-compliance-results)
  - [Setting up Vector Databases](#setting-up-vector-databases)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [License](#license)

## Features

- **ESG Compliance Analysis**: Provides an AI-powered assessment of a company's compliance with ESG goals based on their internal and publicly available reports.
- **Report-Based Analysis**: Uses vector databases created from company reports to retrieve relevant information and assess ESG compliance.
- **Internet-Based Analysis**: Integrates web searches to complement report-based analysis for a comprehensive view of company compliance.
- **Interactive Dashboards**: Uses Streamlit for an interactive web application to visualize ESG compliance and other analytical insights.
- **Automated PDF Processing**: Downloads company reports, extracts text, and builds searchable vector databases.

## Installation

### Prerequisites

- Python 3.8 or higher
- [OpenAI API key](https://beta.openai.com/signup/)
- [DuckDuckGo API Key](https://duckduckgo.com)
- Install dependencies via `requirements.txt`

### Clone the repository

```bash
git clone https://github.com/nexus-group-ai/sam-esg-compliance.git
cd sam-esg-compliance
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file at the root of the project with the following contents:

```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Setup

Before running the application, ensure the company data and reports are available in the `data/handcrafted` folder. You can automate downloading reports and setting up vector databases by following the next steps.

### Setting up Vector Databases

Run the `setup_vector_dbs.ipynb` notebook to download the company reports, process the text, and create vector databases:

1. Launch the Jupyter notebook:

    ```bash
    jupyter notebook setup_vector_dbs.ipynb
    ```

2. Follow the steps in the notebook to:
   - Download company reports.
   - Process the PDFs to extract text.
   - Create and save vector databases for each company.

### Generating ESG Compliance Results

Run the `gen_compliance.ipynb` notebook to generate compliance results for each company based on the EU Green Deal goals:

1. Launch the Jupyter notebook:

    ```bash
    jupyter notebook gen_compliance.ipynb
    ```

2. Follow the steps in the notebook to:
   - Load company-specific vector databases.
   - Use the pre-defined compliance criteria.
   - Run the analysis to generate ESG compliance reports.
   - Save the results as CSV files for each company.

### Running the Application

Once the vector databases and compliance results are prepared, you can run the Streamlit application:

```bash
streamlit run app.py
```

The app will launch in your default browser, allowing you to:

- Select a company from the sidebar.
- View ESG compliance results, report-based analysis, or internet-based analysis for the selected company.

## Project Structure

```bash
├── app.py                   # Main Streamlit app
├── data/handcrafted/         # Company data and reports folder
├── setup_vector_dbs.ipynb    # Jupyter notebook to create vector databases
├── gen_compliance.ipynb      # Jupyter notebook to generate ESG compliance results
├── criteria.csv              # CSV file with ESG compliance goals and criteria
├── img/                      # Images and logos used in the app
├── requirements.txt          # Python dependencies
└── .env                      # Environment variables (add your API keys here)
```

## Technologies

- **Streamlit**: For building the interactive dashboard and web application.
- **LangChain**: Framework for working with language models and vector databases.
- **FAISS**: Vector search engine for retrieving relevant information from reports.
- **OpenAI API**: Used for generating responses and analyzing compliance.
- **Pandas**: For data manipulation and processing.
- **Plotly**: For generating visual elements like compliance dials.
- **PyPDF2**: For extracting text from PDF files.
- **TQDM**: For progress tracking in notebooks.

## Mermaid Diagram

[![](https://mermaid.ink/img/pako:eNptVdtu4kAM_RUrUt_a_YA-rAQJ0O6WliUUrTpUaEgGMmoyE81M6LJN_309F0Lo8oAU7HMc28d2PqJM5iy6jbalfM8KqgwskpUAGJDUKEarkhsY1PUr3Nx8b2MptnzXKGq4FC0MyYzuGKTMNDVQkUNqDiXTr47vCCOx50qKignTQkweJM0BbbCkitPNOTahhoJFcLFrIfFga-xhUp6zDVUtjEh4hEe65zuXUA83pVwAZmvci8fE_U-4rkt6gHEjsoC3jLFjDAQtD5prrKZkma9vQjrr4lAzF3_i60onGL6qS05Fxlq4I7qQ72umd-usM_fwc1ZLZW6GVLMcjkFbuPc05bzrjfWusaPC9Kj3WIQS7H_yD0_mwf-FbgPcBdGqmooDzJluSoPEn2TOsLVWsGNL4nR59L-eM30pCKRlCw8kw5kwrFfkOkeP49w7zhKbJxUkwxamXsKjBZW0KfawvwYQF6hMC4_ED9GcGcXZnpbe4bA_ztqA-lCVFS08kaTJ3uxvIoPRwZ986gU1tqJaCo36zMgdllsy-NLNIyI07OoKHnEdNGwxYW0zWu9d9ut8o7_x-iA2FqabzU7RuuhKG_odsD6AX8RN8kzJjGmNwwwzXrOSC186AlyG3Xjje1qY-175OdGnsT-iE_kuSouYJWOUMCV5MITh0WfoBftjegm0sCCjP0bRzPg1re1WW9AZK3bawqjasNwuIb7nmUyYYMqaL8kI8OzXku5PAFR-SZyFaiw9eyuP4LkDj3mJGpxK_02CpcDcMkW3BoXxQ8vDgWAivyAQTnpvEC_I05vfUAautU_lhVx0fpXq5SRVrDgmyWkLg0E4ZGdHoAP8T-11Zji8uBSunrCoZ_wF0-bs0sQx6Wx2q3sJ2BgTGbbxGMAJ0a1-knhleqzgAyPtFQjcgT-j98Zrb6MiezTC5GUNco9q2eqdPVCGw3Dp3AbjxZBZY-8-8sZj0pnneF73VJiTP_Dj2PMbAQ8PUzy-ExKe_S2wCXbD2L9V3XDc9cRKl7ZYa52erH4YcTzddZ2c7v7fXoseVyK6jiqmKspz_DB-WPAqMgWr2Cq6xcecqrdVtBKfiKONkelBZNGtUQ27jpRsdkV0u8W-4L-mzjFbvJ04j1VnZTlH7af-u-s-v5__AOn1ktw?type=png)](https://mermaid.live/edit#pako:eNptVdtu4kAM_RUrUt_a_YA-rAQJ0O6WliUUrTpUaEgGMmoyE81M6LJN_309F0Lo8oAU7HMc28d2PqJM5iy6jbalfM8KqgwskpUAGJDUKEarkhsY1PUr3Nx8b2MptnzXKGq4FC0MyYzuGKTMNDVQkUNqDiXTr47vCCOx50qKignTQkweJM0BbbCkitPNOTahhoJFcLFrIfFga-xhUp6zDVUtjEh4hEe65zuXUA83pVwAZmvci8fE_U-4rkt6gHEjsoC3jLFjDAQtD5prrKZkma9vQjrr4lAzF3_i60onGL6qS05Fxlq4I7qQ72umd-usM_fwc1ZLZW6GVLMcjkFbuPc05bzrjfWusaPC9Kj3WIQS7H_yD0_mwf-FbgPcBdGqmooDzJluSoPEn2TOsLVWsGNL4nR59L-eM30pCKRlCw8kw5kwrFfkOkeP49w7zhKbJxUkwxamXsKjBZW0KfawvwYQF6hMC4_ED9GcGcXZnpbe4bA_ztqA-lCVFS08kaTJ3uxvIoPRwZ986gU1tqJaCo36zMgdllsy-NLNIyI07OoKHnEdNGwxYW0zWu9d9ut8o7_x-iA2FqabzU7RuuhKG_odsD6AX8RN8kzJjGmNwwwzXrOSC186AlyG3Xjje1qY-175OdGnsT-iE_kuSouYJWOUMCV5MITh0WfoBftjegm0sCCjP0bRzPg1re1WW9AZK3bawqjasNwuIb7nmUyYYMqaL8kI8OzXku5PAFR-SZyFaiw9eyuP4LkDj3mJGpxK_02CpcDcMkW3BoXxQ8vDgWAivyAQTnpvEC_I05vfUAautU_lhVx0fpXq5SRVrDgmyWkLg0E4ZGdHoAP8T-11Zji8uBSunrCoZ_wF0-bs0sQx6Wx2q3sJ2BgTGbbxGMAJ0a1-knhleqzgAyPtFQjcgT-j98Zrb6MiezTC5GUNco9q2eqdPVCGw3Dp3AbjxZBZY-8-8sZj0pnneF73VJiTP_Dj2PMbAQ8PUzy-ExKe_S2wCXbD2L9V3XDc9cRKl7ZYa52erH4YcTzddZ2c7v7fXoseVyK6jiqmKspz_DB-WPAqMgWr2Cq6xcecqrdVtBKfiKONkelBZNGtUQ27jpRsdkV0u8W-4L-mzjFbvJ04j1VnZTlH7af-u-s-v5__AOn1ktw)