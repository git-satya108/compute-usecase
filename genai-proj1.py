import os
import openai
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from pypdf import PdfReader
import docx2txt
import pandas as pd
import numpy as np
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load OpenAI API key
load_dotenv(find_dotenv(), override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    return docx2txt.process(file)

# Function to extract text from TXT
def extract_text_from_txt(file):
    return file.read().decode('utf-8')

# Function to extract text and data from Excel
def extract_text_and_data_from_excel(file):
    df = pd.read_excel(file)
    return df.to_string(), df

# Load documents in different formats
def load_document(uploaded_file):
    name, extension = os.path.splitext(uploaded_file.name)
    if extension == '.pdf':
        return extract_text_from_pdf(uploaded_file), None
    elif extension == '.docx':
        return extract_text_from_docx(uploaded_file), None
    elif extension == '.txt':
        return extract_text_from_txt(uploaded_file), None
    elif extension == '.xlsx':
        text, df = extract_text_and_data_from_excel(uploaded_file)
        return text, df
    else:
        st.error('Document format not supported!')
        return None, None

# Function to break content into chunks
def break_into_chunks(text, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Initialize embeddings and FAISS
def initialize_faiss_index(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# Function to create summary report
def create_summary_report(df):
    report = df.describe().to_string()
    return report

# Function to generate a chart
def generate_chart(df, chart_type, x_col, y_col):
    if chart_type == 'bar':
        fig = px.bar(df, x=x_col, y=y_col)
    elif chart_type == 'line':
        fig = px.line(df, x=x_col, y=y_col)
    elif chart_type == 'scatter':
        fig = px.scatter(df, x=x_col, y=y_col)
    elif chart_type == 'histogram':
        fig = px.histogram(df, x=x_col, y=y_col)
    st.plotly_chart(fig)

# Function to perform VLOOKUP
def perform_vlookup(df, lookup_value, lookup_column, return_column):
    result = df[df[lookup_column] == lookup_value][return_column]
    return result.to_string(index=False)

# Function to handle mathematical computations
def perform_math_operation(df, operation):
    result = eval(f"df.{operation}")
    return result.to_string()

# Streamlit app layout
st.title("Contextual Chat Assistant")

# Document upload
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents (PDF, DOCX, TXT, XLSX)", type=['pdf', 'docx', 'txt', 'xlsx'], accept_multiple_files=True)

all_text = ''
dfs = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_text, df = load_document(uploaded_file)
        if file_text:
            all_text += file_text + ' '
        if df is not None:
            dfs.append(df)
    chunks = break_into_chunks(all_text)

    # Embedding and storing chunks
    if st.button("Add Data"):
        vectorstore = initialize_faiss_index(chunks)
        st.session_state['vectorstore'] = vectorstore
        st.write("Data has been added to the vector store.")

# Main input for chat
user_input = st.text_input("Ask something about the uploaded documents or request an analysis:")
if st.button("Send"):
    if user_input.lower().startswith('summary report'):
        for df in dfs:
            summary = create_summary_report(df)
            st.text(summary)
    elif user_input.lower().startswith('chart'):
        # Expected format: "chart bar x_col y_col"
        _, chart_type, x_col, y_col = user_input.split()
        for df in dfs:
            if x_col in df.columns and y_col in df.columns:
                generate_chart(df, chart_type, x_col, y_col)
    elif user_input.lower().startswith('vlookup'):
        # Expected format: "vlookup lookup_value lookup_column return_column"
        _, lookup_value, lookup_column, return_column = user_input.split()
        for df in dfs:
            if lookup_column in df.columns and return_column in df.columns:
                result = perform_vlookup(df, lookup_value, lookup_column, return_column)
                st.text(result)
    elif user_input.lower().startswith('math'):
        # Expected format: "math operation"
        _, operation = user_input.split(maxsplit=1)
        for df in dfs:
            try:
                result = perform_math_operation(df, operation)
                st.text(result)
            except Exception as e:
                st.error(f"Error performing operation: {e}")
    else:
        response = chat_with_assistant(user_input)
        st.write(response)
