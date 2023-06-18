import os
import streamlit as st #for UI

from langchain.llms import OpenAI #OpenAI's main LLM
from langchain.embeddings import OpenAIEmbeddings

from langchain.document_loaders import PyPDFLoader 
from langchain.vectorstores import Chroma #chroma as the vector store

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

os.environ['OPENAI_API_KEY'] = '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'

llm = OpenAI(temperature=0.9, verbose=True)
embeddings = OpenAIEmbeddings()

data = PyPDFLoader('Medical.pdf')
pages = data.load_and_split()
store = Chroma.from_documents(pages, embeddings, collection_name='medical_report')


st.title('Personal Healthbot👨‍⚕️💊')

prompt = st.text_input('Input your query here: ')


if prompt:
    response = llm(prompt)
    #streamlit expander
    with st.expander('GPT\'s Version'):
        st.write(response)
    
    #streamlit expander
    with st.expander('Based on your medical history'):
        search = store.similarity_search_with_score(prompt)
        
        st.write(search[0][0].page_content)
