import pickle 
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local")
api_key=os.getenv("OPENAI_API_KEY") # specify the file manually
from PyPDF2 import PdfReader  
from streamlit_extras.add_vertical_space import add_vertical_space # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

#Sidebar contants
with st.sidebar:
    st.title('LLM Chatbot')
    st.markdown('''
    ## About  
    This app is an LLM powered chatbot built using:
    - [streamlit](https://streamlit.io/) 
    - [LangChain](https://python.langChain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM Module
    
                
    ''')

def main():
    st.header("chat with PDF")
    
    load_dotenv()
    #upload a PDF file
    pdf = st.file_uploader("Upload your PDF ",type='pdf')
    st.write(pdf.name)
    st.write(pdf)

   # st.write(pdf)
    if pdf is not None:
        
        pdf_reader = PdfReader(pdf)
            
        text = ""
        for page in pdf_reader.pages:
            text +=page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
            )    
        chunks = text_splitter.split_text(text=text)

    #Embedding
        embeddings = OpenAIEmbeddings()
        vectorStore = FAISS.from_texts(chunks,embedding=embeddings)
        store_name= pdf.name[:-4]
        with open(f"{store_name}.pkl","wb") as f:
            pickle.dump(vectorStore,f)
    



    #st.write(chunks)

    #st.write(text)   

    


 
if __name__ == '__main__':
   main()


