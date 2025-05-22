 
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local")
api_key=os.getenv("OPENAI_API_KEY") # specify the file manually
from PyPDF2 import PdfReader  
from streamlit_extras.add_vertical_space import add_vertical_space # type: ignore
from langchain.text_splitter import RecuvsiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer




def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecuvsiveCharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    return chunks

# Optional: Detect and display common themes
themes = detect_themes(text_chunks, num_clusters=5, top_keywords=5)
st.subheader("🔍 Detected Themes from Documents:")
for cluster_id, keywords in themes:
    st.markdown(f"**Theme {cluster_id + 1}:** {', '.join(keywords)}")



def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


    if "source_documents" in response:
        st.subheader("Sources:")
        for doc in response["source_documents"]:
            st.markdown(f"- {doc.metadata.get('source', 'Unknown Source')}")

def detect_themes(text_chunks, num_clusters=5, top_keywords=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)

    clustered_texts = {i: [] for i in range(num_clusters)}
    for idx, label in enumerate(kmeans.labels_):
        clustered_texts[label].append(text_chunks[idx])

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    themes = []
    for cluster_id, texts in clustered_texts.items():
        if not texts:
            continue
        X = vectorizer.fit_transform(texts)
        scores = X.sum(axis=0).A1
        keywords = vectorizer.get_feature_names_out()
        sorted_indices = scores.argsort()[::-1]
        top_terms = [keywords[i] for i in sorted_indices[:top_keywords]]
        themes.append((cluster_id, top_terms))
    return themes

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
