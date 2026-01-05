import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
####
import os
from dotenv import load_dotenv
load_dotenv()
hf_token=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-miniLM-L6-v2")

#streamlit interface
st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDFs, chat with the content, and maintain chat history!")
#input groq api key
api_key = st.text_input("Enter your Groq API Key:", type="password")

if api_key:
    llm=ChatGroq(api_key=api_key, model_name='llama-3.3-70b-versatile')
    #chat interface
    session_id = st.text_input("Session_ID",value="default_session")
    #history manage

    if 'store' not in st.session_state:
        st.session_state.store={}
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    #process uploaded files
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f'./temp.pdf'
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
        #split,embedding the documents
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vectorstore.as_retriever()

        #prompt for contextualizing questions and answering based on retrieved context
        contextualize_q_system_prompt = (
            'Given a chat history and the latest user question which might reference context in the chat history, '
            'formulate a standalone application question which can be understood without the chat history. '
            'Do not answer the question, just reformulate it if needed and otherwise return this.'
         )
        contextualize_q_prompt=ChatPromptTemplate([
            ('system',contextualize_q_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human','{input}')
        ])

        history_aware_retriever=(
            {
            'input':lambda x:x["input"],
            'chat_history':lambda x: x["chat_history"]
            }
            |contextualize_q_prompt
            |llm
            |StrOutputParser()
            |retriever
            
            )
        
        system_prompt = (
        'You are an assistant for the question answer task. Use the following pieces of retrieved context to answer the question. '
        'If you do not know the answer, say that you do not know. Use three sentences maximum and keep the answer concise.'
        '{context}'
         )
        qa_prompt=ChatPromptTemplate([
            ('system',system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human','{input}')
        ])
        qa_chain=(
            {
                'context': history_aware_retriever,
                'input': lambda x:x["input"],
                'chat_history': lambda x:x["chat_history"],
            }
            |qa_prompt
            |llm
            |StrOutputParser()

        )

        def get_session_history(session_id:str):
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
    
        conversational_rag_chain=RunnableWithMessageHistory(
            qa_chain,
            get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history'
           
        )

        user_input=st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke({
                'input':user_input,},
                config={
                    'configurable':{'session_id':session_id}
                }
            )
            st.write(st.session_state.store)
            st.success(f"Assistant: {response}")
            st.write("chat_history:",session_history.messages)
else:
    st.warning("Please enter your Groq API Key to proceed.")


