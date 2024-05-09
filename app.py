import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Read the pdf uploaded by the user.
def get_text(docs):
    text=""
    for doc in docs:
        doc_reader=PdfReader(doc)
        for page in doc_reader.pages:
            text+=page.extract_text()
    return text

# Preprocessing the extracted text from the PDFs.
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200
        )
    chunks=splitter.split_text(text)
    return chunks

# Creating embeddings of the processed text.Store in the vector database
def get_vectorstore(text_chunk,key):
    embeddings=OpenAIEmbeddings(openai_api_key=key)
    vectorstore=FAISS.from_texts(texts=text_chunk,embedding=embeddings)
    return vectorstore

# Create a conversation buffer and retrieve chain
def get_conversation_chain(vectorstore,key):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125",openai_api_key=key)
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

# Generate response based on the input prompt
def handle_userinput(user_quest):
    response=st.session_state.conversation({'question':user_quest})
    st.session_state.chat_history =response['chat_history']
    for i,message in enumerate(st.session_state.chat_history):
        st.write(message.content)


def main():
    st.set_page_config(page_title="Chat with PDFs")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=None

    st.header("Chat with PDFs")
    user_quest=st.text_input("Ask any Query about your documents")
    if user_quest:
        handle_userinput(user_quest)

    with st.sidebar:
        key=st.text_input("Enter your OpenAI API key",placeholder="OpenAI_API_Key",type="password")

        st.subheader("Documents")
        docs=st.file_uploader("Upload your PDFs here and click 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text=get_text(docs)
                text_chunk=get_text_chunks(raw_text)
                vectorstore=get_vectorstore(text_chunk,key=key)
                st.session_state.conversation=get_conversation_chain(vectorstore,key)

if __name__ == '__main__':
    main()