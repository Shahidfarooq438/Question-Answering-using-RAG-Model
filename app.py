import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import Agent,AgentExecutor,create_openai_functions_agent
from langchain import hub
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


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

# Create a conversation buffer
def get_conversation(vectorstore,key):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125",openai_api_key=key)
    retriever=vectorstore.as_retriever()
    retriever_tool=create_retriever_tool(
        retriever,
        "PDF_Search",
        "Search the uploaded PDF's for any information"
        )
    prompt=hub.pull("hwchase17/openai-functions-agent")
    tools=[retriever_tool]

    agent=create_openai_functions_agent(llm=llm, tools=tools,prompt=prompt)
    agent_executor=AgentExecutor(agent=agent, tools=tools,max_consecutive_queries=20)

    message_history = ChatMessageHistory()
    
    agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    )  
    
    return agent_with_chat_history

# Generate response based on the input prompt
def handle_userinput(user_quest):
    response=st.session_state.agent.invoke({"input":user_quest,},config={"configurable": {"session_id": "<foo>"}})
    st.session_state.chat_history =response["chat_history"]
    for i,message in enumerate(st.session_state.chat_history):
        st.write(message.content)
    


def main():
    st.set_page_config(page_title="Chat with PDFs")
    
    if "agent" not in st.session_state:
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
                st.session_state.agent=get_conversation(vectorstore,key)

if __name__ == '__main__':
    main()