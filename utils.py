import os 
from dotenv import dotenv_values
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

config = dotenv_values('.env')

def read_file(pathname):
    documents = os.listdir(pathname)
    text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=200)
    loaders = []
    for filename in documents:
        loaders += text_splitter.split_text(TextLoader(os.path.join(pathname, filename)).load()[0].page_content)
    loaders = [Document(page_content=loader) for loader in loaders]
    return loaders

def create_vectordb(loaders, filepath, load_from_file=False):
    embedding = OpenAIEmbeddings(openai_api_key=config["OPENAI_API_KEY"])
    if load_from_file:
        vectorstore = Chroma(persist_directory=filepath, embedding_function=embedding)
    else:
        vectorstore = Chroma.from_documents(documents=loaders, embedding=embedding, persist_directory=filepath)
        # for loader in loaders:
        #     vectorstore.add_documents(documents=loader)
    return vectorstore

def build_chatbot(uid, msg, user_history):
    loaders = read_file('./documents')
    vectordb = create_vectordb(loaders, "./vector_db", True)

    prompt = """You're a food assistance who know about บรรทัดทอง food. No matter what the language of question you must answer in Thai.
    Chat History:
    {chat_history}
    Follow Up Input: {question}"""

    prompt_template = PromptTemplate.from_template(prompt)
        
    try: 
        conversation = user_history[uid]
    except: 
        chat = ChatOpenAI(openai_api_key=config["OPENAI_API_KEY"], model="gpt-3.5-turbo-16k") 

        memory = ConversationSummaryBufferMemory(
            llm=chat,
            memory_key="chat_history", 
            return_messages=True,
            max_token_limit=2000
        )

        retriever = vectordb.as_retriever()
        conversation = ConversationalRetrievalChain.from_llm(
            llm=chat, 
            retriever=retriever, 
            memory=memory, 
            condense_question_prompt=prompt_template
        )

        user_history[uid] = conversation

    response = conversation(msg)

    print(response)

    return response["answer"]