import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from langchain.schema import Document as LCDocument
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# Set API key
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
os.environ['OPENAI_API_KEY'] = openai_api_key

def load_data(pdf_path, excel_path):
    """Load PDFs and Excel files, converting them into LangChain documents."""
    pdf_documents = SimpleDirectoryReader(pdf_path, required_exts=[".pdf"]).load_data()
    
    df = pd.read_excel(excel_path)
    xlsx_documents = [Document(text=' '.join(map(str, row.values)), extra_info={"row_index": index}) 
                      for index, row in df.iterrows()]

    return pdf_documents + xlsx_documents

def create_rag_chain(documents):
    """Create a retrieval-augmented generation (RAG) system."""
    lc_documents = [LCDocument(page_content=doc.text, metadata=doc.metadata) for doc in documents]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(lc_documents)

    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", 
    encode_kwargs={"normalize_embeddings": True})
    vector_store = FAISS.from_documents(split_docs, embedding=embedding)
    
    model = ChatOpenAI(temperature=0.4, model='gpt-4o-mini')

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
     "You are an assistant that recommends board games based on user preferences. And You are a woman."
     "- If the user types in **Thai**, respond in Thai."
     "- If the user types in **English**, respond in English. "
     "Use the provided context, which contains board game details, to make personalized recommendations. "
     "When a user asks for a board game recommendation, follow this process:\n"
    "1. Ask **only one question at a time** to gather necessary information.\n"
    "2. First, ask how many players will be playing.\n"
    "3. After receiving that, ask how much time they have to play.\n"
    "4. Only when both number of players and available time are given, ask what type of game they prefer (e.g., strategy, party, cooperative) and ask if they have a favorite board game so that you can recommend similar games based on that. If they are not sure, tell them it’s totally fine.\n"
    "5. After collecting all 3 pieces of information, suggest board games that match their preferences.\n"
    "6. If there is no perfect match, suggest the closest available options.\n"
    "Do not recommend any board games until both number of players and available time are provided.\n"
    "Another job is to answer general questions about boardgames for users."
    "Only recommend or discuss board games that are listed in the provided Excel catalog."
    "- If a game is not listed in the catalog, politely inform the user that it is not available in the current collection."  
    "- If a game **is listed in the catalog** but the context does not contain enough information about it, you may use your own general knowledge to describe or review it."
    "- However, never generate or suggest details about games that are not in the catalog."
    "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])


    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    retriever = vector_store.as_retriever()

    #retriever_prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="chat_history"),
    #("user", "{input}"),
    #("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    #])

    #history_aware_retriever = create_history_aware_retriever(
    #    llm=model,
    #    retriever=retriever,
    #    prompt=retriever_prompt
    #)
    
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

def chat_loop(retrieval_chain):
    chat_history = []
    while True:
        user_input = input("ํYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = retrieval_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response["answer"]))
        print("Assistant:", response["answer"])

if __name__ == "__main__":
    pdf_path = r"Data/การให้บริการบอร์ดเกม.pdf"
    #excel_path = r"Data/List_CULibraryBoardgame.xlsx"#
    excel_path = r"Data/test.xlsx"

    documents = load_data(pdf_path, excel_path)
    retrieval_chain = create_rag_chain(documents)
    

    chat_loop(retrieval_chain)
