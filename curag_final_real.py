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
from datetime import datetime

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
    #xlsx_documents = [Document(text=' '.join(map(str, row.values)), extra_info={"row_index": index}) 
                      #for index, row in df.iterrows()]#
    def parse_categories(category_string):
        if pd.isna(category_string):
            return []
        return [cat.strip() for cat in category_string.split('/')]

    df['Board game Categories'] = df['Board game Categories'].apply(parse_categories)

    xlsx_documents = [
    Document(
        text=(
            f"Game name: {row['List of board games']} | "
            f"TITLE: {row['TITLE']} | "
            f"LANG: {row['LANG']} | "
            f"Total Check-out: {row['Total Check-out']} | "
            f"Board game Categories: {', '.join(row['Board game Categories'])}"
        ),
        extra_info={"row_index": index}
    ) 
    for index, row in df.iterrows()
]
    return pdf_documents + xlsx_documents

def create_rag_chain(documents):
    """Create a retrieval-augmented generation (RAG) system."""
    lc_documents = [LCDocument(page_content=doc.text, metadata=doc.metadata) for doc in documents]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(lc_documents)

    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embedding=embedding)
    
    model = ChatOpenAI(temperature=0.2, model='gpt-4o-mini')
    

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
"You are a female assistant that recommends board games available in the Excel file, "
"Use only the provided context to determine what games are available., "
"You may use your general knowledge (e.g., about number of players or game mechanics) ONLY for the games that are found in the context., "
"Do NOT mention the word 'Excel file' to users, use other words instead."
"Language Handling:, "
"- If the user types in Thai, respond in Thai., "
"- If the user types in English, respond in English., "

"Recommendation Process (Follow this step):, "
"1. Ask only one question at a time to collect preferences., "
"2. First, ask how many players will be playing., "
"3. Then ask how much time they have to play., "
"4.Once both the number of players and play time are answered:"
    "- If the user has already mentioned a favorite game or a game they like or a game type or a game they would like to play, skip asking again."
    "- If not, ask them what type of game they would like and whether they have a favorite game, in the same turn. Tell them that if they are not sure, it's fine"
        "When asking about game types, include examples such as:"
        "• strategy (e.g. managing resources, planning ahead)"
        "• party (e.g. fun and social with groups)"
        "• cooperative (e.g. players working together)"
        "• bluffing/deception (e.g. hidden roles)"
        "• word games, deduction, storytelling, and more."

#"Only after steps 2, 3, and 4 (if applicable), recommend games."#
"5. Recommendation:"  
"- If the user provides a favorite game, recommend other board games from the Excel file that share at least one category listed in the 'Board game Categories' column."  
"- Do NOT recommend the same game."  
"- Check all available games in the Excel file and recommend up to 3 of the most relevant ones."  
"- For each recommendation, provide a brief explanation of why the game is similar (e.g., shared category, similar playstyle)."  
"- If no games share any category with the favorite game, you may use general knowledge to recommend similar games — but only from those found in the Excel file."  
#"- If no suitable games can be found, respond with: 'Sorry, we don’t currently have a game that matches your preferences, but feel free to ask about other options!'"  
"- If the user provides a type of game, recommend games that match from the Excel file."  
"- If there is no perfect match, suggest the closest available options from the Excel file only."



"Answering Availability Questions:, "
"- If the user asks whether a specific game is available (e.g., 'Do you have Dixit?'), carefully check the context., "
"- Try to match the game name even if it's written differently (e.g., 'DiXit', 'Avalon', '7Wonders'). Be flexible with spelling and capitalization., "
"- If the game is found, confirm that it is available. If not, clearly say it is not available in the the Excel file, "

"Critical Rules:, "
"- Recommend ONLY games listed in the Excel file, "
"- If a game is the Excel file but lacks full context, you may use general knowledge to explain it., "
"- NEVER use or mention any game that is not in the Excel file, "
"- Violating this rule is considered a critical failure., "

"Additional Task:, "
"- You can also answer general questions about board games, but only about games found in the Excel file., "
"Today is {current_datetime},. Use this current date and time to help answer questions such as whether the board game zone is open."

    

    "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])


    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 30}) #*20 กำลังดี#

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

def chat_loop(retrieval_chain):
    current_datetime = datetime.now().strftime("%A, %d %B %Y %H:%M")
    chat_history = []
    while True:
        user_input = input("ํYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = retrieval_chain.invoke({
            "input": user_input,
            "chat_history": chat_history,
            "current_datetime": current_datetime
        })
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response["answer"]))
        #print("\n🔍 Context used:\n", response.get("context", "No context"))#
        print("Assistant:", response["answer"])

if __name__ == "__main__":
    pdf_path = r"C:\Users\Pattranit\RAG\Data"
    #excel_path = r"C:\Users\Pattranit\RAG\Data\List_CULibraryBoardgame.xlsx"#
    excel_path = r"C:\Users\Pattranit\RAG\Data\Test.xlsx"

    documents = load_data(pdf_path, excel_path)
    retrieval_chain = create_rag_chain(documents)
    

    chat_loop(retrieval_chain)
