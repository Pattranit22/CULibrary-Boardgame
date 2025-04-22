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
from langchain.schema import Document as LCDocument
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from datetime import datetime
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
# Set API key
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
os.environ['OPENAI_API_KEY'] = openai_api_key
model = ChatOpenAI(temperature=0.2, model='gpt-4o-mini')

def load_data(pdf_path, excel_path):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = [page.page_content for page in pages]
    pdf_docs = text_splitter.create_documents(texts)

    # Load Excel & Embeddings
    df = pd.read_excel(excel_path)
    excel_docs = []
    for index, row in df.iterrows():
        raw_categories = row['Board game Categories']
        category_list = [cat.strip() for cat in raw_categories.split('/') if cat.strip()]
        content = f"""
        Game: {row['List of board games']}
        Title: {row['TITLE']}
        Language: {row['LANG']}
        Total Check-out: {row['Total Check-out']}
        Min Players: {row['Min Players']}
        Max Players: {row['Max Players']}
        Min Play Time: {row['Minimum_Playing_Time']}
        Average Playing Time: {row['Average Playing Time']} mins
        Recommended Age: {row['Recommended Player Age']}+
        Categories: {', '.join(category_list)}
        """
        excel_docs.append(Document(page_content=content, metadata={"index": index}))
    
    all_docs = pdf_docs + excel_docs
    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(all_docs, embedding=embedding)

    return df, pdf_docs, vector_store

def create_chains(vector_store, pdf_docs, df, model):
    model = ChatOpenAI(temperature=0.2, model='gpt-4o-mini')
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are a female assistant board game assistant. The games below have already been filtered to match the user's conditions.

1. Ask only one question at a time to collect preferences.

2. Ask how many players will be playing — but only if this hasn't already been mentioned.

3. Ask how much time they have to play — but only if it hasn't already been mentioned.

4. Once you have both the number of players and play time:
   - Check if the user has mentioned a **game type** (e.g. strategy, party, cooperative, etc.) OR a **favorite game**.
   - If the user has NOT mentioned either of them, you MUST ask **both in the same turn**.
     > What type of game do you want to play ? (e.g., strategy, party, bluffing, cooperative, etc.) And do you have a favorite board game?

5. If the user has already mentioned either a game type or a favorite game, skip the question and proceed to recommendations.

Only proceed to recommendations after collecting all 3: number of players, play time, and either a game type or favorite game.
         
Your task:
- Recommend 3 **different** board games that best match the user's preferences based on the context.
- For each recommended game, include:
  - Game name
  - Number of players
  - Categories (e.g., Party, Strategy)
  - Estimated play time
  - A brief reason why it fits the user's request.

Then, list up to 20 remaining game names from the context, excluding the 3 you just recommended. 
Do not invent or add games that are not listed in the context.
         
Game list guidelines: 
- Do **not** list the same game more than once.
- Do **not** skip, summarize, interpret, or remove any entries.
- Each game should be listed on its own line.
- Do **not** include any explanation or formatting around the list.
         
"Language Handling:, "
- Always respond in the same language the user uses. 
- If the user uses Thai, ask and answer questions in Thai. 
- If the user uses English, continue in English.

Use natural, friendly language when asking for preferences.

"Answering Availability Questions:, "
"- If the user asks whether a specific game is available (e.g., 'Do you have Dixit?'), carefully check the context., "
"- Try to match the game name even if it's written differently (e.g., 'DiXit', 'Avalon', '7Wonders'). Be flexible with spelling and capitalization., "
"- If the game is found, confirm that it is available. If not, clearly say it is not available in the the Excel file, "

"Critical Rules:, "
"- Recommend ONLY games listed in the Excel file, "
"- If a game is the Excel file but lacks full context, you may use general knowledge to explain it., "
"- NEVER use or mention any game that is not in the Excel file, "
"- Violating this rule is considered a critical failure. "

 "Additional Task:"
"Today is {{current_datetime}}. Use this current date and time to help answer questions about the opening hours of the board game zone. By following this information:"
"Regular Board Game Service Hours"
"During the Academic Semester (August–December, January–May):"
    "Monday to Friday: 3:00 PM – 8:00 PM"
    "Saturday and Sunday: Closed"
"During the Semester Break (June–July):"
    "Monday to Friday: 3:00 PM – 5:00 PM"
    "Saturday, Sunday, and Public Holidays: Closed"
"Answer briefly and clearly unless the user asks for more detail."
"Be sure to remind the user to verify the information with the official announcements from Chulalongkorn University Library, as schedules may change."

"""
"Context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 120, "lambda_mult": 0.5}
    )
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return document_chain, retrieval_chain

def map_categories_with_llm(user_categories, available_categories):
    model = ChatOpenAI(temperature=0.2, model='gpt-4o-mini')
    category_prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an assistant that maps user-input board game categories to a list of allowed categories.

Your task:
- You are given two lists:
    - `user_categories`: a list of categories from user input (which may be misspelled or written differently).
    - `available_categories`: the valid list of categories used in the Excel file (database).

Instructions:
- For each category in `user_categories`, match it to the **closest** valid category from `available_categories` based on meaning and spelling.
- Only include categories that have a **clear and direct match**. Do **not** guess or include extra related categories.
- Do **not** add more than what the user mentioned. Only return what maps directly.

If a user category cannot be matched to anything, ignore it.

Example output format:
["City building", "Strategy"]
        """),
        ("user", "User categories: {user_categories}\nAvailable: {available_categories}")
    ])

    chain = category_prompt | model | StrOutputParser()
    try:
        return json.loads(chain.invoke({
            "user_categories": user_categories,
            "available_categories": list(set(cat for sublist in df["Board game Categories"].str.split("/") for cat in sublist if cat)),
        }))
    except:
        return []
    
def analyze_query_for_filter(user_input, model):
    model = ChatOpenAI(temperature=0.2, model='gpt-4o-mini')
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a function that extracts conditions from user queries about board games.

Your task is to extract the following information from the user's question:
- Number of players (min_players, max_players)
- Game categories (as a list of strings)
- Maximum play time (in minutes)

Guidelines:
- Be flexible with spelling and capitalization.
- If the query includes words that sound like board game types (e.g., "party", "strategy", "tile laying", "deck building", etc.), treat them as categories.
- If the query is only a number (e.g. "12"), assume it refers to the number of players.
- If a value is not mentioned, return null for that field.

Return a JSON object in this format:
{{
  "min_players": 4,
  "max_players": 8,
  "categories": ["Party", "Strategy"],
  "max_play_time": 30
}}
        """),
        ("user", "{input}")
    ])

    chain = prompt | model | StrOutputParser()

    try:
        result = chain.invoke({"input": user_input})
        return json.loads(result)
    except:
        return None
    
def filter_games(df, min_p=None, max_p=None, categories=None, max_play_time=None):

    filtered = df

    if min_p is not None:
        if max_p is not None:
            filtered = filtered[(filtered["Min Players"] <= max_p) & (filtered["Max Players"] >= min_p)]
        else:
            filtered = filtered[filtered["Max Players"] >= min_p]

    if categories is not None:
        mapped = map_categories_with_llm(categories, df["Board game Categories"])
        if mapped:
            mapped = [cat.lower() for cat in mapped]
            def has_category(cell):
                game_categories = [c.strip().lower() for c in str(cell).split('/')]
                return any(cat.lower() in game_categories for cat in mapped)
            filtered = filtered[filtered["Board game Categories"].apply(has_category)]


    if max_play_time is not None:
        filtered = filtered[(filtered["Minimum_Playing_Time"] <= max_play_time) | (filtered["Average Playing Time"] <= max_play_time)]

    return filtered    



def chat_loop(document_chain, retrieval_chain, df, pdf_file):
    current_datetime = datetime.now().strftime("%A, %d %B %Y %H:%M")
    chat_history = []

    active_filters = {
        "min_players": None,
        "max_players": None,
        "categories": None,
        "max_play_time": None
        }

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        filters = analyze_query_for_filter(user_input, model)
        if 'active_filters' not in locals():
            active_filters = {
                "min_players": None,
                "max_players": None,
                "categories": None,
                "max_play_time": None
            }

        if filters:
            for key in active_filters:
        # ต้องเช็กว่า key นั้นมีอยู่ใน filters และไม่ใช่ None ถึงจะอัปเดต
                if key in filters and filters[key] is not None:
                    active_filters[key] = filters[key]  # เก็บค่าที่ user พูดไว้

            print("🔁 Filters after merge:", active_filters)

            # ถ้ามี filter อะไรสักอย่าง ก็ไปกรองข้อมูล
            if any(value is not None for value in active_filters.values()):
                matches = filter_games(
                    df,
                    min_p=active_filters["min_players"],
                    max_p=active_filters["max_players"],
                    categories=active_filters["categories"],
                    max_play_time=active_filters["max_play_time"]
                )

                if matches.empty:
                    print("Assistant: ไม่พบเกมที่ตรงกับเงื่อนไขที่ระบุค่ะ")
                    continue
                else:
                    matched_docs = []
                    for idx, row in matches.iterrows():
                        raw_categories = row['Board game Categories']
                        category_list = [cat.strip() for cat in raw_categories.split('/') if cat.strip()]
                        content = f"""
                    Game: {row['List of board games']}
                    Title: {row['TITLE']}
                    Language: {row['LANG']}
                    Total Check-out: {row['Total Check-out']}
                    Min Players: {row['Min Players']}
                    Max Players: {row['Max Players']}
                    Min Play Time: {row['Minimum_Playing_Time']}
                    Average Playing Time: {row['Average Playing Time']} mins
                    Recommended Age: {row['Recommended Player Age']}+
                    Categories: {', '.join(category_list)}
                    """
                        matched_docs.append(Document(page_content=content))
            
                    combined_context = matched_docs + pdf_file
                    answer = document_chain.invoke({
                    "context": combined_context,
                    "input": user_input,
                    "chat_history": chat_history,
                    "current_datetime": current_datetime
                    })

                    chat_history.append(HumanMessage(content=user_input))
                    chat_history.append(AIMessage(content=answer))
                    print("Assistant:", answer)
                    continue

        # fallback retriever
        response = retrieval_chain.invoke({
            "input": user_input,
            "chat_history": chat_history,
            "current_datetime": current_datetime
        })
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response["answer"]))
        print("Assistant:", response["answer"])

if __name__ == "__main__":
    pdf_path = r"C:\Users\Pattranit\RAG\Data\การให้บริการบอร์ดเกม.pdf"
    excel_path = r"C:\Users\Pattranit\RAG\Data\test3.xlsx"

    df, pdf_docs, vector_store = load_data(pdf_path, excel_path)
    model = ChatOpenAI(temperature=0.2, model='gpt-4o-mini')

    document_chain, retrieval_chain = create_chains(vector_store, pdf_docs, df, model)

    chat_loop(document_chain, retrieval_chain, df, pdf_docs)
