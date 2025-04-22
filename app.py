import streamlit as st
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from Final_rag import load_data, create_chains, analyze_query_for_filter, filter_games, map_categories_with_llm
from langchain.schema import Document
st.set_page_config(page_title="CULibrary Board game", page_icon="🎲")
# ตั้งชื่อให้ตรงกับชื่อไฟล์ python ที่มีโค้ดของคุณ
# แทน "your_script_name" ด้วยชื่อไฟล์นั้น (เช่น rag_chatbot.py)

# โหลดข้อมูล
#PDF_PATH = r"C:\Users\Pattranit\RAG\Data\การให้บริการบอร์ดเกม.pdf"
#EXCEL_PATH = r"C:\Users\Pattranit\RAG\Data\test3.xlsx"
EXCEL_PATH = "Data/test3.xlsx"
PDF_PATH = "Data/การให้บริการบอร์ดเกม.pdf"
df, pdf_docs, vector_store = load_data(PDF_PATH, EXCEL_PATH)
document_chain, retrieval_chain = create_chains(vector_store, pdf_docs, df, None)

# UI
#logo
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600&display=swap');
            
body {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
            
.logo-container {
    text-align: center;
    margin-top: 0px !important;
}                   
/* Custom title */
.custom-title {
    font-family: 'Prompt', sans-serif;
    color: #DE5C8E;
    text-align: center;
    font-size: 42px;
    margin-bottom: 10px;
    font-weight: bold;
}

/* Welcome paragraph */
.custom-paragraph {
    font-family: 'Prompt', sans-serif;
    color: #737373;
    font-size: 16px;
    text-align: center;
    line-height: 1.6;
}
            
/* Welcome paragraph */
.eng-paragraph {
    font-family: 'Prompt', sans-serif;
    color: #737373;
    font-size: 16px;
    margin-bottom: 50px;
    text-align: center;
    line-height: 1.6;
}

/* Styled buttons - circular with pink border */
div.stButton > button {
    font-family: 'Prompt', sans-serif !important;
    background-color: white;
    width: 250px;        
    color: #737373;
    padding: 0.75em 2em;
    border: 1px solid #f2f2f2;
    border-radius: 10px;
    font-size: 16px;
    box-shadow: 1px 2px 5px rgba(0, 0, 0, 0.15);
    margin: 0.5em;
    
}

div.stButton > button:hover {
    background-color: #f2f2f2;
    color: #737373;
    cursor: pointer;
}
/* Layout for 2-column buttons */
.button-wrapper {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 2rem;
    margin-top: 1.5rem;
    padding: 0 10%;
}

.left-buttons, .right-buttons {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}
            
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
   st.image("boardgame_logo.png", width=150)
# Title
st.markdown("<div class='custom-title'>CULibrary Board_Game</div>", unsafe_allow_html=True)

# Welcome message
st.markdown("""
<div class='custom-paragraph'>
    ยินดีต้อนรับสู่บอร์ดเกมของหอสมุดกลางจุฬาลงกรณ์มหาวิทยาลัย!<br>
    ถ้าอยากรู้ว่าเกมไหนน่าเล่น กติกาเป็นยังไง เล่นยังไงให้ชนะ หรืออยากอ่านรีวิวเกมสนุกๆ <br>
    — ถามเราได้เลย! 
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='eng-paragraph'>
    Welcome to the Board Game Corner at Chulalongkorn University Library! <br>
    Not sure what to play? Want to know the rules, how to win, or read some fun game reviews? — just ask me anything! 
</div>
""", unsafe_allow_html=True)
# Section title

# สร้างสองคอลัมน์แบบ Streamlit (ใช้งานได้ดีกว่า JS)
col1, col2 = st.columns([2, 1])

starter_question = None

with col1:
    if st.button("แนะนำบอร์ดเกมให้หน่อย"):
        starter_question = "แนะนำบอร์ดเกมให้หน่อย"
    if st.button("บอร์ดเกมเปิดให้บริการวันไหนบ้าง"):
        starter_question = "บอร์ดเกมเปิดให้บริการวันไหนบ้าง"

with col2:
    if st.button("How to play Azul?"):
        starter_question = "How to play Azul?"
    if st.button("Any reviews on Sheriff of Nottingham?"):
        starter_question = "Any reviews on Sheriff of Nottingham?"


   
# เก็บ session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active_filters" not in st.session_state:
    st.session_state.active_filters = {
        "min_players": None,
        "max_players": None,
        "categories": None,
        "max_play_time": None
    }

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(msg.content)

current_datetime = datetime.now().strftime("%A, %d %B %Y %H:%M")

# กล่องสำหรับป้อนข้อความ
user_input = st.chat_input("Ask me anything about board games...")

# แสดงประวัติการสนทนา

# ถ้ามีข้อความจากผู้ใช้
if starter_question and not user_input:
    user_input = starter_question

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # 1. สร้าง filters เริ่มต้นก่อนส่งเข้า analyze_query_for_filter
    initial_filters = {
    "min_players": None,
    "max_players": None,
    "categories": None,
    "max_play_time": None
}
    filters = analyze_query_for_filter(user_input, initial_filters)
    st.write("Analyze filters from query:", filters)
# 2. ใช้ session_state เก็บ active_filters (ปลอดภัยต่อการ rerun)
    if "active_filters" not in st.session_state:
        st.session_state.active_filters = {
        "min_players": None,
        "max_players": None,
        "categories": None,
        "max_play_time": None
    }
    active_filters = st.session_state.active_filters

# 3. อัปเดต active_filters ถ้า filters มีค่าใหม่เข้ามา
    if filters:
        for key in active_filters:
            if key in filters and filters[key] is not None:
                active_filters[key] = filters[key]


    st.write("Active filters:", active_filters)
    if any(value is not None for value in active_filters.values()):
        matches = filter_games(
            df,
            min_p=active_filters["min_players"],
            max_p=active_filters["max_players"],
            categories=active_filters["categories"],
            max_play_time=active_filters["max_play_time"],
            skip_mapping=True
        )
        st.write("Filtered games:", matches)
        
        if matches.empty:
            answer = "ไม่พบเกมที่ตรงกับเงื่อนไขที่ระบุค่ะ"
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

            combined_context = matched_docs + pdf_docs
            answer = document_chain.invoke({
                "context": combined_context,
                "input": user_input,
                "chat_history": st.session_state.chat_history,
                "current_datetime": current_datetime
            })

    else:
        response = retrieval_chain.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history,
            "current_datetime": current_datetime
        })
        answer = response["answer"]

    st.session_state.chat_history.append(AIMessage(content=answer))
    with st.chat_message("assistant"):
        st.markdown(answer)
