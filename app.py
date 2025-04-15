import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from curag_final_real import load_data, create_rag_chain  # แก้ชื่อไฟล์ Python ด้านบนให้ตรง เช่น boardgame_rag.py
import asyncio
import nest_asyncio
from datetime import datetime
nest_asyncio.apply()
st.set_page_config(page_title="CULibrary Board game", page_icon="🎲")
# เตรียมเอกสารและ chain ตอนเริ่มรัน
@st.cache_resource
def setup_chain():
    pdf_path = "Data"
    #excel_path = "Data/List_CULibraryBoardgame.xlsx"
    excel_path = "Data/test3.xlsx"
    #pdf_path = r"C:\Users\Pattranit\RAG\Data"
    #excel_path = r"C:\Users\Pattranit\RAG\Data\Test.xlsx"

    documents = load_data(pdf_path, excel_path)
    return create_rag_chain(documents)

retrieval_chain = setup_chain()

# UI
# Inject custom CSS

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
    width: 230px;        
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
    if st.button("Any 4 player bluffing games you'd recommend?"):
        starter_question = "Any 4 player bluffing games you'd recommend?"

# เก็บประวัติแชท
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# แสดงประวัติแชททั้งหมด (ต้องทำ **ก่อน** input ใหม่)
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").markdown(msg.content)

# Input กล่องข้อความ
user_input = st.chat_input("Ask me anything about board games...")
if starter_question:
    current_datetime = datetime.now().strftime("%A, %d %B %Y %H:%M")
    st.session_state.chat_history.append(HumanMessage(content=starter_question))

    response = retrieval_chain.invoke({
        "input": starter_question,
        "chat_history": st.session_state.chat_history,
        "current_datetime": current_datetime
    })

    ai_msg = response["answer"]
    st.session_state.chat_history.append(AIMessage(content=ai_msg))

    st.chat_message("user").markdown(starter_question)
    st.chat_message("assistant").markdown(ai_msg)

if user_input:
    current_datetime = datetime.now().strftime("%A, %d %B %Y %H:%M")
    # เพิ่มข้อความผู้ใช้ในประวัติ
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # ส่งไปที่ RAG
    response = retrieval_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history,
        "current_datetime": current_datetime})

    ai_msg = response["answer"]
    st.session_state.chat_history.append(AIMessage(content=ai_msg))

    # แสดงข้อความล่าสุดทันที (optional เพราะแสดงใน loop ข้างบนแล้ว)
    st.chat_message("user").markdown(user_input)
    st.chat_message("assistant").markdown(ai_msg)

#streamlit run c:/Users/Pattranit/RAG/Data/app.py#
