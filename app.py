import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from curag_final_real import load_data, create_rag_chain  # แก้ชื่อไฟล์ Python ด้านบนให้ตรง เช่น boardgame_rag.py
from sentence_transformers import SentenceTransformer
SentenceTransformer("BAAI/bge-m3")
import asyncio
import nest_asyncio
nest_asyncio.apply()
st.set_page_config(page_title="CULibrary Board game", page_icon="🎲")
# เตรียมเอกสารและ chain ตอนเริ่มรัน
@st.cache_resource
def setup_chain():
    pdf_path = "Data"
    #excel_path = "Data/List_CULibraryBoardgame.xlsx"#
    excel_path = "Data/test.xlsx"
    documents = load_data(pdf_path, excel_path)
    return create_rag_chain(documents)

retrieval_chain = setup_chain()

# UI

st.title("CULibrary Board game")
st.markdown("""
ยินดีต้อนรับสู่ผู้ช่วยแนะนำบอร์ดเกมของหอสมุดจุฬาฯ!
สามารถถามคำถามเกี่ยวกับบอร์ดเกมที่มีให้บริการในห้องสมุดจุฬาฯ ได้ทุกเรื่อง
ไม่ว่าจะเป็นการแนะนำเกม กติกาเกม รีวิวเกม กลยุทธ์ในการเล่น หรือข้อมูลการยืมเกมจากห้องสมุด
""")

st.markdown("ข้อความเริ่มบทสนทนา")  # Section title in Thai

col1, col2 = st.columns(2)
with col1:
    if st.button("แนะนำบอร์ดเกมให้หน่อย"):
        user_input = "แนะนำบอร์ดเกมให้หน่อย"
with col2:
    if st.button("อยากรู้รีวิวเกม Sheriff of Nottingham "):
        user_input = "อยากรู้รีวิวเกม Sheriff of Nottingham"

col3, col4 = st.columns(2)
with col3:
    if st.button("เกม DiXit เล่นยังไง"):
        user_input = "เกม DiXit เล่นยังไง"
with col4:
    if st.button("บอร์ดเกมเปิดให้บริการวันไหนบ้าง"):
        user_input = "บอร์ดเกมเปิดให้บริการวันไหนบ้าง"

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

if user_input:
    # เพิ่มข้อความผู้ใช้ในประวัติ
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # ส่งไปที่ RAG
    response = retrieval_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history
    })

    ai_msg = response["answer"]
    st.session_state.chat_history.append(AIMessage(content=ai_msg))

    # แสดงข้อความล่าสุดทันที (optional เพราะแสดงใน loop ข้างบนแล้ว)
    st.chat_message("user").markdown(user_input)
    st.chat_message("assistant").markdown(ai_msg)

#streamlit run c:/Users/Pattranit/RAG/Data/app.py#