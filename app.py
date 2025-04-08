import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from curag_final import load_data, create_rag_chain  # แก้ชื่อไฟล์ Python ด้านบนให้ตรง เช่น boardgame_rag.py

st.set_page_config(page_title="Board Game Recommender", page_icon="🎲")
# เตรียมเอกสารและ chain ตอนเริ่มรัน
@st.cache_resource
def setup_chain():
    pdf_path = r"C:\Users\Pattranit\RAG\Data"
    excel_path = r"C:\Users\Pattranit\RAG\Data\Test.xlsx"
    documents = load_data(pdf_path, excel_path)
    return create_rag_chain(documents)

retrieval_chain = setup_chain()

# UI

st.title("🎲 Board Game Chat Assistant")

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
