import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


llm = OllamaLLM(model='llama3')
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

promt = PromptTemplate(
    input_variables=['chat_history','question'],
    template="Previous conversation : {chat_history}\n User:{question}\nAI:"
)

def run_chain(question):
    chat_text = "\n".join([f"{msg.type.capitalize()}:{msg.content}" for msg in st.session_state.chat_history.messages])
    response =llm.invoke(promt.format(chat_history=chat_text,question=question))

    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)

    return response


st.title("MY FIRST AI CHATBOT")
st.write("Ask me anything")

user_input=st.text_input("your question:")
if user_input : 
    response = run_chain(user_input)
    st.write(f"**You:** {user_input}")
    st.write(f"**AI:** {response}")



st.subheader("CHAT HISTORY")
for msg in st.session_state.chat_history.messages:
    st.write(f"**{msg.type.capitalize()}**:{msg.content}")