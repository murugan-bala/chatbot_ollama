import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory


# Set up Ollama with DeepSeek
llm = OllamaLLM(model="deepseek-r1:1.5b")  # gemma3:27b   deepseek-r1:1.5b  deepseek-r1:14b vicuna:13b #gpt-oss:20b

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create the chain
chain = prompt | llm

# Wrap with message history
def get_session_history(session_id: str):
    if session_id not in st.session_state:
        st.session_state[session_id] = InMemoryChatMessageHistory()
    return st.session_state[session_id]

chat_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Streamlit UI
st.set_page_config(page_title="EATON Chatbot", layout="centered")
st.title("🤖 EATON- AI Assistant ")

user_input = st.chat_input("Ask me anything...")

# Display chat history
session_id = "chat1"
history = get_session_history(session_id)
for msg in history.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

# Handle user input
if user_input:
    st.chat_message("user").write(user_input)
    with st.spinner('🤔 Model is thinking...'):
        response = chat_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
    st.chat_message("assistant").write(response)
