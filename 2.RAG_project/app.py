from search import RAGSearch
import streamlit as st

def handle_smalltalk(query: str) -> str:
    q = query.lower().strip()

    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    farewells = ["bye", "goodbye", "see you", "take care"]
    thanks = ["thanks", "thank you", "thx", "appreciate"]

    if any(g in q for g in greetings):
        return "👋 Hi there! How can I help you with the FernTel IP4 or IP160 telephone manual today?"
    elif any(t in q for t in thanks):
        return "😊 You're very welcome! Glad I could help."
    elif any(f in q for f in farewells):
        return "👋 Goodbye! Have a great day!"
    else:
        return ""

# Streamlit UI
def Home_page(rag_search):
    
    st.set_page_config(page_title="EATON Chatbot", layout="centered")
    st.title("🤖 EATON- AI Assistant ")

    user_input = st.chat_input("Ask me anything about FernTel IP4 / IP160 telephone ...")
    if user_input:
        st.chat_message("user").write(user_input)
        with st.spinner('🤔 Model is thinking...'):

            # --- Step 1: handle greetings / smalltalk
            smalltalk_reply = handle_smalltalk(user_input)
            if smalltalk_reply:
                response = smalltalk_reply
            else:
                response = rag_search.search_and_summarize(user_input, top_k=3)
        st.chat_message("assistant").write(response)

# Example usage
if __name__ == "__main__":
   
    rag_search = RAGSearch()
    Home_page(rag_search)