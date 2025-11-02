import os
from dotenv import load_dotenv
# from src.vectorstore import FaissVectorStore
from vectorstore import FaissVectorStore
from langchain_ollama import OllamaLLM
import random

# --- List of fallback messages ---
FALLBACK_MESSAGES = [
    # Professional & Helpful
    "I’m here to assist you with the FernTel IP4 / IP160 telephone. You can ask about installation, configuration menus, or device troubleshooting.",
    "Let’s focus on the FernTel IP4 / IP160 telephone. Try asking about setup steps, network settings, or how to reset the device.",
    "I specialize in the FernTel IP4 / IP160 system. Please ask about setup, configuration, or common operational issues.",
    
    # Friendly & Conversational
    "Hey there! I can help you with the FernTel IP4 / IP160 phone — setup, configuration, or fixing issues. What would you like to know?",
    "I’m your FernTel IP4 / IP160 assistant! Ask me how to set up, configure, or troubleshoot your telephone. 📞",
    "Need help with your FernTel IP4 / IP160? I can walk you through setup, settings, or troubleshooting.",
    
    # Guiding & Supportive
    "It looks like I didn’t quite catch that. Try asking something like “How do I reset the FernTel IP4?” or “How do I configure the SIP settings?”",
    "I couldn’t find a match for that query. You can ask questions such as setup instructions, network configuration, or error troubleshooting.",
    "Please try rephrasing your question. I can help you with things like resetting, configuring, or maintaining your FernTel IP4 / IP160.",
    
    # Assistant-Style Default
    "I'm your Eaton AI Assistant for the FernTel IP4 / IP160. You can ask me about setup, configuration options, or troubleshooting steps.",
    "I didn’t find relevant information for that. Let’s focus on the FernTel IP4 / IP160 telephone — ask about setup, configuration, or maintenance.",
    "Sorry, I couldn’t find a related answer. I can help with topics like setup, registration, or resetting the FernTel IP4 / IP160."
]

# --- Function to get a random fallback message ---
def get_random_fallback():
    return random.choice(FALLBACK_MESSAGES)

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gpt-oss:20b"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        print(f'faiss_path:{faiss_path}')
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            # from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        self.llm = OllamaLLM(model=llm_model)
        print(f"[INFO] Open source GPT LLM initialized: {llm_model}")


    def search_and_summarize(self, query: str, top_k: int = 3) -> str:
        # Threshold
        THRESHOLD =1.0
        results = self.vectorstore.query(query, top_k=top_k)
        # print("====================================================================================================")
        # print(f'result : {results}')
        # print("====================================================================================================")
        # Check minimum distance among results
        min_distance = min(float(r['distance']) for r in results)
        print(f'Distance : {min_distance}')
        if min_distance > THRESHOLD:
            #print("I'm here to help with the FernTel IP4 / IP160 telephone. Try asking about setup, configuration, or troubleshooting.🙂")
            #return "I'm here to help with the FernTel IP4 / IP160 telephone. Try asking about setup, configuration, or troubleshooting.🙂"
            return get_random_fallback()
        else:
            texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
            context = "\n\n".join(texts)
            if not context:
                return "No relevant documents found."
            prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
            response = self.llm.invoke([prompt])
            return response

# Example usage
if __name__ == "__main__":
    ''' 
    Sample questions?
    # How to do Reset configuration ?
    
    # what are all the Warning and safety instructions ?
    
    # what is FernTel IP4 / IP160 telephone ? can u pleaase describe the Device?
    '''
    rag_search = RAGSearch()
    query = "what is FernTel IP4 / IP160 telephone ? can u pleaase describe the Device?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
    
    

    
