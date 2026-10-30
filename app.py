import streamlit as st
import uuid
from datetime import datetime
from pymongo import MongoClient
from agent import chat_with_agent

# 1. Page Configuration
st.set_page_config(page_title="Cognitive AI Agent", page_icon="🧠", layout="wide")

# 2. Database Connection for the UI
@st.cache_resource
def get_db_collections():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["cognitive_agent_db"]
    return db["chat_sessions"], db["episodic_logs"]

sessions_collection, episodic_collection = get_db_collections()

# 3. Initialize Current Active Session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- SIDEBAR: CHAT MANAGEMENT ---
with st.sidebar:
    st.title("🧠 Your Chats")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
        
    st.divider()
    
    # Fetch all past sessions from MongoDB
    past_sessions = list(sessions_collection.find().sort("created_at", -1))
    
    if not past_sessions:
        st.caption("No past chats found.")
    else:
        for sess in past_sessions:
            col1, col2 = st.columns([5, 1])
            
            with col1:
                # If clicked, switch the active session ID and refresh
                if st.button(sess.get("title", "Untitled"), key=f"btn_{sess['session_id']}", use_container_width=True):
                    st.session_state.session_id = sess["session_id"]
                    st.rerun()
                    
            with col2:
                # Delete chat and its corresponding audit logs
                if st.button("🗑️", key=f"del_{sess['session_id']}"):
                    sessions_collection.delete_one({"session_id": sess["session_id"]})
                    episodic_collection.delete_many({"session_id": sess["session_id"]})
                    
                    if st.session_state.session_id == sess["session_id"]:
                        st.session_state.session_id = str(uuid.uuid4())
                    st.rerun()

# --- MAIN CHAT WINDOW ---
st.title("Enterprise Cognitive Agent")

# 4. ALWAYS load the active chat history directly from MongoDB
# This guarantees the screen will never go blank if the data exists in the database.
current_logs = list(episodic_collection.find({"session_id": st.session_state.session_id}).sort("timestamp", 1))

for log in current_logs:
    with st.chat_message("user"):
        st.markdown(log["user_message"])
    if log.get("agent_response"):
        with st.chat_message("assistant"):
            st.markdown(log["agent_response"])

# 5. Input Handler
user_input = st.chat_input("Send a message to the agent...")

if user_input:
    # Draw user input instantly
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call Backend Agent
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            agent_reply = chat_with_agent(st.session_state.session_id, user_input)
            st.markdown(agent_reply)
    
    # Auto-Name New Chats
    # We know it's a brand new chat if current_logs was empty before they sent this message
    if len(current_logs) == 0:
        chat_title = user_input[:25] + "..." if len(user_input) > 25 else user_input
        sessions_collection.insert_one({
            "session_id": st.session_state.session_id,
            "title": chat_title,
            "created_at": datetime.utcnow()
        })
        # Rerun to update the sidebar. Since we load messages from the database now, it is perfectly safe.
        st.rerun()
