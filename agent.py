from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from langchain_core.output_parsers import StrOutputParser
import redis
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from datetime import datetime
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.pydantic_v1 import BaseModel, Field
load_dotenv()
# 1. Define the exact shape the data must take
class FactSchema(BaseModel):
    topic: str = Field(description="The core subject (e.g., 'Favorite Database')")
    fact: str = Field(description="The detailed fact to save (e.g., 'User prefers MongoDB')")


# 2. Initialize Redis (Short-Term Memory)
redis_client = redis.Redis(
    host='localhost', 
    port=6379, 
    db=0, 
    decode_responses=True
)

# 3. Initialize MongoDB (Long-Term/Episodic Memory)
mongo_uri = os.getenv("MONGO_URI")
mongo_client = MongoClient(mongo_uri)
db = mongo_client["cognitive_agent_db"]
long_term_collection = db["long_term_memory"]
episodic_collection = db["episodic_logs"]

# 4. Initialize Groq LLM (Processing Engine)
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    temperature=0, 
    groq_api_key=groq_api_key, 
    model_name="openai/gpt-oss-20b")
    # --- PROCEDURAL MEMORY (CHROMADB) ---

# 1. Load a lightweight local AI model to translate text into mathematical vectors
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Initialize ChromaDB (This will create a physical folder named 'chroma_db' in your workspace)
procedural_memory = Chroma(
    collection_name="agent_rules",
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
) # You can swap this for any supported Groq model

@tool(args_schema=FactSchema)
def extract_and_save_fact(topic: str, fact: str):
    """
    Use this tool ONLY when the user mentions a persistent fact, preference, 
    personal detail, or project requirement that should be remembered forever.
    Do NOT use this for general conversation or greetings.
    """
    save_long_term_fact(topic, fact)
    return f"Successfully saved fact about {topic}."
llm_with_tools = llm.bind_tools([extract_and_save_fact])
prompt = ChatPromptTemplate.from_messages([

("system" ,"""You are an enterprise cognitive agent. Use tools to manage long-term facts.
User Profile:
{user_profile}

Procedural Rules (Current active rules to apply):
{procedural_rules}

CRITICAL DIRECTIVE: If the user asks to change, update, or remove one of these rules, you must politely acknowledge and accept the new instruction. Do not refuse. The system will process their update in the background.

If the user states a new permanent fact or preference, save it using your tool."""),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("human", "{question}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 2. Define the Agent and Executor
tools = [extract_and_save_fact]
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
def get_redis_history(session_id: str):
    return RedisChatMessageHistory(session_id, url="redis://localhost:6379/0")
agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_redis_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)
def save_episodic_log(session_id, user_message, agent_response):
    """Saves the exact interaction to MongoDB for chronological history."""
    log_entry = {
        "session_id": session_id,
        "timestamp": datetime.utcnow(),
        "user_message": user_message,
        "agent_response": agent_response
    }
    episodic_collection.insert_one(log_entry)
    print(f"Episodic memory saved for session: {session_id}")
def get_user_profile():
    """Fetches all long-term facts from MongoDB to build the agent's core knowledge."""
    facts = long_term_collection.find({}, {"_id": 0, "topic": 1, "fact": 1})
    
    profile_lines = []
    for entry in facts:
        profile_lines.append(f"- {entry['topic']}: {entry['fact']}")
        
    if not profile_lines:
        return "No long-term facts known yet."
        
    return "\n".join(profile_lines)
def get_relevant_rules(user_input: str, k: int = 1):
    """Fetches the most relevant procedural rules based on vector similarity."""
    results = procedural_memory.similarity_search(user_input, k=k)
    
    if not results:
        return "No specific procedural rules apply."
        
    rules = [doc.page_content for doc in results]
    return "\n".join(rules)
router_prompt = PromptTemplate.from_template(
    """Classify the user's input into one of two categories:
    CASUAL: Greetings, thank yous, conversational filler, or simple agreements (e.g., "hello", "thanks", "how are you", "ok").
    TASK: Questions, coding requests, factual statements, or anything requiring thought, memory, or tools (e.g., "what is python", "my name is Damini", "write code").
    
    User input: {question}
    
    Respond with exactly one word: CASUAL or TASK."""
)
intent_router = router_prompt | llm | StrOutputParser()
casual_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly AI. Respond concisely to the user's casual chat or greeting."),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("human", "{question}")
])
casual_chain = casual_prompt | llm | StrOutputParser()
casual_with_memory = RunnableWithMessageHistory(
    casual_chain,
    get_redis_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

def save_long_term_fact(topic: str, fact: str):
    """Forces MongoDB to overwrite old facts with the same topic instead of duplicating."""
    
    normalized_topic = topic.strip().lower()
    db["long_term_memory"].update_one(
        {"topic": normalized_topic},
        {"$set": {"fact": fact, "last_updated": datetime.utcnow()}},
        upsert=True
    )
def save_procedural_rule(rule_text: str):
    """Saves an operational guideline or workflow to the vector database."""
    procedural_memory.add_texts(texts=[rule_text])
    print(f"Procedural rule saved: '{rule_text[:50]}...'")

def chat_with_agent(session_id: str, user_input: str):
    """Evaluates intent and routes with Graceful Degradation (Circuit Breaker)."""
    
    try:
        intent = intent_router.invoke({"question": user_input}).strip().upper()
        print(f"\n[System] Intent Detected: {intent}")
        
        if "CASUAL" in intent:
            agent_reply = casual_with_memory.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": session_id}}
            )
        else:
            current_profile = get_user_profile()
            active_rules = get_relevant_rules(user_input)
            
            response = agent_with_memory.invoke(
                {
                    "question": user_input,
                    "user_profile": current_profile,
                    "procedural_rules": active_rules
                },
                config={"configurable": {"session_id": session_id}}
            )
            agent_reply = response["output"]
            
    except Exception as e:
        print(f"\n[CRITICAL FAULT] Pipeline execution failed: {str(e)}")
        
        
        agent_reply = (
            "I am currently experiencing a connection issue with my core systems. "
            "Your data is safe, but please try asking me again in a moment."
        )
        
    
    try:
        save_episodic_log(session_id, user_input, agent_reply)
    except Exception as e:
        print(f"[Warning] Failed to save episodic memory: {e}")
        
    return agent_reply



