import os
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
load_dotenv()

def run_reflection_batch():
    """Analyzes recent episodic logs to extract, categorize, and upsert procedural rules."""
    # 1. Connect to Databases
    mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    db = mongo_client["cognitive_agent_db"]
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        collection_name="agent_rules",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"
    )
    
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
    
    # 2. Upgraded Prompt forcing Category IDs
    extraction_prompt = PromptTemplate.from_template(
        """Review the user interaction. If the user sets a new rule or updates an old one, extract a 1-word category ID (e.g., EMAIL, CODING, PERSONA) and the rule.
        Format exactly like this: CATEGORY_ID | The rule text.
        If no rule is present, output 'NONE'.
        User: {user_msg}
        Output:"""
    )
    
    chain = extraction_prompt | llm
    
    # 3. Process recent unprocessed logs
    recent_logs = list(db["episodic_logs"].find().sort("timestamp", -1).limit(10))
    
    for log in recent_logs:
        result = chain.invoke({"user_msg": log["user_message"]}).content.strip()
        
        # 4. Split the output and Upsert into ChromaDB
        if result != "NONE" and "|" in result:
            try:
                rule_id, rule_text = result.split("|", 1)
                rule_id = rule_id.strip().upper()
                rule_text = rule_text.strip()
                
                print(f"Upserting Rule [{rule_id}]: {rule_text}")
                
                # Using Chroma's native upsert to overwrite old rules with the same ID
                vector_db._collection.upsert(
                    ids=[rule_id],
                    documents=[rule_text],
                    metadatas=[{"category": rule_id}]
                )
            except Exception as e:
                print(f"Failed to parse rule: {e}")

if __name__ == "__main__":
    print("Initiating batch reflection process...")
    run_reflection_batch()
    print("Reflection complete.")