🧠 Multi-Tier Memory Enterprise Agent

An enterprise-grade cognitive AI backend demonstrating advanced state management, latency optimization, and fault-tolerant architecture. This project solves the core scaling challenges of production LLMs—context amnesia, token limit exhaustion, and catastrophic failure—by distributing operational logic across a multi-tier memory pipeline.

🚀 Architectural Overview

Standard AI chatbots rely on a single, massive system prompt that degrades as context grows. This architecture decouples state into distinct infrastructure layers, orchestrating them via a LangChain backend and a Semantic Router.

Short-Term Memory (Redis): Lightning-fast RAM cache for the active conversational session. Automatically disconnected on session refresh to prevent token bloat and reduce cloud inference costs.

Long-Term Semantic Memory (MongoDB): Persistent graph of user facts and preferences. The agent uses autonomous function-calling (tools) with strict Upsert logic to continuously update records without duplicating database entries.

Procedural Memory (ChromaDB + HuggingFace): RAG-based behavioral rulebook. Instead of overloading the prompt with static instructions, it mathematically maps user intent to local embeddings (all-MiniLM-L6-v2) and dynamically injects formatting rules or safety guardrails at runtime.

Episodic Memory (MongoDB Logs): An immutable, background flight-data recorder that silently archives every interaction, preserving complete cross-session audit trails.

🛠️ Key Engineering Features

Semantic Routing (Cost & Latency Optimization): Intercepts traffic at the front door. Casual greetings are bypassed to a fast, cheap LLM execution path, while complex queries are routed to the heavy tool-calling AgentExecutor, drastically reducing Time-to-First-Token (TTFB).

Autonomous Reflection Loop: A decoupled asynchronous background worker (reflection.py) that batch-analyzes episodic logs to extract new operational rules. It generates unique category IDs and physically upserts new instruction vectors into ChromaDB overnight, allowing the agent to self-improve without real-time latency penalties.

Circuit Breaker Pattern (Graceful Degradation): Defensive engineering wrapping the Groq API and database calls. If external infrastructure drops or times out, the backend intercepts fatal tracebacks and serves a safe fallback UI state rather than crashing the application.

Multi-Session Streamlit UI: A fully featured ChatGPT-style interface allowing users to create, auto-name, switch, and delete asynchronous chat sessions pulled directly from the single-source-of-truth database.

💻 Tech Stack

LLM Orchestration: LangChain, Groq API (Llama 3)

Databases: Redis (State), MongoDB (Persistence), ChromaDB (Vector)

Embeddings: HuggingFace sentence-transformers

Frontend: Streamlit

Python Utilities: Pydantic (Schema validation), python-dotenv

⚙️ Installation & Setup

1. Clone the repository:

git clone https://github.com/yourusername/multi-tier-memory-agent.git
cd multi-tier-memory-agent


(Note: Replace yourusername with your actual GitHub username once you push it)

2. Set up the virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


3. Configure Environment Variables:
Create a .env file in the root directory:

GROQ_API_KEY=your_api_key_here
MONGO_URI=mongodb://localhost:27017/
REDIS_URL=redis://localhost:6379/0


4. Ensure databases are running:
Start your local MongoDB and Redis servers or Docker containers.

5. Run the UI:

streamlit run app.py


🧪 Testing the Architecture

To verify the system's decoupled layers, run these targeted tests:

Test Semantic Routing: Say "Hello!" and watch the backend terminal bypass the Agent Executor for an instant response.

Test Long-Term Upserts: Tell the agent a fact (e.g., your current tech stack). Open a new chat, verify it remembers, then change the fact. Check MongoDB to verify the database overwrote the record using Upsert logic instead of duplicating it.

Test Procedural Reflection & Prompt Locks:

Command the agent to adopt a new persona or formatting rule (e.g., "Always sign off emails as AI Engineer").

Stop the UI, run python reflection.py to trigger the offline vector upsert loop.

Restart the UI, ask a fresh question, and watch the Vector RAG pipeline dynamically enforce the new mathematical rule.

Update the rule again to prove the Meta-Instruction successfully overrides System Prompt Conflicts (Prompt Locks).
