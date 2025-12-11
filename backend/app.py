import os
import time
import json
import ast
import oracledb
import logging
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# SQLAlchemy & Database
from sqlalchemy import create_engine, text, inspect
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent

# AI & LangChain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentActionMessageLog

# --- NEW: RAG DEPENDENCIES ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 1. SETUP & LOGGING ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# --- 2. ORACLE CLIENT INITIALIZATION (THICK MODE) ---
def init_oracle_client():
    """Initializes Oracle Client for Thick Mode (Required for 11g)."""
    try:
        # Update this path to match your specific deployment environment
        ORACLE_HOME_PATH = os.getenv("ORACLE_HOME", r"D:\app\Administrator\product\11.2.0\dbhome_1\bin")
        if os.path.exists(ORACLE_HOME_PATH):
            oracledb.init_oracle_client(lib_dir=ORACLE_HOME_PATH)
            logger.info("✅ Oracle Client initialized.")
        else:
            logger.warning(f"⚠️ Oracle Path not found at {ORACLE_HOME_PATH}. Assuming Thin mode or System Path.")
    except Exception as e:
        logger.warning(f"⚠️ Client Init Warning: {e}")

init_oracle_client()

# --- 3. DATABASE ENGINE ---
DB_USER = os.getenv("DB_USER", "MIS")
DB_PASS = os.getenv("DB_PASSWORD", "mis")
DB_HOST = os.getenv("DB_HOST", "webportal-db")
DB_PORT = os.getenv("DB_PORT", "1521")
DB_SERVICE = os.getenv("DB_SERVICE", "web")
DB_SCHEMA = os.getenv("DB_SCHEMA", "MIS")

# Connection String
connection_string = f"oracle+oracledb://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/?service_name={DB_SERVICE}"
engine = create_engine(connection_string)

# --- 4. THE LIBRARIAN (OPTIMIZED RAG ENGINE) ---
class SchemaLibrarian:
    """
    Handles the indexing and retrieval of table metadata using Vector Search.
    Replaces the previous LLM-based selection logic.
    """
    def __init__(self, db_engine, schema):
        self.engine = db_engine
        self.schema = schema
        self.vector_store = None
        self.all_table_names = []
        self._initialize_index()

    def _initialize_index(self):
        """Builds a FAISS index of table names and comments from Oracle."""
        logger.info("📚 Librarian: Indexing database schema...")
        try:
            inspector = inspect(self.engine)
            self.all_table_names = inspector.get_table_names(schema=self.schema)
            
            # Try to fetch comments to make the index smarter
            # If your DB has no comments, this gracefully falls back to just table names
            table_docs = []
            
            try:
                with self.engine.connect() as conn:
                    # Oracle specific query to get table comments
                    query = text("""
                        SELECT TABLE_NAME, COMMENTS 
                        FROM ALL_TAB_COMMENTS 
                        WHERE OWNER = :schema
                    """)
                    result = conn.execute(query, {"schema": self.schema}).fetchall()
                    comments_map = {row[0]: row[1] for row in result if row[1]}
            except Exception as e:
                logger.warning(f"Could not fetch table comments: {e}")
                comments_map = {}

            # Create Documents for Vector Store
            for table in self.all_table_names:
                comment = comments_map.get(table, "")
                # Content: The text we search against
                page_content = f"Table: {table}. Description: {comment}" if comment else f"Table: {table}"
                # Metadata: The actual value we return
                metadata = {"table_name": table}
                
                table_docs.append(Document(page_content=page_content, metadata=metadata))

            if not table_docs:
                logger.error("❌ No tables found to index!")
                return

            # Build Index (using local CPU embeddings - fast and free)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_store = FAISS.from_documents(table_docs, embeddings)
            logger.info(f"✅ Librarian: Indexed {len(table_docs)} tables successfully.")

        except Exception as e:
            logger.error(f"❌ Error initializing SchemaLibrarian: {e}")

    def get_relevant_tables(self, query: str, k: int = 5) -> List[str]:
        """Returns the top K most relevant table names for a user query."""
        if not self.vector_store:
            # Fallback if index failed
            return self.all_table_names[:5]

        # Perform Semantic Search
        docs = self.vector_store.similarity_search(query, k=k)
        selected_tables = [doc.metadata["table_name"] for doc in docs]
        
        logger.info(f"🔍 Librarian selected: {selected_tables} for query: '{query}'")
        return selected_tables

# Initialize Librarian Global Instance
librarian = SchemaLibrarian(engine, DB_SCHEMA)

# --- 5. THE ANALYST (CHART & SQL LOGIC) ---
class ChartAnalyst:
    """Handles SQL execution for visualization purposes."""
    
    @staticmethod
    def detect_chart_type(data: List[Dict], keys: List[str]) -> str:
        """Heuristic to determine if data should be Bar, Line, Pie or KPI."""
        if len(data) == 1 and len(keys) == 1:
            return "kpi"
            
        label_key = keys[0]
        # Check if labels look like dates for Line charts
        first_label = str(data[0].get(label_key, "")).lower()
        if any(x in first_label for x in ['202', 'jan', 'feb', 'mar', 'date', 'year', 'month']):
            return "line"
            
        if len(data) <= 5:
            return "pie"
            
        return "bar"

    @staticmethod
    def execute_and_format(sql_query: str, db_engine) -> Tuple[List[Dict], str]:
        try:
            clean_sql = sql_query.replace("``````", "").strip()
            if clean_sql.endswith(';'):
                clean_sql = clean_sql[:-1]

            with db_engine.connect() as conn:
                result = conn.execute(text(clean_sql))
                keys = list(result.keys())
                raw_data = [dict(zip(keys, row)) for row in result.fetchall()]

            if not raw_data:
                return None, None

            # Normalization Logic
            if len(raw_data) == 1 and len(keys) == 1:
                val = list(raw_data[0].values())[0]
                return [{"label": "Total", "value": float(val or 0)}], "kpi"

            label_key = keys[0]
            # Find the first numeric column for values
            value_key = next((k for k, v in raw_data[0].items() if isinstance(v, (int, float))), keys[1] if len(keys) > 1 else keys[0])
            
            normalized_data = []
            for row in raw_data:
                try:
                    val = row[value_key]
                    # Handle Oracle Decimal/Numbers
                    if hasattr(val, 'to_eng_string'): val = float(val)
                    normalized_data.append({
                        "label": str(row[label_key]),
                        "value": float(val)
                    })
                except: continue

            chart_type = ChartAnalyst.detect_chart_type(normalized_data, keys)
            return normalized_data, chart_type

        except Exception as e:
            logger.error(f"❌ Chart Execution Error: {e}")
            return None, None

# --- 6. AGENT SETUP ---
class SQLCaptureHandler(BaseCallbackHandler):
    """Intercepts the SQL generated by the agent."""
    def __init__(self):
        self.captured_sql = None

    def on_tool_start(self, serialized, input_str, **kwargs):
        if serialized.get("name") == "sql_db_query":
            clean_input = input_str
            try:
                # Handle JSON formatted input often sent by LangChain agents
                if "{" in input_str:
                    input_dict = ast.literal_eval(input_str)
                    if 'query' in input_dict:
                        clean_input = input_dict['query']
            except:
                pass
            self.captured_sql = clean_input
            logger.info(f"🎯 Agent Generated SQL: {self.captured_sql}")

def get_sql_agent(relevant_tables: List[str]):
    """Creates a fresh agent instance restricted to specific tables."""
    
    # Model: Smartest available on Groq for Logic
    llm = ChatGroq(
        temperature=0,
        model_name="qwen/qwen3-32b", # Excellent for SQL
        api_key=os.getenv("GROQ_API_KEY"),
        streaming=True
    )

    # Dynamic DB Context: Only exposes relevant tables to the agent
    dynamic_db = SQLDatabase(
        engine,
        schema=DB_SCHEMA,
        include_tables=relevant_tables
    )

    toolkit = SQLDatabaseToolkit(db=dynamic_db, llm=llm)
    
    system_instruction = (
        "You are an expert Oracle SQL Analyst. "
        "Your goal is to answer the user query by querying the database.\n"
        "RULES:\n"
        "1. Use Oracle 11g syntax (ROWNUM <= N instead of LIMIT).\n"
        "2. Only query the tables provided in the schema.\n"
        "3. If the user asks for a visualization (chart, graph), ensure your SQL returns a Label column and a Value column.\n"
        "4. DO NOT perform DML operations (INSERT, UPDATE, DELETE).\n"
        "5. Output the final answer as a summary of the data."
    )

    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="tool-calling",
        handle_parsing_errors=True,
        prefix=system_instruction
    )

# --- 7. API ROUTES ---

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')
    raw_history = data.get('history', [])

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    def generate():
        # Step 1: Librarian Selection (Fast & Cheap)
        # We find top 8 tables to give the agent enough context without overwhelming it
        relevant_tables = librarian.get_relevant_tables(user_query, k=8)
        
        yield json.dumps({
            "type": "agent_action", 
            "tool": "librarian", 
            "tool_input": f"Selected tables: {relevant_tables}"
        }) + "\n"

        # Step 2: Initialize Agent
        sql_handler = SQLCaptureHandler()
        agent_executor = get_sql_agent(relevant_tables)

        # Construct Prompt
        history_text = "\n".join([f"{m['role']}: {m['content'][:200]}" for m in raw_history[-3:]])
        prompt = f"History: {history_text}\nUser Question: {user_query}"

        try:
            # Step 3: Stream Agent Execution
            for chunk in agent_executor.stream(
                {"input": prompt},
                config={"callbacks": [sql_handler]}
            ):
                # A. Tool Usage (Thinking)
                if isinstance(chunk, AgentActionMessageLog):
                    for action in chunk.actions:
                        if isinstance(action, AgentAction):
                            yield json.dumps({
                                "type": "agent_action",
                                "tool": action.tool,
                                "tool_input": action.tool_input
                            }) + "\n"
                
                # B. Final Answer (Text)
                elif isinstance(chunk, dict) and "output" in chunk:
                    yield json.dumps({"type": "text", "content": chunk["output"]}) + "\n"

            # Step 4: Handle Visualization
            if sql_handler.captured_sql:
                chart_data, chart_type = ChartAnalyst.execute_and_format(sql_handler.captured_sql, engine)
                
                if chart_data:
                    yield json.dumps({
                        "type": "chart",
                        "chartData": chart_data,
                        "chartType": chart_type,
                        "sql": sql_handler.captured_sql
                    }) + "\n"
                else:
                    yield json.dumps({"type": "debug_sql", "sql": sql_handler.captured_sql}) + "\n"

        except Exception as e:
            logger.error(f"Execution Error: {e}")
            yield json.dumps({"type": "text", "content": f"I encountered an error: {str(e)}"}) + "\n"

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
