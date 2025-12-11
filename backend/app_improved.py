import os
import sys
import time
import json
import ast
import oracledb
import logging
import hashlib
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
from functools import lru_cache
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pydantic import BaseModel, Field, field_validator

# SQLAlchemy & Database
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.pool import QueuePool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent

# AI & LangChain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentActionMessageLog

# RAG Dependencies
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# =============================================================================
# 1. CONFIGURATION & SETUP
# =============================================================================

load_dotenv()

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*")}})

# Rate Limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.getenv("REDIS_URL", "memory://")
)

# =============================================================================
# 2. CONFIGURATION CONSTANTS
# =============================================================================

class Config:
    """Centralized configuration management"""
    # Database
    DB_USER = os.getenv("DB_USER", "MIS")
    DB_PASS = os.getenv("DB_PASSWORD", "mis")
    DB_HOST = os.getenv("DB_HOST", "webportal-db")
    DB_PORT = os.getenv("DB_PORT", "1521")
    DB_SERVICE = os.getenv("DB_SERVICE", "web")
    DB_SCHEMA = os.getenv("DB_SCHEMA", "MIS")

    # Oracle Client
    ORACLE_HOME_PATH = os.getenv("ORACLE_HOME", r"D:\app\Administrator\product\11.2.0\dbhome_1\bin")

    # Agent Configuration
    LIBRARIAN_TOP_K = int(os.getenv("LIBRARIAN_TOP_K", "8"))
    AGENT_HISTORY_LIMIT = int(os.getenv("AGENT_HISTORY_LIMIT", "3"))
    AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen/qwen3-32b")
    AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0"))

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Security
    BLOCKED_SQL_KEYWORDS = ['DROP', 'TRUNCATE', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'GRANT', 'REVOKE']
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))

    # Performance
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", "100"))

# =============================================================================
# 3. PYDANTIC MODELS FOR REQUEST VALIDATION (PYDANTIC V2 COMPATIBLE)
# =============================================================================

class MessageHistory(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|bot)$")  # Added 'bot'
    content: str = Field(..., min_length=1, max_length=2000)
    
    @field_validator('role')
    @classmethod
    def normalize_role(cls, v):
        """Normalize 'bot' to 'assistant' for consistency."""
        if v == 'bot':
            return 'assistant'
        return v


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=Config.MAX_QUERY_LENGTH)
    history: List[MessageHistory] = Field(default=[])

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        # Basic sanitization
        if any(char in v for char in ['<script>', ';--', '/*', '*/']):
            raise ValueError("Invalid characters in query")
        return v.strip()

# =============================================================================
# 4. ORACLE CLIENT INITIALIZATION
# =============================================================================

def init_oracle_client():
    """Initializes Oracle Client for Thick Mode (Required for 11g)."""
    try:
        if os.path.exists(Config.ORACLE_HOME_PATH):
            oracledb.init_oracle_client(lib_dir=Config.ORACLE_HOME_PATH)
            logger.info("[OK] Oracle Client initialized successfully")
        else:
            logger.warning(f"[WARNING] Oracle Path not found at {Config.ORACLE_HOME_PATH}. Using system path or thin mode.")
    except oracledb.DatabaseError as e:
        logger.error(f"[ERROR] Oracle Client initialization failed: {e}")
        raise
    except Exception as e:
        logger.warning(f"[WARNING] Client Init Warning: {e}")

init_oracle_client()

# =============================================================================
# 5. DATABASE ENGINE WITH CONNECTION POOLING
# =============================================================================

def create_db_engine():
    """Creates SQLAlchemy engine with optimized connection pooling."""
    connection_string = (
        f"oracle+oracledb://{Config.DB_USER}:{Config.DB_PASS}@"
        f"{Config.DB_HOST}:{Config.DB_PORT}/?service_name={Config.DB_SERVICE}"
    )

    # Remove encoding parameters - not supported by oracledb 2.x
    engine = create_engine(
        connection_string,
        poolclass=QueuePool,
        pool_size=Config.DB_POOL_SIZE,
        max_overflow=Config.DB_MAX_OVERFLOW,
        pool_pre_ping=True,  # Validates connections before use
        pool_recycle=3600,   # Recycle connections every hour
        echo=False
    )

    # Test connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1 FROM DUAL"))
        logger.info("[OK] Database connection pool established")
    except Exception as e:
        logger.error(f"[ERROR] Database connection failed: {e}")
        raise

    return engine

engine = create_db_engine()

# =============================================================================
# 6. SQL SECURITY VALIDATOR
# =============================================================================

class SQLValidator:
    """Validates SQL queries for security and safety."""

    @staticmethod
    def validate_sql(sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validates SQL query for dangerous operations.
        Returns (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Empty SQL query"

        sql_upper = sql.upper()

        # Check for blocked keywords
        for keyword in Config.BLOCKED_SQL_KEYWORDS:
            if keyword in sql_upper:
                return False, f"Blocked keyword detected: {keyword}"

        # Ensure it's a SELECT statement
        if not sql_upper.strip().startswith('SELECT'):
            return False, "Only SELECT statements are allowed"

        # Check for excessive wildcards or dangerous patterns
        if sql_upper.count('*') > 5:
            return False, "Too many wildcards in query"

        return True, None

    @staticmethod
    def sanitize_sql(sql: str) -> str:
        """Cleans up SQL query."""
        clean_sql = sql.replace("```sql", "").replace("```", "").strip()
        if clean_sql.endswith(';'):
            clean_sql = clean_sql[:-1]
        return clean_sql

# =============================================================================
# 7. CACHING LAYER
# =============================================================================

class QueryCache:
    """Simple in-memory cache for query results."""

    def __init__(self, max_size: int = Config.CACHE_SIZE):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl = 3600  # 1 hour TTL

    def get_hash(self, query: str) -> str:
        """Generates hash for query."""
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[Any]:
        """Retrieves cached result if available and not expired."""
        query_hash = self.get_hash(query)
        if query_hash in self.cache:
            result, timestamp = self.cache[query_hash]
            if time.time() - timestamp < self.ttl:
                logger.info(f"[CACHE HIT] Query hash: {query_hash[:8]}")
                return result
            else:
                del self.cache[query_hash]
        return None

    def set(self, query: str, result: Any):
        """Stores result in cache."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        query_hash = self.get_hash(query)
        self.cache[query_hash] = (result, time.time())
        logger.info(f"[CACHED] Query hash: {query_hash[:8]}")

query_cache = QueryCache()

# =============================================================================
# 8. SCHEMA LIBRARIAN (RAG-BASED TABLE SELECTION)
# =============================================================================

class SchemaLibrarian:
    """
    Handles indexing and retrieval of table metadata using Vector Search.
    Optimized for fast table selection based on semantic similarity.
    """

    def __init__(self, db_engine, schema: str):
        self.engine = db_engine
        self.schema = schema
        self.vector_store: Optional[FAISS] = None
        self.all_table_names: List[str] = []
        self.embeddings = None
        self._initialize_index()

    def _initialize_index(self):
        """Builds FAISS index of table names and comments from Oracle."""
        logger.info("[LIBRARIAN] Starting schema indexing...")

        try:
            inspector = inspect(self.engine)
            self.all_table_names = inspector.get_table_names(schema=self.schema)

            if not self.all_table_names:
                logger.error("[ERROR] No tables found in schema!")
                return

            # Fetch table comments for richer context
            comments_map = self._fetch_table_comments()

            # Fetch column information for even better context
            column_info_map = self._fetch_column_info()

            # Create Documents for Vector Store
            table_docs = []
            for table in self.all_table_names:
                comment = comments_map.get(table, "")
                columns = column_info_map.get(table, [])
                column_text = ", ".join(columns[:5]) if columns else ""  # First 5 columns

                # Rich page content for better semantic search
                page_content_parts = [f"Table: {table}"]
                if comment:
                    page_content_parts.append(f"Description: {comment}")
                if column_text:
                    page_content_parts.append(f"Columns: {column_text}")

                page_content = ". ".join(page_content_parts)
                metadata = {"table_name": table}
                table_docs.append(Document(page_content=page_content, metadata=metadata))

            # Build FAISS Index
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                cache_folder="./embeddings_cache"
            )
            self.vector_store = FAISS.from_documents(table_docs, self.embeddings)

            logger.info(f"[OK] Librarian: Successfully indexed {len(table_docs)} tables")

        except Exception as e:
            logger.error(f"[ERROR] Error initializing SchemaLibrarian: {e}", exc_info=True)

    def _fetch_table_comments(self) -> Dict[str, str]:
        """Fetches table comments from Oracle data dictionary."""
        comments_map = {}
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT TABLE_NAME, COMMENTS
                    FROM ALL_TAB_COMMENTS
                    WHERE OWNER = :schema AND COMMENTS IS NOT NULL
                """)
                result = conn.execute(query, {"schema": self.schema}).fetchall()
                comments_map = {row[0]: row[1] for row in result}
            logger.info(f"[INFO] Fetched comments for {len(comments_map)} tables")
        except Exception as e:
            logger.warning(f"Could not fetch table comments: {e}")
        return comments_map

    def _fetch_column_info(self) -> Dict[str, List[str]]:
        """Fetches column names for each table."""
        column_info_map = {}
        try:
            inspector = inspect(self.engine)
            for table in self.all_table_names:
                try:
                    columns = inspector.get_columns(table, schema=self.schema)
                    column_info_map[table] = [col['name'] for col in columns]
                except Exception as e:
                    logger.debug(f"Could not fetch columns for {table}: {e}")
                    continue
            logger.info(f"[INFO] Fetched column info for {len(column_info_map)} tables")
        except Exception as e:
            logger.warning(f"Could not fetch column info: {e}")
        return column_info_map

    def get_relevant_tables(self, query: str, k: int = Config.LIBRARIAN_TOP_K) -> List[str]:
        """
        Returns top K most relevant table names for a user query.
        Uses semantic similarity search.
        """
        if not self.vector_store:
            logger.warning("[WARNING] Vector store not initialized, returning all tables")
            return self.all_table_names[:k]

        try:
            # Perform semantic search
            docs = self.vector_store.similarity_search(query, k=k)
            selected_tables = [doc.metadata["table_name"] for doc in docs]

            logger.info(f"[LIBRARIAN] Selected {len(selected_tables)} tables: {selected_tables[:3]}...")
            return selected_tables

        except Exception as e:
            logger.error(f"[ERROR] Error in table selection: {e}")
            return self.all_table_names[:k]

    def refresh_index(self):
        """Refreshes the vector index (useful if schema changes)."""
        logger.info("[REFRESH] Refreshing schema index...")
        self._initialize_index()

# Initialize Librarian
librarian = SchemaLibrarian(engine, Config.DB_SCHEMA)

# =============================================================================
# 9. CHART ANALYST (VISUALIZATION LOGIC)
# =============================================================================

class ChartAnalyst:
    """Handles SQL execution and chart data formatting."""

    @staticmethod
    def detect_chart_type(data: List[Dict], keys: List[str]) -> str:
        """
        Heuristic to determine optimal chart type.
        Returns: 'kpi', 'line', 'pie', or 'bar'
        """
        if not data:
            return "bar"

        # Single value = KPI
        if len(data) == 1 and len(keys) <= 2:
            return "kpi"

        # Check if labels look like time series
        label_key = keys[0]
        first_label = str(data[0].get(label_key, "")).lower()
        time_indicators = ['202', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                          'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 
                          'date', 'year', 'month', 'quarter', 'week', 'day']

        if any(indicator in first_label for indicator in time_indicators):
            return "line"

        # Small datasets = Pie chart
        if len(data) <= 6:
            return "pie"

        # Default = Bar chart
        return "bar"

    @staticmethod
    def execute_and_format(sql_query: str, db_engine) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """
        Executes SQL and formats result for charting.
        Returns: (normalized_data, chart_type) or (None, None) on error
        """
        try:
            # Validate SQL
            is_valid, error_msg = SQLValidator.validate_sql(sql_query)
            if not is_valid:
                logger.error(f"[ERROR] SQL Validation Failed: {error_msg}")
                return None, None

            # Clean SQL
            clean_sql = SQLValidator.sanitize_sql(sql_query)

            # Check cache first
            cached_result = query_cache.get(clean_sql)
            if cached_result:
                return cached_result

            # Execute query
            with db_engine.connect() as conn:
                result = conn.execute(text(clean_sql))
                keys = list(result.keys())
                raw_data = [dict(zip(keys, row)) for row in result.fetchall()]

            if not raw_data:
                logger.info("[INFO] Query returned no data")
                return None, None

            # Handle single value (KPI)
            if len(raw_data) == 1 and len(keys) == 1:
                val = list(raw_data[0].values())[0]
                normalized = [{"label": "Total", "value": float(val or 0)}]
                query_cache.set(clean_sql, (normalized, "kpi"))
                return normalized, "kpi"

            # Multi-row normalization
            label_key = keys[0]

            # Find first numeric column for values
            value_key = None
            for k in keys[1:]:  # Skip first key (label)
                sample_val = raw_data[0].get(k)
                if isinstance(sample_val, (int, float)) or (hasattr(sample_val, 'to_eng_string')):
                    value_key = k
                    break

            if not value_key:
                value_key = keys[1] if len(keys) > 1 else keys[0]

            # Normalize data
            normalized_data = []
            for row in raw_data:
                try:
                    label = str(row[label_key])
                    val = row[value_key]

                    # Handle Oracle NUMBER type
                    if hasattr(val, 'to_eng_string'):
                        val = float(val)

                    normalized_data.append({
                        "label": label,
                        "value": float(val) if val is not None else 0
                    })
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"[WARNING] Skipping row due to conversion error: {e}")
                    continue

            if not normalized_data:
                return None, None

            chart_type = ChartAnalyst.detect_chart_type(normalized_data, keys)

            # Cache the result
            query_cache.set(clean_sql, (normalized_data, chart_type))

            logger.info(f"[OK] Chart data prepared: {chart_type} with {len(normalized_data)} points")
            return normalized_data, chart_type

        except Exception as e:
            logger.error(f"[ERROR] Chart Execution Error: {e}", exc_info=True)
            return None, None

# =============================================================================
# 10. SQL AGENT CALLBACK HANDLER
# =============================================================================

class SQLCaptureHandler(BaseCallbackHandler):
    """Intercepts SQL generated by the agent."""

    def __init__(self):
        self.captured_sql: Optional[str] = None
        self.tool_calls: List[Dict] = []

    def on_tool_start(self, serialized: Dict, input_str: str, **kwargs):
        """Captures tool usage."""
        tool_name = serialized.get("name", "unknown")

        if tool_name == "sql_db_query":
            clean_input = input_str

            # Parse JSON-formatted input
            try:
                if "{" in input_str:
                    input_dict = ast.literal_eval(input_str)
                    if 'query' in input_dict:
                        clean_input = input_dict['query']
            except (ValueError, SyntaxError):
                pass

            self.captured_sql = clean_input
            logger.info(f"[SQL AGENT] Generated SQL: {self.captured_sql[:100]}...")

        self.tool_calls.append({
            "tool": tool_name,
            "input": input_str[:200]  # Truncate for logging
        })

    def on_tool_end(self, output: str, **kwargs):
        """Logs tool completion."""
        logger.debug(f"[TOOL] Completed with output length: {len(output)}")

# =============================================================================
# 11. SQL AGENT FACTORY
# =============================================================================

def get_sql_agent(relevant_tables: List[str]):
    """
    Creates a fresh SQL agent instance restricted to specific tables.
    Uses tool-calling for better reliability.
    """
    try:
        # Initialize LLM
        llm = ChatGroq(
            temperature=Config.AGENT_TEMPERATURE,
            model_name=Config.AGENT_MODEL,
            api_key=Config.GROQ_API_KEY,
            streaming=True,
            timeout=30
        )

        # Create dynamic database context (only expose relevant tables)
        dynamic_db = SQLDatabase(
            engine,
            schema=Config.DB_SCHEMA,
            include_tables=relevant_tables,
            sample_rows_in_table_info=2
        )

        # Create toolkit
        toolkit = SQLDatabaseToolkit(db=dynamic_db, llm=llm)

        # System instruction
        system_instruction = f"""You are an expert Oracle SQL Analyst for Incepta Pharmaceuticals.

Your goal: Answer user questions by querying the database intelligently.

CRITICAL RULES:
1. **Oracle 11g Syntax**: Use ROWNUM <= N instead of LIMIT N
2. **Available Tables**: Only query these tables: {', '.join(relevant_tables[:5])}{'...' if len(relevant_tables) > 5 else ''}
3. **Visualization Queries**: Return exactly 2 columns - one for labels, one for numeric values
4. **Read-Only**: NEVER use INSERT, UPDATE, DELETE, DROP, or ALTER
5. **Clarity**: Provide clear, business-friendly summaries of results
6. **Error Handling**: If query fails, explain why and suggest alternatives

Current Date: {datetime.now().strftime('%Y-%m-%d')}

Think step by step. First understand the question, then query the right tables, then summarize findings."""

        # Create agent
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="tool-calling",
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=30,
            prefix=system_instruction
        )

        return agent_executor

    except Exception as e:
        logger.error(f"[ERROR] Error creating SQL agent: {e}", exc_info=True)
        raise

# =============================================================================
# 12. API ROUTES
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1 FROM DUAL"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    return jsonify({
        "status": "running",
        "database": db_status,
        "librarian_tables": len(librarian.all_table_names),
        "cache_size": len(query_cache.cache),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/tables', methods=['GET'])
def get_tables():
    """Returns available tables in the schema."""
    return jsonify({
        "schema": Config.DB_SCHEMA,
        "tables": librarian.all_table_names,
        "count": len(librarian.all_table_names)
    })

@app.route('/api/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    """Main chat endpoint with streaming response."""
    try:
        # Validate request
        data = request.json
        chat_request = ChatRequest(**data)
        user_query = chat_request.query
        raw_history = [msg.model_dump() for msg in chat_request.history]

        logger.info(f"[NEW QUERY] {user_query[:100]}...")

    except Exception as e:
        logger.error(f"[ERROR] Request validation failed: {e}")
        return jsonify({"error": f"Invalid request: {str(e)}"}), 400

    def generate():
        """Generator function for streaming response."""
        try:
            # Step 1: Table Selection via Librarian
            start_time = time.time()
            relevant_tables = librarian.get_relevant_tables(user_query, k=Config.LIBRARIAN_TOP_K)

            yield json.dumps({
                "type": "agent_action",
                "tool": "librarian",
                "tool_input": f"Selected {len(relevant_tables)} relevant tables",
                "tables": relevant_tables[:5]  # Show first 5
            }) + "\n"

            # Step 2: Initialize Agent
            sql_handler = SQLCaptureHandler()
            agent_executor = get_sql_agent(relevant_tables)

            # Step 3: Build prompt with history
            history_text = "\n".join([
                f"{m['role']}: {m['content'][:150]}" 
                for m in raw_history[-Config.AGENT_HISTORY_LIMIT:]
            ])

            prompt = f"""Previous conversation:
{history_text}

Current question: {user_query}

Please analyze the database and provide a clear, concise answer."""

            # Step 4: Stream Agent Execution
            agent_response = ""

            for chunk in agent_executor.stream(
                {"input": prompt},
                config={"callbacks": [sql_handler]}
            ):
                # Handle different chunk types
                if isinstance(chunk, dict):
                    # Agent action (tool usage)
                    if "actions" in chunk:
                        for action in chunk["actions"]:
                            if isinstance(action, AgentAction):
                                yield json.dumps({
                                    "type": "agent_action",
                                    "tool": action.tool,
                                    "tool_input": str(action.tool_input)[:200]
                                }) + "\n"

                    # Final answer
                    if "output" in chunk:
                        agent_response = chunk["output"]
                        yield json.dumps({
                            "type": "text",
                            "content": agent_response
                        }) + "\n"

            # Step 5: Handle Visualization
            if sql_handler.captured_sql:
                chart_data, chart_type = ChartAnalyst.execute_and_format(
                    sql_handler.captured_sql, 
                    engine
                )

                if chart_data:
                    yield json.dumps({
                        "type": "chart",
                        "chartData": chart_data,
                        "chartType": chart_type,
                        "sql": sql_handler.captured_sql
                    }) + "\n"
                else:
                    yield json.dumps({
                        "type": "debug_sql",
                        "sql": sql_handler.captured_sql
                    }) + "\n"

            # Log performance
            elapsed = time.time() - start_time
            logger.info(f"[OK] Query completed in {elapsed:.2f}s")

            yield json.dumps({
                "type": "metadata",
                "execution_time": elapsed,
                "tables_used": len(relevant_tables)
            }) + "\n"

        except Exception as e:
            logger.error(f"[ERROR] Execution error: {e}", exc_info=True)
            yield json.dumps({
                "type": "text",
                "content": f"I encountered an error: {str(e)}. Please try rephrasing your question."
            }) + "\n"

    return Response(
        stream_with_context(generate()), 
        mimetype='application/x-ndjson',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/api/refresh-schema', methods=['POST'])
def refresh_schema():
    """Refreshes the table index (admin endpoint)."""
    try:
        librarian.refresh_index()
        return jsonify({
            "status": "success",
            "message": f"Schema refreshed. {len(librarian.all_table_names)} tables indexed."
        })
    except Exception as e:
        logger.error(f"[ERROR] Schema refresh failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clears the query cache (admin endpoint)."""
    query_cache.cache.clear()
    return jsonify({
        "status": "success",
        "message": "Cache cleared"
    })

# =============================================================================
# 13. ERROR HANDLERS
# =============================================================================

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later."
    }), 429

@app.errorhandler(500)
def internal_error_handler(e):
    logger.error(f"[ERROR] Internal error: {e}", exc_info=True)
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong. Please contact support."
    }), 500

# =============================================================================
# 14. APPLICATION STARTUP
# =============================================================================

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("[STARTUP] Incepta AI Chatbot Backend")
    logger.info("="*60)
    logger.info(f"[CONFIG] Schema: {Config.DB_SCHEMA}")
    logger.info(f"[CONFIG] Model: {Config.AGENT_MODEL}")
    logger.info(f"[CONFIG] Tables indexed: {len(librarian.all_table_names)}")
    logger.info("="*60)

    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=os.getenv("FLASK_DEBUG", "False") == "True",
        threaded=True
    )
