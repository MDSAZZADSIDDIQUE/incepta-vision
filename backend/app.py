import os
import sys
import time
import json
import ast
import re
import threading
import hashlib
import logging
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, date
from uuid import uuid4
from functools import lru_cache
from decimal import Decimal

import oracledb
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError

# SQLAlchemy & Database
from sqlalchemy import create_engine, text, inspect, bindparam
from sqlalchemy.pool import QueuePool

# LangChain / LLM tooling
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_groq import ChatGroq
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction

# RAG / embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# =============================================================================
# 0. ENV & LOGGING
# =============================================================================

load_dotenv()

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*")}})

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=os.getenv("REDIS_URL", "memory://"),
)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================


class Config:
    """Centralized configuration management."""

    # Database
    DB_USER = os.getenv("DB_USER", "MIS")
    DB_PASS = os.getenv("DB_PASSWORD", "mis")
    DB_HOST = os.getenv("DB_HOST", "webportal-db")
    DB_PORT = os.getenv("DB_PORT", "1521")
    DB_SERVICE = os.getenv("DB_SERVICE", "web")
    DB_SCHEMA = os.getenv("DB_SCHEMA", "MIS")

    # Oracle Client (full DB home root, NOT bin)
    ORACLE_HOME = os.getenv(
        "ORACLE_HOME",
        r"D:\app\Administrator\product\11.2.0\dbhome_1",
    )

    # Agent Configuration
    LIBRARIAN_TOP_K = int(os.getenv("LIBRARIAN_TOP_K", "15"))
    AGENT_HISTORY_LIMIT = int(os.getenv("AGENT_HISTORY_LIMIT", "3"))
    AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen/qwen3-32b")
    AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.0"))
    AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "1024"))
    AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Security
    BLOCKED_SQL_KEYWORDS = [
        "DROP",
        "TRUNCATE",
        "DELETE",
        "INSERT",
        "UPDATE",
        "ALTER",
        "CREATE",
        "GRANT",
        "REVOKE",
    ]
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "500"))   # NL
    MAX_SQL_LENGTH = int(os.getenv("MAX_SQL_LENGTH", "5000"))      # SQL text
    MAX_SQL_ROWS = int(os.getenv("MAX_SQL_ROWS", "1000"))          # row cap

    # Admin
    ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

    # Performance
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", "100"))

    # Librarian / Embeddings persistence
    FAISS_DIR = os.getenv("FAISS_DIR", "./cache/faiss")
    FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "schema_tables")
    FAISS_MANIFEST_NAME = os.getenv("FAISS_MANIFEST_NAME", "manifest.json")
    
    # --- STRATEGY 1: GOLDEN DATA DICTIONARY PATH ---
    DICTIONARY_PATH = os.getenv("DICTIONARY_PATH", "dictionary.json")

    # Feedback / Golden Queries (SQLite by default)
    FEEDBACK_DB_PATH = os.getenv("FEEDBACK_DB_PATH", "./cache/feedback.sqlite")
    EMBEDDINGS_BACKEND = os.getenv("EMBEDDINGS_BACKEND", "huggingface")  # huggingface | hosted
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    EMBEDDINGS_CACHE_DIR = os.getenv("EMBEDDINGS_CACHE_DIR", "./embeddings_cache")

    # Startup warm-up
    WARMUP_ON_STARTUP = os.getenv("WARMUP_ON_STARTUP", "1") == "1"
    WARMUP_FORCE_REBUILD = os.getenv("WARMUP_FORCE_REBUILD", "0") == "1"

    # --- RESTRICTED TABLE LIST ---
    # The application will ONLY be aware of these tables.
    ALLOWED_TABLES_LIST = [
        # --- Original MIS Tables ---
        "MIS.YEARLY_SALES_ANALYSIS_WEB",
        "MIS.DASHBOARD_USERS_INFO",
        "MIS.MONTH_WISE_SALES_REPORT",
        "MIS.MONTH_WISE_SALES_RP_WB03_B04",
        "MIS.SUMMARY_OF_SALES",
        "MIS.SUMMARY_OF_SALES_WB03_B04",
        "MIS.ALL_COMPANY_SALES",
        "MIS.ALL_COMPANY_SALES_WB03_B04",
        "MIS.ALL_COMPANY_SALES_CHANNEL_WISE",
        "MIS.ALL_COMP_SALES_CWISE_WB03_B04",
        "MIS.TEAM_WISE_SALES_GROWTH",
        "MIS.BRAND_WISE_SALES_REPORT",
        "MIS.PRO_COM_PRODUCT_BRAND_RANKIN",
        "MIS.BRAND_WISE_SALES_DATA",
        "MIS.PRODUCT_RANKING_DEPOT_SALES",
        "MIS.BRAND_RANKING_YEARLY",
        "MIS.PRODUCT_GROUP_WISE_SALES",
        "MIS.DEPOT_GROUP_WISE_SALES",
        "DWH.PRODUCT_INFO_M",
        "MIS.DEPOT_SALE",
        "DWH.PRODUCT_INFO_M@WEB_TO_IPLDW2",
        "DEPOT@WEB_TO_IMSFA",
        "MIS.DHK_GRP_MKT_WISE_SALES",
        "MIS.DASH_NATIONAL_REPORT",
        "SAMPLE_NEW.DAILY_STOCK@WEB_TO_SAMPLE_MSD",
        "DWH.OS_SALES_ORGANIZATION_INFO@WEB_TO_IPLDW2",
        "DWH.OS_SALES_AREA_INFO@WEB_TO_IPLDW2",
        "DWH.OS_COMPANY_INFO@WEB_TO_IPLDW2",
        "MIS.COMPANY_WISE_SALES_SUMMARY",
        "MIS.YEAR_WISE_TEAM_PERFOR_WB03_B04",
        "MIS.YEAR_WISE_TEAM_PERFORMANCE",
        "MIS.GM_SM_SALES_ANALYSIS_ACH",
        "MIS.RM_SALES_ANALYSIS_ACH",
        "MIS.GROUP_WISE_SALES_RM_TO_GM",
        "MIS.NATIONAL_TARGET_SALES_ACHVM",

        # --- New Dictionary Tables (Export & SCM) ---
        "MIS.EXPO_INFO",
        "MIS.EXPO_ABC",
        "MIS.EXPO_COUNTRY_WISE_PRODUCTS",
        "MIS.EXPO_PLANT",
        "MIS.EXPORT_INSTITUTE_SALES_DETAILS",
        "MIS.EXPORT_SALES_GROWTH",
        "MIS.LONG_LIST_INFORMATION",
        "MIS.SCM_BLOCKLIST_MATERIAL",
        "MIS.SCM_BLOCKLIST_MATERIAL_24_25",
        "MIS.SCM_APP_BLOCKLIST",
        "MIS.SCM_APPLY_REJECTED_DATA",
        "MIS.SCM_CLEARANCE",
        "MIS.SCM_COMPANY_INFO",
        "MIS.SCM_MATERIAL_MANAGEMENT",
        "MIS.SCM_MATERIAL_PURCHASE_INFO",
        "MIS.SCM_FINANCE_MANAGEMENT",
        "MIS.SCM_COMPARATIVE_MASTER_UP",
        "MIS.SCM_PR_RQ_RAW_MAT_UP",
        "MIS.SCM_SAP_MATERIAL_GROUP",
        "MIS.SCM_SHORT_PRODUCT_LIST",
        "MIS.SCM_WARNING_MATERIAL",
        "MIS.SCM_TRIAL_REQ",
        "MIS.SCM_TRIAL_FINAL",
        "MIS.SCM_SUPPLIER_LIST",
        "MIS.SCM_MANUFACTURER_LIST",
        "MIS.SCM_FP_DATA",
        "MIS.SCM_MM_DATA",
        "MIS.SCM_UNIT_LIST"
    ]


# =============================================================================
# 2. REQUEST MODELS
# =============================================================================


class MessageHistory(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|bot)$")
    content: str = Field(default="", max_length=50000)

    @field_validator("content", mode="before")
    @classmethod
    def truncate_content(cls, v: Any) -> Any:
        if isinstance(v, str) and len(v) > 50000:
            return v[:50000]
        return v

    @field_validator("role")
    @classmethod
    def normalize_role(cls, v: str) -> str:
        return "assistant" if v == "bot" else v


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=Config.MAX_QUERY_LENGTH)
    history: List[MessageHistory] = Field(default=[])
    filters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if any(bad in v for bad in ["<script>", ";--", "/*", "*/"]):
            raise ValueError("Invalid characters in query")
        return v.strip()


# =============================================================================
# 2B. FEEDBACK MODELS
# =============================================================================


class FeedbackSQLItem(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    sql: str = Field(..., min_length=1, max_length=Config.MAX_SQL_LENGTH)


class FeedbackRequest(BaseModel):
    rating: str = Field(..., pattern="^(up|down)$")
    message_id: Optional[str] = Field(default=None, max_length=64)
    user_query: Optional[str] = Field(default=None, max_length=Config.MAX_QUERY_LENGTH)
    assistant_text: Optional[str] = Field(default=None, max_length=5000)
    sql: Optional[str] = Field(default=None, max_length=Config.MAX_SQL_LENGTH)
    sql_list: Optional[List[FeedbackSQLItem]] = Field(default=None)
    meta: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# 3. ORACLE CLIENT & ENGINE
# =============================================================================


def init_oracle_client() -> None:
    home = Config.ORACLE_HOME
    if not home or not os.path.exists(home):
        raise RuntimeError(f"ORACLE_HOME '{home}' invalid.")

    lib_dir = os.path.join(home, "bin")
    if not os.path.exists(lib_dir):
        raise RuntimeError(f"Oracle client bin '{lib_dir}' invalid.")

    os.environ["ORACLE_HOME"] = home
    os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")

    try:
        oracledb.init_oracle_client(lib_dir=lib_dir)
        logger.info(f"[OK] Oracle Client initialized from {lib_dir}")
    except Exception as e:
        logger.error(f"[ERROR] Oracle Client init failed: {e}")
        raise


def create_db_engine():
    dsn = f"{Config.DB_HOST}:{Config.DB_PORT}/?service_name={Config.DB_SERVICE}"
    connection_string = f"oracle+oracledb://{Config.DB_USER}:{Config.DB_PASS}@{dsn}"

    engine = create_engine(
        connection_string,
        poolclass=QueuePool,
        pool_size=Config.DB_POOL_SIZE,
        max_overflow=Config.DB_MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
    )
    return engine


init_oracle_client()
engine = create_db_engine()

# =============================================================================
# 3B. FEEDBACK DB (SQLite)
# =============================================================================


def create_feedback_engine():
    path = Config.FEEDBACK_DB_PATH
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )


feedback_engine = create_feedback_engine()


def init_feedback_db() -> None:
    with feedback_engine.begin() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS feedback_event (id TEXT PRIMARY KEY, created_at TEXT, rating TEXT, message_id TEXT, user_query TEXT, assistant_text TEXT, meta_json TEXT)"))
        conn.execute(text("CREATE TABLE IF NOT EXISTS feedback_sql (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, chart_title TEXT, sql_text TEXT, FOREIGN KEY (event_id) REFERENCES feedback_event(id))"))


init_feedback_db()


# =============================================================================
# 4. SQL SECURITY VALIDATOR
# =============================================================================


class SQLValidator:
    """Validates and sanitizes SQL queries for safety."""

    @staticmethod
    def sanitize_sql(sql: str) -> str:
        clean = sql.replace("```sql", "").replace("```", "").strip()
        clean = re.sub(r"/\*.*?\*/", " ", clean, flags=re.DOTALL)
        clean = clean.split("--", 1)[0].strip()
        if clean.endswith(";"):
            clean = clean[:-1].strip()
        return clean

    @staticmethod
    def validate_sql(sql: str) -> Tuple[bool, Optional[str]]:
        if not sql or not sql.strip():
            return False, "Empty SQL query"

        clean_sql = SQLValidator.sanitize_sql(sql)
        if len(clean_sql) > Config.MAX_SQL_LENGTH:
            return False, "SQL query is too long"

        if ";" in clean_sql:
            return False, "Multiple SQL statements are not allowed"

        sql_upper = clean_sql.upper()
        if not sql_upper.strip().startswith("SELECT") and not sql_upper.strip().startswith("WITH"):
            return False, "Only SELECT statements are allowed"

        pattern = r"\b(" + "|".join(re.escape(k) for k in Config.BLOCKED_SQL_KEYWORDS) + r")\b"
        if re.search(pattern, sql_upper):
            return False, "Query contains blocked keywords"

        return True, None


# =============================================================================
# 5. THREAD-SAFE QUERY CACHE
# =============================================================================


class QueryCache:
    def __init__(self, max_size: int = Config.CACHE_SIZE, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._lock = threading.Lock()

    def get(self, query: str) -> Optional[Any]:
        qh = hashlib.md5(query.encode("utf-8")).hexdigest()
        with self._lock:
            if qh in self.cache:
                result, ts = self.cache[qh]
                if time.time() - ts < self.ttl:
                    return result
                del self.cache[qh]
        return None

    def set(self, query: str, result: Any) -> None:
        with self._lock:
            if len(self.cache) >= self.max_size:
                min(self.cache.keys(), key=lambda k: self.cache[k][1]) 
                del self.cache[list(self.cache.keys())[0]] 
            qh = hashlib.md5(query.encode("utf-8")).hexdigest()
            self.cache[qh] = (result, time.time())
    
    def clear(self):
        with self._lock:
            self.cache.clear()


query_cache = QueryCache()


# =============================================================================
# 6. SCHEMA LIBRARIAN
# =============================================================================


def _read_json(path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


class SchemaLibrarian:
    """Manages the ALLOWED table list and builds vector index for retrieval."""

    def __init__(self, db_engine, schema: str):
        self.engine = db_engine
        self.schema = (schema or "").upper()
        self.index_dir = os.path.join(Config.FAISS_DIR, "restricted")
        self.index_name = Config.FAISS_INDEX_NAME
        self.manifest_path = os.path.join(self.index_dir, Config.FAISS_MANIFEST_NAME)
        
        # Parse allowed tables into structured objects
        self.allowed_tables = self._parse_allowed_tables(Config.ALLOWED_TABLES_LIST)
        self.all_table_names = [t["full_name"] for t in self.allowed_tables]

        self.vector_store: Optional[FAISS] = None
        self.embeddings = None
        self._ready = False
        self._lock = threading.Lock()

    def _parse_allowed_tables(self, raw_list: List[str]) -> List[Dict[str, str]]:
        parsed = []
        for raw in raw_list:
            raw = raw.strip().upper()
            if not raw: continue
            
            # Handle DB Links (@)
            db_link = None
            base = raw
            if "@" in raw:
                base, db_link = raw.split("@", 1)
            
            # Handle Schema (e.g. MIS.TABLE)
            schema = Config.DB_SCHEMA.upper()
            table = base
            if "." in base:
                schema, table = base.split(".", 1)
            
            parsed.append({
                "full_name": raw,
                "schema": schema,
                "table": table,
                "db_link": db_link
            })
        return parsed

    def _get_embeddings(self):
        if self.embeddings: return self.embeddings
        logger.info("[LIBRARIAN] Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDINGS_MODEL,
            cache_folder=Config.EMBEDDINGS_CACHE_DIR,
            model_kwargs={"device": "cpu"},
        )
        return self.embeddings

    def _fetch_column_info_bulk(self) -> Dict[str, List[str]]:
        """Fetches columns for tables that are accessible (local or synonymous)."""
        tables_by_schema: Dict[str, List[str]] = {}
        for t in self.allowed_tables:
            if t["db_link"]: continue 
            tables_by_schema.setdefault(t["schema"], []).append(t["table"])

        column_map: Dict[str, List[str]] = {}
        
        try:
            with self.engine.connect() as conn:
                for schema, tables in tables_by_schema.items():
                    if not tables: continue
                    chunk_size = 900
                    for i in range(0, len(tables), chunk_size):
                        chunk = tables[i:i + chunk_size]
                        sql = text("""
                            SELECT TABLE_NAME, COLUMN_NAME
                            FROM ALL_TAB_COLUMNS
                            WHERE OWNER = :schema
                              AND TABLE_NAME IN :tables
                            ORDER BY TABLE_NAME, COLUMN_ID
                        """).bindparams(bindparam('tables', expanding=True))
                        
                        rows = conn.execute(sql, {"schema": schema, "tables": chunk}).fetchall()
                        
                        for t_name, c_name in rows:
                            full_key = f"{schema}.{t_name}"
                            column_map.setdefault(full_key, []).append(c_name)
        except Exception as e:
            logger.warning(f"[LIBRARIAN] Metadata fetch warning: {e}")

        return column_map

    def _build_and_persist(self) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Load Golden Dictionary
        golden_dict = {}
        try:
            if os.path.exists(Config.DICTIONARY_PATH):
                with open(Config.DICTIONARY_PATH, "r", encoding="utf-8") as f:
                    golden_dict = json.load(f)
                logger.info(f"[LIBRARIAN] Loaded Golden Dictionary with {len(golden_dict)} entries.")
        except Exception as e:
            logger.warning(f"[LIBRARIAN] Could not load dictionary.json: {e}")

        column_info = self._fetch_column_info_bulk()
        docs: List[Document] = []
        
        for t in self.allowed_tables:
            full_name = t["full_name"] # e.g. MIS.EXPO_INFO
            table_only = t["table"]    # e.g. EXPO_INFO
            
            # Look up columns using Schema.Table key
            lookup_key = f"{t['schema']}.{t['table']}"
            cols = column_info.get(lookup_key, [])
            
            # Enhanced Description Construction
            # 1. Start with basic info
            desc_parts = [f"Table: {full_name} in schema {t['schema']}"]
            
            # 2. Add Golden Dictionary Info (Matches by table name, e.g. 'EXPO_INFO')
            golden_entry = golden_dict.get(table_only, {})
            if golden_entry:
                if golden_entry.get("description"):
                    desc_parts.append(f"Description: {golden_entry['description']}")
                if golden_entry.get("synonyms"):
                    desc_parts.append(f"Keywords: {', '.join(golden_entry['synonyms'])}")
                
                # Add Golden column comments if available
                if golden_entry.get("columns"):
                    col_desc = []
                    for c_name, c_desc in golden_entry["columns"].items():
                        col_desc.append(f"{c_name}: {c_desc}")
                    if col_desc:
                        desc_parts.append(f"Column Details: {'; '.join(col_desc)}")
            
            if t["db_link"]:
                desc_parts.append(f"Accessed via DB Link {t['db_link']}")
            
            # 3. Combine with raw columns
            meta_text = "\n".join(desc_parts) + f"\nColumns: {', '.join(cols[:150])}"
            docs.append(Document(page_content=meta_text, metadata={"table_name": full_name}))

        if docs:
            self.vector_store = FAISS.from_documents(docs, self._get_embeddings())
            self.vector_store.save_local(self.index_dir, index_name=self.index_name)
        
        _write_json(self.manifest_path, {"built_at": datetime.now().isoformat(), "count": len(self.all_table_names)})
        self._ready = True
        logger.info(f"[LIBRARIAN] Index built for {len(self.all_table_names)} allowed tables.")

    def ensure_ready(self, force_rebuild: bool = False) -> None:
        if self._ready and not force_rebuild: return
        with self._lock:
            if self._ready and not force_rebuild: return
            if not force_rebuild and os.path.exists(os.path.join(self.index_dir, f"{self.index_name}.faiss")):
                try:
                    self.vector_store = FAISS.load_local(self.index_dir, self._get_embeddings(), index_name=self.index_name, allow_dangerous_deserialization=True)
                    self._ready = True
                    return
                except Exception as e:
                    logger.warning(f"[LIBRARIAN] Failed to load index: {e}")
            
            self._build_and_persist()

    def get_relevant_tables(self, query: str, k: int) -> List[str]:
        self.ensure_ready()
        if not self.vector_store:
            return self.all_table_names[:k]
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [d.metadata.get("table_name") for d in docs]
        except Exception:
            return self.all_table_names[:k]

    def get_table_names(self) -> List[str]:
        return self.all_table_names

    def refresh_index(self):
        self._ready = False
        self.ensure_ready(force_rebuild=True)
    
    def status(self):
        return {"ready": self._ready, "count": len(self.all_table_names)}


@lru_cache(maxsize=1)
def get_librarian() -> SchemaLibrarian:
    return SchemaLibrarian(engine, Config.DB_SCHEMA)


# =============================================================================
# 7. FK JOIN GRAPH (Updated for Multi-Schema)
# =============================================================================

class JoinGraph:
    def __init__(self, db_engine):
        self.db_engine = db_engine
        self._loaded = False
        self._adj = {}

    def ensure_loaded(self):
        if self._loaded: return
        
        # Identify all relevant schemas from the Allowed List
        schemas = set()
        for t in Config.ALLOWED_TABLES_LIST:
            clean = t.split("@")[0].strip().upper()
            if "." in clean:
                schemas.add(clean.split(".")[0])
            else:
                schemas.add(Config.DB_SCHEMA)
        
        schema_list = list(schemas)
        logger.info(f"[JOIN] Loading FKs for schemas: {schema_list}")

        with self.db_engine.connect() as conn:
            sql = text("""
                SELECT
                    c.owner || '.' || a.table_name  AS child_table,
                    c_pk.owner || '.' || c_pk.table_name AS parent_table,
                    a.column_name AS child_col,
                    b.column_name AS parent_col
                FROM all_constraints c
                JOIN all_cons_columns a ON c.owner = a.owner AND c.constraint_name = a.constraint_name
                JOIN all_constraints c_pk ON c.r_owner = c_pk.owner AND c.r_constraint_name = c_pk.constraint_name
                JOIN all_cons_columns b ON c_pk.owner = b.owner AND c_pk.constraint_name = b.constraint_name AND a.position = b.position
                WHERE c.constraint_type = 'R'
                  AND c.owner IN :schemas
            """).bindparams(bindparam("schemas", expanding=True))
            
            rows = conn.execute(sql, {"schemas": schema_list}).fetchall()

            self._loaded = True

JOIN_GRAPH = JoinGraph(engine)


# =============================================================================
# 8. CHART & DASHBOARD ANALYST
# =============================================================================


class ChartAnalyst:
    @staticmethod
    def execute_and_format(sql_query: str, db_engine, params=None) -> Tuple[Optional[List[Dict]], Optional[str], Optional[Dict]]:
        try:
            is_valid, error = SQLValidator.validate_sql(sql_query)
            if not is_valid: return None, None, None

            clean_sql = SQLValidator.sanitize_sql(sql_query)
            # Row limit
            if "ROWNUM" not in clean_sql.upper() and "FETCH FIRST" not in clean_sql.upper():
                clean_sql = f"SELECT * FROM ({clean_sql}) WHERE ROWNUM <= {Config.MAX_SQL_ROWS}"

            with db_engine.connect() as conn:
                result = conn.execute(text(clean_sql), params or {}).fetchall()
            
            if not result: return None, None, None
            
            keys = list(result[0]._mapping.keys())
            rows = [dict(r._mapping) for r in result]
            
            # Simple Heuristic: First string col is label, rest are values
            label_key = keys[0]
            normalized = []
            for row in rows:
                val = 0
                if len(keys) > 1 and isinstance(row[keys[1]], (int, float, Decimal)):
                    val = float(row[keys[1]] or 0)
                normalized.append({"label": str(row[label_key]), "value": val})
            
            chart_type = "bar" if len(normalized) > 1 else "kpi"
            if any(x in label_key.lower() for x in ["year", "date", "month"]): chart_type = "line"
            
            return normalized, chart_type, {"rows": rows, "columns": keys}
        except Exception as e:
            logger.error(f"[CHART] Error: {e}")
            return None, None, None


def build_dashboard_from_raw(sql: str, agent_text: str, normalized: List[Dict], chart_type: str, raw_meta: Dict) -> Dict:
    # Construct dashboard payload
    rows = raw_meta.get("rows", [])
    cols = raw_meta.get("columns", [])
    
    # Auto-detect multiple metrics
    charts = []
    if len(cols) > 2:
        label = cols[0]
        for metric_col in cols[1:6]: # Limit to 5 metrics
             if isinstance(rows[0].get(metric_col), (int, float, Decimal)):
                 charts.append({
                     "chartData": [{"label": str(r[label]), "value": float(r[metric_col] or 0)} for r in rows],
                     "chartType": chart_type,
                     "title": f"{metric_col} by {label}",
                     "sql": sql
                 })
    else:
        charts.append({
            "chartData": normalized,
            "chartType": chart_type,
            "title": "Overview",
            "sql": sql
        })

    return {
        "kpis": [], # Logic for KPIs can be added here
        "charts": charts,
        "tables": [{"headers": cols, "rows": [list(r.values()) for r in rows[:50]]}],
        "summary": agent_text
    }


# =============================================================================
# 9. SALES DASHBOARD (Hardcoded)
# =============================================================================

def build_sales_dashboard(db_engine, filters: Dict) -> Dict:
    # Simplified Sales Dashboard Builder compatible with new table list
    # Tables: MIS.MONTH_WISE_SALES_REPORT, MIS.SUMMARY_OF_SALES
    
    combined = {"charts": [], "tables": [], "summary": "Sales Dashboard"}
    
    # 1. Yearly Trend
    sql1 = "SELECT SALES_YEAR, GRAND_TOTAL FROM MIS.MONTH_WISE_SALES_REPORT ORDER BY SALES_YEAR"
    norm1, type1, meta1 = ChartAnalyst.execute_and_format(sql1, db_engine)
    if norm1:
        combined["charts"].append({"chartData": norm1, "chartType": "line", "title": "Yearly Sales Trend", "sql": sql1})
        combined["tables"].append({"headers": meta1["columns"], "rows": [list(r.values()) for r in meta1["rows"]]})
    
    # 2. Segment Breakdown (Latest Year)
    sql2 = """
        SELECT * FROM (
            SELECT SALES_YEAR, DEPO_TOTAL, INST_SALES, EXPORT_PRODUCT 
            FROM MIS.MONTH_WISE_SALES_REPORT 
            ORDER BY SALES_YEAR DESC
        ) WHERE ROWNUM = 1
    """
    _, _, meta2 = ChartAnalyst.execute_and_format(sql2, db_engine)
    if meta2 and meta2["rows"]:
        row = meta2["rows"][0]
        data = [
            {"label": "Depot", "value": float(row.get("DEPO_TOTAL") or 0)},
            {"label": "Institution", "value": float(row.get("INST_SALES") or 0)},
            {"label": "Export", "value": float(row.get("EXPORT_PRODUCT") or 0)},
        ]
        combined["charts"].append({"chartData": data, "chartType": "pie", "title": f"Sales Mix ({row.get('SALES_YEAR')})", "sql": sql2})

    return combined


# =============================================================================
# 10. AGENT FACTORY
# =============================================================================

def get_sql_agent(relevant_tables: List[str]):
    llm = ChatGroq(
        temperature=Config.AGENT_TEMPERATURE,
        model_name=Config.AGENT_MODEL,
        api_key=Config.GROQ_API_KEY,
        streaming=True
    )

    # 1. Separate Local (MIS) vs Remote (DWH/Other) tables
    local_tables = []
    remote_tables = []
    
    current_schema = Config.DB_SCHEMA.upper()
    
    for t in relevant_tables:
        t_upper = t.upper().strip()
        if t_upper.startswith(f"{current_schema}."):
            # Strip schema for local tables so SQLDatabase validation passes
            local_tables.append(t_upper[len(current_schema)+1:])
        elif "." in t_upper:
            # Keep remote tables separate
            remote_tables.append(t_upper)
        else:
            # No schema prefix? Assume local.
            local_tables.append(t_upper)

    # 2. Init SQLDatabase with ONLY local tables to avoid ValueError
    # sample_rows_in_table_info=2 ensures we get column info + samples for local tables
    
    # Store the final mapped names to use in the System Prompt
    # Map: Config_Name (UPPER) -> Actual_DB_Name (Preserved Case)
    prompt_table_map = {} 

    try:
        # Fix: SQLAlchemy/Oracle reflection often returns lowercase names (e.g. 'depot_sale')
        # while our config has uppercase ('DEPOT_SALE'). We must map them to avoid 
        # "include_tables not found" errors which trigger the fallback to ALL tables.
        inspector = inspect(engine)
        all_names = set(inspector.get_table_names())
        try:
            all_names.update(inspector.get_view_names())
        except Exception:
            pass
            
        # Map Config Case -> Actual DB Case
        # e.g. {'DEPOT_SALE': 'depot_sale', ...}
        name_map = {n.upper(): n for n in all_names}
        
        validated_local_tables = []
        for t in local_tables:
            if t in name_map:
                actual_name = name_map[t]
                validated_local_tables.append(actual_name)
                # Map the full MIS.TABLE config name to the actual local name
                prompt_table_map[f"{current_schema}.{t}"] = actual_name
                # Also map the short name
                prompt_table_map[t] = actual_name
            else:
                # Table from config not found in DB reflection. 
                logger.warning(f"[AGENT] Config table '{t}' not found in DB reflection.")

        if not validated_local_tables and local_tables:
            logger.warning(f"[AGENT] None of the local tables matched DB reflection.")
            
        # Explicitly pass schema to avoid ambiguity
        db = SQLDatabase(engine, schema=Config.DB_SCHEMA, include_tables=validated_local_tables, sample_rows_in_table_info=2)
    except Exception as e:
        logger.error(f"[AGENT] SQLDatabase init failed: {e}")
        # CRITICAL FIX: Do NOT fall back to loading all tables (include_tables=None).
        # Fall back to an EMPTY database wrapper.
        db = SQLDatabase(engine, schema=Config.DB_SCHEMA, include_tables=[], sample_rows_in_table_info=0)

    # 3. Manually Fetch Metadata for Remote Tables (DWH, etc.)
    remote_schema_info = ""
    if remote_tables:
        try:
            remote_schema_info = "\n\n**External Table Schemas (Read-Only):**\n"
            
            # Group by schema to optimize queries
            schema_map = {}
            for rt in remote_tables:
                if "@" in rt: 
                    # For DB links, we can't easily fetch metadata, but we should add them to the map
                    prompt_table_map[rt] = rt
                    continue 
                
                parts = rt.split(".", 1)
                if len(parts) == 2:
                    sch, tbl = parts
                    schema_map.setdefault(sch, []).append(tbl)
                    prompt_table_map[rt] = rt # Remote tables keep full name
            
            with engine.connect() as conn:
                for sch, tbls in schema_map.items():
                    if not tbls: continue
                    # Fetch columns for these remote tables
                    for i in range(0, len(tbls), 500):
                        chunk = tbls[i:i+500]
                        sql = text("""
                            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE 
                            FROM ALL_TAB_COLUMNS 
                            WHERE OWNER = :o AND TABLE_NAME IN :t
                            ORDER BY TABLE_NAME, COLUMN_ID
                        """).bindparams(bindparam('t', expanding=True))
                        
                        rows = conn.execute(sql, {"o": sch, "t": chunk}).fetchall()
                        
                        curr_table = None
                        for r in rows:
                            tn = r[0]
                            cn = r[1]
                            dt = r[2]
                            full_name = f"{sch}.{tn}"
                            if full_name != curr_table:
                                remote_schema_info += f"\nTable: {full_name}\nColumns:\n"
                                curr_table = full_name
                            remote_schema_info += f"- {cn} ({dt})\n"
        except Exception as e:
            logger.warning(f"[AGENT] Failed to fetch remote schemas: {e}")

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Helper to resolve table names in prompt
    def resolve_name(config_name):
        return prompt_table_map.get(config_name, config_name)

    system_prompt = f"""
    You are an Oracle SQL Data Analyst.
    
    **MANDATORY TABLE MAPPING RULES (Sales Report Section):**
    1. For "Sales Comparison" or "Historical Report", USE table `{resolve_name('MIS.YEARLY_SALES_ANALYSIS_WEB')}`.
    2. For "Channel wise yearly sales", USE `{resolve_name('MIS.DASHBOARD_USERS_INFO')}` joined with `{resolve_name('MIS.MONTH_WISE_SALES_REPORT')}`.
    3. For "Company wise yearly sale", USE `{resolve_name('MIS.ALL_COMPANY_SALES')}`.
    4. For "Team wise Sales", USE `{resolve_name('MIS.TEAM_WISE_SALES_GROWTH')}`.
    5. For "Brand Ranking", USE `{resolve_name('MIS.BRAND_RANKING_YEARLY')}` or `{resolve_name('MIS.BRAND_WISE_SALES_REPORT')}`.
    6. For "Depot product activity", USE `{resolve_name('MIS.DEPOT_SALE')}` joined with `DWH.PRODUCT_INFO_M`.
    7. For "National Stock", USE `SAMPLE_NEW.DAILY_STOCK@WEB_TO_SAMPLE_MSD`.

    **General Rules:**
    - For tables listed above without a schema prefix (e.g. `{resolve_name('MIS.ALL_COMPANY_SALES')}`), use the name EXACTLY as shown. DO NOT add 'MIS.' prefix.
    - For tables with a schema prefix or '@' (e.g. `DWH.PRODUCT_INFO_M`), use the full name EXACTLY as shown.
    - Use `FETCH FIRST N ROWS ONLY` instead of LIMIT.
    - Provide 2-6 metrics in SELECT for dashboard/trend questions.

    {remote_schema_info}
    """

    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="tool-calling",
        handle_parsing_errors=True,
        max_iterations=Config.AGENT_MAX_ITERATIONS,
        prefix=system_prompt
    )


# =============================================================================
# 11. API ROUTES
# =============================================================================


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_query = data.get("query", "")
    filters = data.get("filters", {})
    
    # 1. Semantic Routing for "Sales Dashboard"
    if "dashboard" in user_query.lower() and "sales" in user_query.lower() and not filters:
        try:
            dash = build_sales_dashboard(engine, filters)
            return jsonify({"type": "dashboard", "dashboard": dash})
        except Exception as e:
            logger.error(f"Dashboard fail: {e}")

    # 2. General Agent Flow
    @stream_with_context
    def generate():
        librarian = get_librarian()
        tables = librarian.get_relevant_tables(user_query, k=Config.LIBRARIAN_TOP_K)
        yield json.dumps({"type": "agent_action", "tool": "librarian", "tool_input": f"Tables: {tables}"}) + "\n"
        
        agent = get_sql_agent(tables)
        
        # Capture SQL via callback (simplified inline here for brevity)
        response_text = ""
        try:
            res = agent.invoke({"input": user_query})
            response_text = res["output"]
            yield json.dumps({"type": "text", "content": response_text}) + "\n"
        except Exception as e:
             yield json.dumps({"type": "text", "content": f"Error: {str(e)}"}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "tables_loaded": len(Config.ALLOWED_TABLES_LIST)})

@app.route("/api/feedback", methods=["POST"])
def feedback():
    return jsonify({"status": "ok", "message": "Feedback received"}) # Placeholder

@app.route("/api/refresh-schema", methods=["POST"])
def refresh_schema():
    librarian = get_librarian()
    librarian.refresh_index()
    return jsonify({"status": "ok", "message": "Schema index refreshed"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)