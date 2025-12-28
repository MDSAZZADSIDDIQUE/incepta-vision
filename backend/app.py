import os
import re
import io
import json
import time
import base64
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import oracledb
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.utils import ImageReader

load_dotenv()

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("mis-ai")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
logger.addHandler(_handler)

# -------------------------
# Flask
# -------------------------
app = Flask(__name__)
CORS(app)

# -------------------------
# LLM (Groq)
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen-2.5-32b")

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set. /chat will fail until it is configured.")

llm = Groq(api_key=GROQ_API_KEY)

# -------------------------
# Oracle Client init (optional thick mode)
# -------------------------
try:
    oracle_home = os.getenv("ORACLE_HOME")
    if oracle_home:
        oracledb.init_oracle_client(lib_dir=os.path.join(oracle_home, "bin"))
except Exception as e:
    logger.warning(f"Could not init Oracle Client (thick mode). Using thin mode. Error: {e}")

def get_db_connection():
    dsn = oracledb.makedsn(
        os.getenv("DB_HOST"),
        os.getenv("DB_PORT"),
        service_name=os.getenv("DB_SERVICE"),
    )
    return oracledb.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dsn=dsn,
    )

# -------------------------
# Page → Allowed tables (normalize to uppercase, include schema)
# IMPORTANT: Use these keys in your Power BI "Talk with AI" URL: ?page=<key>
# -------------------------
PAGE_TABLES: Dict[str, List[str]] = {
    # Historical Report
    "sales_comparison": ["MIS.YEARLY_SALES_ANALYSIS_WEB"],

    "channel_wise_yearly_sales": [
        "MIS.DASHBOARD_USERS_INFO",
        "MIS.MONTH_WISE_SALES_REPORT",
        "MIS.MONTH_WISE_SALES_RP_WB03_B04",
    ],

    "channel_group_wise_sales": [
        "MIS.DASHBOARD_USERS_INFO",
        "MIS.SUMMARY_OF_SALES",
        "MIS.SUMMARY_OF_SALES_WB03_B04",
    ],

    "company_wise_yearly_sale": [
        "MIS.DASHBOARD_USERS_INFO",
        "MIS.ALL_COMPANY_SALES",
        "MIS.ALL_COMPANY_SALES_WB03_B04",
    ],

    "channel_company_wise_sales": [
        "MIS.DASHBOARD_USERS_INFO",
        "MIS.ALL_COMPANY_SALES_CHANNEL_WISE",
        "MIS.ALL_COMP_SALES_CWISE_WB03_B04",
    ],

    "team_wise_sales": [
        "MIS.YEAR_WISE_SUMMARY_DATA",
        "MIS.DASHBOARD_USERS_INFO",
        "MIS.TEAM_WISE_SALES_GROWTH",
    ],

    # Brand Ranking
    "brand_ranking": [
        "MIS.BRAND_WISE_SALES_REPORT",
        "MIS.PRO_COM_PRODUCT_BRAND_RANKIN",
        "MIS.BRAND_WISE_SALES_DATA",
    ],
    "brand_ranking_monthly": ["MIS.PRODUCT_RANKING_DEPOT_SALES"],
    "brand_ranking_yearly": ["MIS.BRAND_RANKING_YEARLY"],

    # Current Month Report
    "product_group_wise_sales": ["MIS.PRODUCT_GROUP_WISE_SALES"],

    "depot_wise_sales": [
        "MIS.PRODUCT_GROUP_WISE_SALES",
        "MIS.DEPOT_GROUP_WISE_SALES",
    ],

    "depot_product_activity": [
        "DWH.PRODUCT_INFO_M",
        "MIS.DEPOT_SALE",
        "DWH.PRODUCT_INFO_M@WEB_TO_IPLDW2",
        "DEPOT@WEB_TO_IMSFA",
    ],

    "dhaka_depot_sale": ["MIS.DHK_GRP_MKT_WISE_SALES"],

    "national_report": ["MIS.DASH_NATIONAL_REPORT"],

    "national_stock": [
        "SAMPLE_NEW.DAILY_STOCK@WEB_TO_SAMPLE_MSD",
        "DWH.PRODUCT_INFO_M@WEB_TO_IPLDW2",
        "DWH.OS_SALES_ORGANIZATION_INFO@WEB_TO_IPLDW2",
        "DWH.OS_SALES_AREA_INFO@WEB_TO_IPLDW2",
        "DWH.OS_COMPANY_INFO@WEB_TO_IPLDW2",
    ],
}

# -------------------------
# Dictionary.json schema helper
# -------------------------
def _candidate_dict_paths() -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        os.getenv("DICTIONARY_PATH", ""),
        os.path.join(here, "dictionary.json"),
        os.path.join(os.getcwd(), "dictionary.json"),
        "/mnt/data/dictionary.json",
    ]

def _load_dictionary() -> Dict[str, Any]:
    for p in _candidate_dict_paths():
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                logger.info(f"Loaded dictionary.json from: {p}")
                return json.load(f)
    logger.warning("dictionary.json not found. Will rely on DB introspection for columns.")
    return {}

DICTIONARY = _load_dictionary()

def normalize_table_name(name: str) -> str:
    """Return uppercase table name, ensure schema if missing."""
    n = (name or "").strip().strip('"').upper()
    if not n:
        return n
    # Keep DB link suffix if present
    if "." not in n and n not in ("DUAL",):
        n = f"MIS.{n}"
    return n

def base_table_name(name: str) -> str:
    """Strip schema and DB link to match dictionary keys."""
    n = normalize_table_name(name)
    # remove schema
    if "." in n:
        n = n.split(".", 1)[1]
    # remove db link
    if "@" in n:
        n = n.split("@", 1)[0]
    return n

def build_llm_context_from_dictionary(table_norm: str) -> Optional[str]:
    key = base_table_name(table_norm)  # dictionary keys are like ALL_COMPANY_SALES
    details = DICTIONARY.get(key)
    if not details:
        return None

    desc = details.get("description", "")
    synonyms = details.get("synonyms", []) or []
    cols = details.get("columns", {}) or {}

    ctx = [f"**TABLE: {table_norm}**"]
    if desc:
        ctx.append(f"- Description: {desc}")
    if synonyms:
        ctx.append(f"- Keywords: {', '.join(synonyms)}")
    if cols:
        ctx.append("- Columns: " + ", ".join(cols.keys()))
    return "\n".join(ctx) + "\n"

# Cache for introspected columns
_INTROSPECT_CACHE: Dict[str, List[str]] = {}

def introspect_columns(conn: oracledb.Connection, table_norm: str) -> List[str]:
    """Safely fetch column names without scanning data (works for db links too)."""
    table_norm = normalize_table_name(table_norm)
    if table_norm in _INTROSPECT_CACHE:
        return _INTROSPECT_CACHE[table_norm]

    cols: List[str] = []
    try:
        cur = conn.cursor()
        # 1=0 ensures no rows are returned; we only need cursor.description
        cur.execute(f"SELECT * FROM {table_norm} WHERE 1=0")
        if cur.description:
            cols = [d[0] for d in cur.description if d and d[0]]
        cur.close()
    except Exception as e:
        logger.warning(f"Could not introspect columns for {table_norm}: {e}")
        cols = []

    _INTROSPECT_CACHE[table_norm] = cols
    return cols

def build_schema_context(conn: oracledb.Connection, allowed_tables: List[str]) -> str:
    """Build compact schema context for the LLM, only for allowed tables."""
    parts: List[str] = []
    for t in allowed_tables:
        tn = normalize_table_name(t)
        ctx = build_llm_context_from_dictionary(tn)
        if ctx:
            parts.append(ctx)
            continue

        # fallback: DB introspection
        cols = introspect_columns(conn, tn)
        if cols:
            parts.append(f"**TABLE: {tn}**\n- Columns: {', '.join(cols)}\n")
        else:
            parts.append(f"**TABLE: {tn}**\n- Columns: (unknown)\n")
    return "\n".join(parts)

# -------------------------
# SQL safety + validation
# -------------------------
FORBIDDEN_KEYWORDS = ["DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", "ALTER", "GRANT", "REVOKE", "CREATE", "REPLACE", "MERGE"]

def is_safe_sql(sql: str) -> bool:
    su = (sql or "").upper()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", su):
            return False
    return True

def strip_sql_comments(sql: str) -> str:
    # remove /* ... */ and -- ... endline
    s = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    s = re.sub(r"--.*?$", " ", s, flags=re.MULTILINE)
    return s

def extract_table_references(sql: str) -> List[str]:
    """Best-effort extraction of tables used in FROM/JOIN clauses."""
    if not sql:
        return []
    s = strip_sql_comments(sql)
    # drop string literals to reduce false matches
    s = re.sub(r"'.*?'", "''", s, flags=re.DOTALL)
    su = s.upper()

    # capture chunks after FROM/JOIN up to a clause boundary
    boundaries = r"\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|\bUNION\b|\bCONNECT\b|\bSTART\b|\bFETCH\b|\bFOR\b|\bMODEL\b|\bQUALIFY\b"
    refs: List[str] = []

    for kw in ["FROM", "JOIN"]:
        for m in re.finditer(rf"\b{kw}\b\s+", su):
            start = m.end()
            # find boundary
            b = re.search(boundaries, su[start:])
            end = start + (b.start() if b else len(su[start:]))
            chunk = su[start:end].strip()

            # ignore subquery starts
            if chunk.startswith("("):
                continue

            # split by commas (legacy joins)
            for part in chunk.split(","):
                token = part.strip().split()[0] if part.strip() else ""
                if not token or token.startswith("("):
                    continue
                # remove trailing punctuation
                token = token.rstrip(");")
                refs.append(token)

    # normalize + unique
    normed = []
    seen = set()
    for r in refs:
        rn = normalize_table_name(r)
        if rn and rn not in seen:
            seen.add(rn)
            normed.append(rn)
    return normed

def validate_sql_tables(sql: str, allowed_tables: List[str]) -> Tuple[bool, List[str]]:
    allowed_set = {normalize_table_name(t) for t in allowed_tables}
    allowed_set.add("DUAL")  # allow Oracle DUAL
    used = extract_table_references(sql)
    disallowed = [t for t in used if t not in allowed_set]
    return (len(disallowed) == 0), disallowed

def ensure_rownum_limit(sql: str, max_rows: int = 2000) -> str:
    su = sql.upper()
    if re.search(r"\bROWNUM\b", su) or re.search(r"\bFETCH\s+FIRST\b", su):
        return sql
    # Wrap as subquery so we don't accidentally change logic (works for most SELECTs)
    return f"SELECT * FROM (\n{sql}\n) WHERE ROWNUM <= {max_rows}"

def extract_sql_from_llm(text: str) -> str:
    if not text:
        return ""
    # preferred: ```sql ... ```
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(1).strip()
        return sql[:-1] if sql.endswith(";") else sql
    # fallback: try to find first SELECT
    m2 = re.search(r"\bSELECT\b.*", text, flags=re.DOTALL | re.IGNORECASE)
    if m2:
        sql = m2.group(0).strip()
        return sql[:-1] if sql.endswith(";") else sql
    return ""

def safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from a string (best-effort)."""
    if not text:
        return None
    # If whole content is json
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to locate first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# -------------------------
# LLM prompts
# -------------------------
def sql_system_prompt(schema_context: str, allowed_tables: List[str]) -> str:
    today = datetime.now()
    allowed_lines = "\n".join([f"- {normalize_table_name(t)}" for t in allowed_tables])

    return f"""
You are an Oracle 11g SQL expert. Your goal is 100% accurate table selection and readable outputs.

### ALLOWED TABLES (YOU MUST ONLY USE THESE)
{allowed_lines}
- You may also use Oracle's DUAL.
- If you cannot answer using ONLY the allowed tables, return:
  ```sql
  SELECT 'CANNOT_ANSWER: <short reason>' AS MESSAGE FROM DUAL
  ```

### SCHEMA CONTEXT (ONLY TABLES ABOVE)
{schema_context}

### RULES
- Oracle 11g: use WHERE ROWNUM <= N (NEVER LIMIT).
- If the question asks for a time trend (monthly/weekly/daily) the result MUST be aggregated to one row per time bucket.
- If a table stores months as separate columns (JAN..DEC), use UNPIVOT + SUM so charts have exactly 12 rows (one per month).
- When sorting a month output, prefer ORDER BY a numeric month column/position (e.g., ORDER BY 2) or a CASE expression.
- In UNION/UNION ALL queries, ORDER BY should reference output column positions to avoid alias issues (e.g., ORDER BY 2).
- Always select readable labels (NAME/DESCRIPTION) when IDs are present.
- Use UPPER(col) LIKE UPPER('%value%') for string filters.
- If user does not specify time, prefer the most recent year/month available in the table (do not assume "today" unless asked).
- Return ONLY SQL in a ```sql``` code fence. No explanation.

### TEMPORAL CONTEXT
- Today: {today.strftime("%d-%b-%y").upper()}
- Month: {today.strftime("%b-%y").upper()}
""".strip()

REPORT_PROMPT = r"""
You are a BI analyst and dashboard designer.

Given:
- The user's question
- The SQL used
- The returned columns and a small sample of rows

Create a compact "in-app dashboard spec" as JSON ONLY.

JSON schema (no extra keys):
{
  "title": string,
  "narrative_md": string,        // markdown, 2-6 lines, business style
  "visuals": [
    // up to 4 items total; include exactly one "table" item (last)
    // types supported: "kpi", "bar", "line", "table"
    // For charts: use only provided column names
    // For KPI: choose an aggregation that can be computed from the dataset
    {
      "id": string,
      "type": "kpi",
      "label": string,
      "column": string,          // numeric column name
      "agg": "sum"|"avg"|"min"|"max"|"count"|"count_distinct",
      "format": "number"|"currency"|"percent"
    },
    {
      "id": string,
      "type": "line"|"bar",
      "title": string,
      "xKey": string,
      "yKeys": [string]          // 1-3 numeric columns
    },
    { "id": string, "type": "table", "title": string }
  ],
  "followups": [string, string, string]
}

Guidance:
- If there is a date/month/year column, prefer a LINE chart.
- Otherwise prefer BAR chart with a categorical xKey.
- Choose yKeys as numeric value columns (sales/value/amount first).
- Ensure every referenced column exists in the provided column list.
- Keep it simple and Power-BI-like.
Return JSON only.
""".strip()

# -------------------------
# Core: SQL generation + execution
# -------------------------
def generate_sql(user_query: str, schema_context: str, allowed_tables: List[str]) -> str:
    sys = sql_system_prompt(schema_context, allowed_tables)
    completion = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user_query},
        ],
        temperature=0.1,
    )
    return extract_sql_from_llm(completion.choices[0].message.content or "")

def fix_sql(user_query: str, bad_sql: str, error_msg: str, schema_context: str, allowed_tables: List[str]) -> str:
    sys = sql_system_prompt(schema_context, allowed_tables)
    prompt = f"""The SQL failed on Oracle with this error:
{error_msg}

Fix the SQL. Keep it Oracle 11g. Use ONLY allowed tables.

If the error is ORA-00904 (invalid identifier) and it is caused by ORDER BY on an alias, prefer ORDER BY column position (e.g., ORDER BY 2) or repeat the expression.

User question: {user_query}

Bad SQL:
```sql
{bad_sql}
```
"""
    completion = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return extract_sql_from_llm(completion.choices[0].message.content or "")

def run_sql(conn: oracledb.Connection, sql: str) -> Tuple[List[str], List[List[Any]]]:
    cur = conn.cursor()
    cur.execute(sql)
    columns = [d[0] for d in (cur.description or [])]
    rows = []
    for row in cur.fetchall():
        processed = []
        for item in row:
            if item is None:
                processed.append("")
            elif isinstance(item, (bytes, bytearray)):
                processed.append(item.decode(errors="ignore"))
            else:
                # handle CLOB safely
                try:
                    if hasattr(item, "read"):
                        processed.append(str(item.read()))
                    else:
                        processed.append(str(item))
                except Exception:
                    processed.append(str(item))
        rows.append(processed)
    cur.close()
    return columns, rows


# -------------------------
# Small deterministic helper for a common MIS pattern:
# YEARLY_SALES_ANALYSIS_WEB often stores months as separate columns (JAN..DEC).
# If user asks for a monthly trend, build a safe UNPIVOT+SUM query to return 12 rows.
# -------------------------
MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]

def build_monthly_unpivot_sql(conn: oracledb.Connection, table_norm: str, year: int, year_col_guess: List[str] = None) -> Optional[str]:
    year_col_guess = year_col_guess or ["SALES_YEAR", "YEAR", "S_YEAR"]
    cols = introspect_columns(conn, table_norm)
    if not cols:
        return None
    colset = {c.upper() for c in cols}

    year_col = None
    for c in year_col_guess:
        if c in colset:
            year_col = c
            break
    if not year_col:
        return None

    month_cols = [m for m in MONTHS if m in colset]
    if len(month_cols) < 6:  # avoid false positives
        return None

    in_list = ",\n    ".join([f"{m} AS '{m}'" for m in month_cols])
    case_expr = "CASE MONTH_NAME\n" + "\n".join([f"    WHEN '{m}' THEN {i+1}" for i, m in enumerate(MONTHS)]) + "\n    ELSE NULL\nEND"

    # Note: ORDER BY 2 avoids alias/identifier edge cases
    sql = f"""
SELECT
  MONTH_NAME,
  {case_expr} AS MONTH_NUM,
  SUM(SALES) AS SALES
FROM (
  SELECT *
  FROM {table_norm}
  WHERE {year_col} = {int(year)}
) 
UNPIVOT (SALES FOR MONTH_NAME IN (
    {in_list}
))
GROUP BY MONTH_NAME, {case_expr}
ORDER BY 2
""".strip()
    return sql

# -------------------------
# Report spec generation
# -------------------------
def heuristic_report_spec(user_query: str, columns: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    # Basic fallback if LLM JSON fails
    cols = columns or []
    sample_obj = {}
    if rows and cols:
        for i, c in enumerate(cols):
            sample_obj[c] = rows[0][i]

    # pick xKey
    lc = [c.lower() for c in cols]
    xKey = None
    for k in ["date", "month", "year"]:
        for c in cols:
            if k in c.lower():
                xKey = c
                break
        if xKey:
            break
    if not xKey:
        # first non-numeric-looking column
        xKey = cols[0] if cols else ""

    # numeric columns: best-effort by checking sample row parse
    numeric_cols = []
    for c in cols:
        v = sample_obj.get(c, "")
        try:
            float(str(v).replace(",", ""))
            numeric_cols.append(c)
        except Exception:
            pass
    # prefer sales/value columns
    preferred = [c for c in numeric_cols if re.search(r"(sales|value|amount|total|revenue)", c, flags=re.I)]
    yKeys = (preferred or numeric_cols)[:2] if numeric_cols else ([])

    chart_type = "line" if re.search(r"(date|month|year)", xKey or "", flags=re.I) else "bar"

    visuals = []
    if yKeys:
        visuals.append({
            "id": "kpi_1",
            "type": "kpi",
            "label": f"Total {yKeys[0]}",
            "column": yKeys[0],
            "agg": "sum",
            "format": "number",
        })
        visuals.append({
            "id": "chart_1",
            "type": chart_type,
            "title": "Trend" if chart_type == "line" else "Breakdown",
            "xKey": xKey,
            "yKeys": yKeys[:2],
        })
    visuals.append({"id": "table_1", "type": "table", "title": "Data Table"})

    return {
        "title": "AI Report",
        "narrative_md": "Here is a dashboard view based on your query.\n\nUse the chart(s) to spot patterns and the table for detail.",
        "visuals": visuals,
        "followups": [
            "Show the same result for last year vs this year",
            "Break it down by channel/depot",
            "Show top 10 items only",
        ],
    }

def generate_report_spec(user_query: str, sql: str, columns: List[str], rows: List[List[Any]]) -> Dict[str, Any]:
    sample_rows = rows[:10]
    payload = {
        "question": user_query,
        "sql": sql,
        "columns": columns,
        "sample_rows": sample_rows,
    }
    completion = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": REPORT_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.2,
    )
    content = completion.choices[0].message.content or ""
    parsed = safe_json_extract(content)
    if not parsed:
        return heuristic_report_spec(user_query, columns, rows)

    # minimal validation: required keys
    for k in ["title", "narrative_md", "visuals", "followups"]:
        if k not in parsed:
            return heuristic_report_spec(user_query, columns, rows)

    # ensure table exists + last
    visuals = parsed.get("visuals") or []
    if not any(v.get("type") == "table" for v in visuals):
        visuals.append({"id": "table_1", "type": "table", "title": "Data Table"})
    # move table to end
    visuals = [v for v in visuals if v.get("type") != "table"] + [v for v in visuals if v.get("type") == "table"]
    parsed["visuals"] = visuals[:4]  # hard cap

    # validate columns in visuals
    colset = set(columns or [])
    for v in parsed["visuals"]:
        if v.get("type") in ("bar", "line"):
            if v.get("xKey") not in colset:
                v["xKey"] = columns[0] if columns else ""
            yk = v.get("yKeys") or []
            v["yKeys"] = [c for c in yk if c in colset][:3]
        if v.get("type") == "kpi":
            if v.get("column") not in colset:
                # pick first numeric-looking column if available
                v["column"] = columns[0] if columns else ""
                v["agg"] = "count"
                v["format"] = "number"
    return parsed

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "time": datetime.now().isoformat()})

@app.post("/chat")
def chat():
    started = time.time()
    data = request.get_json(force=True) or {}
    user_query = (data.get("message") or "").strip()
    page = (data.get("page") or "").strip()

    if not user_query:
        return jsonify({"response": "Please enter a question.", "sql": "", "raw_data": [], "columns": [], "suggestions": [], "report": None}), 400

    if page not in PAGE_TABLES:
        return jsonify({
            "response": f"Unknown page context '{page}'. Please call /chat with a valid page key.",
            "sql": "",
            "raw_data": [],
            "columns": [],
            "suggestions": list(PAGE_TABLES.keys())[:10],
            "report": None,
        }), 400

    allowed_tables = [normalize_table_name(t) for t in PAGE_TABLES[page]]
    logger.info(f"Query page={page} allowed_tables={allowed_tables} q={user_query}")

    conn = None
    try:
        conn = get_db_connection()
        schema_context = build_schema_context(conn, allowed_tables)

        # 1) Generate SQL (with validation/regeneration)
        # Fast-path: for Sales Comparison monthly trend queries, use a deterministic UNPIVOT pattern
        sql = ""
        if page == "sales_comparison" and "MIS.YEARLY_SALES_ANALYSIS_WEB" in allowed_tables:
            m_year = re.search(r"\b(20\d{2})\b", user_query)
            wants_monthly = re.search(r"\b(month|monthly|trend)\b", user_query, flags=re.I)
            if m_year and wants_monthly:
                y = int(m_year.group(1))
                sql_candidate = build_monthly_unpivot_sql(conn, "MIS.YEARLY_SALES_ANALYSIS_WEB", y)
                if sql_candidate:
                    sql = sql_candidate

        for attempt in range(3):
            # If the deterministic fast-path produced SQL, keep it and skip LLM generation.
            sql = generate_sql(user_query, schema_context, allowed_tables) if (attempt == 0 and not sql) else sql
            if not sql:
                sql = generate_sql(user_query, schema_context, allowed_tables)
            if not sql:
                continue

            if not is_safe_sql(sql):
                return jsonify({"response": "Unsafe SQL was generated and blocked.", "sql": "", "raw_data": [], "columns": [], "suggestions": [], "report": None}), 400

            ok_tables, disallowed = validate_sql_tables(sql, allowed_tables)
            if ok_tables:
                break

            # Ask LLM to regenerate with constraint feedback
            feedback = f"Your SQL used disallowed tables: {', '.join(disallowed)}. Regenerate using ONLY allowed tables."
            regen_prompt = f"{user_query}\n\n{feedback}"
            sql = generate_sql(regen_prompt, schema_context, allowed_tables)

        if not sql:
            return jsonify({"response": "Could not generate SQL for this question.", "sql": "", "raw_data": [], "columns": [], "suggestions": [], "report": None}), 500

        # Special case: CANNOT_ANSWER from model
        if re.search(r"CANNOT_ANSWER", sql, flags=re.IGNORECASE):
            return jsonify({
                "response": "I can’t answer that from this MIS page’s allowed datasets. Try opening the relevant MIS page first, then click Talk with AI again.",
                "sql": sql,
                "raw_data": [],
                "columns": [],
                "suggestions": [],
                "report": None,
            })

        # 2) Execute (with fix loop)
        sql_exec = ensure_rownum_limit(sql, max_rows=2000)
        columns: List[str] = []
        rows: List[List[Any]] = []
        last_error = ""

        for attempt in range(3):
            try:
                columns, rows = run_sql(conn, sql_exec)
                break
            except oracledb.DatabaseError as e:
                error_obj, = e.args
                last_error = getattr(error_obj, "message", str(e))
                logger.error(f"DB Error attempt {attempt+1}: {last_error}")

                if attempt == 2:
                    return jsonify({"response": f"Database Error: {last_error}", "sql": sql_exec, "raw_data": [], "columns": [], "suggestions": [], "report": None}), 500

                # Fix SQL with LLM
                sql_fixed = fix_sql(user_query, sql, last_error, schema_context, allowed_tables)
                if not sql_fixed or not is_safe_sql(sql_fixed):
                    continue

                ok_tables, disallowed = validate_sql_tables(sql_fixed, allowed_tables)
                if not ok_tables:
                    continue

                sql = sql_fixed
                sql_exec = ensure_rownum_limit(sql, max_rows=2000)

        # 3) Build report spec (JSON) + narrative
        report = generate_report_spec(user_query, sql, columns, rows)
        response_md = report.get("narrative_md") or "Done."

        payload = {
            "response": response_md,
            "sql": sql,
            "suggestions": (report.get("followups") or [])[:3],
            "raw_data": rows,
            "columns": columns,
            "report": report,
            "page": page,
            "meta": {
                "row_count": len(rows),
                "elapsed_ms": int((time.time() - started) * 1000),
            },
        }

        # log small preview
        preview = dict(payload)
        if preview.get("raw_data"):
            preview["raw_data"] = f"<{len(rows)} rows; sample={rows[:2]}>"
        logger.info(f"Responding: {json.dumps(preview, ensure_ascii=False)[:3000]}")
        return jsonify(payload)

    except Exception as e:
        logger.exception(e)
        return jsonify({"response": f"Server Error: {str(e)}", "sql": "", "raw_data": [], "columns": [], "suggestions": [], "report": None}), 500
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

@app.post("/export_pdf")
def export_pdf():
    """
    Generate a PDF report from returned data.
    Accepts:
    {
      "title": "...",
      "query": "...",
      "summary": "...",    // markdown/plain
      "sql": "...",
      "columns": [...],
      "raw_data": [...],
      "dashboard_png_base64": "data:image/png;base64,...." (optional)
    }
    """
    data = request.get_json(force=True) or {}
    title = (data.get("title") or "MIS AI Report").strip()
    query = (data.get("query") or "").strip()
    summary = (data.get("summary") or "").strip()
    sql = (data.get("sql") or "").strip()
    columns = data.get("columns") or []
    raw_data = data.get("raw_data") or []
    dashboard_png = data.get("dashboard_png_base64") or ""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.2*cm))

    meta_line = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    story.append(Paragraph(meta_line, styles["Normal"]))
    story.append(Spacer(1, 0.3*cm))

    if query:
        story.append(Paragraph(f"<b>Question:</b> {query}", styles["Normal"]))
        story.append(Spacer(1, 0.2*cm))

    if summary:
        story.append(Paragraph("<b>Summary</b>", styles["Heading3"]))
        # keep it simple; Paragraph supports basic HTML, not full markdown
        safe_summary = summary.replace("\n", "<br/>")
        story.append(Paragraph(safe_summary, styles["Normal"]))
        story.append(Spacer(1, 0.3*cm))

    # Optional dashboard image (client-captured)
    if dashboard_png and "base64" in dashboard_png:
        try:
            b64 = dashboard_png.split("base64,", 1)[1]
            img_bytes = base64.b64decode(b64)
            img = ImageReader(io.BytesIO(img_bytes))
            iw, ih = img.getSize()
            max_w = A4[0] - 3.0*cm
            scale = min(1.0, max_w / float(iw))
            story.append(Paragraph("<b>Dashboard Snapshot</b>", styles["Heading3"]))
            story.append(Spacer(1, 0.2*cm))
            story.append(Image(img, width=iw*scale, height=ih*scale))
            story.append(Spacer(1, 0.4*cm))
        except Exception as e:
            logger.warning(f"Could not embed dashboard image: {e}")

    if sql:
        story.append(Paragraph("<b>SQL</b>", styles["Heading3"]))
        # Use a small mono-ish style via <font>
        safe_sql = sql.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        story.append(Paragraph(f"<font face='Courier' size='8'>{safe_sql}</font>", styles["Normal"]))
        story.append(Spacer(1, 0.4*cm))

    # Table (truncate to keep PDF readable)
    if columns and raw_data:
        story.append(Paragraph("<b>Data (preview)</b>", styles["Heading3"]))
        story.append(Spacer(1, 0.2*cm))

        max_cols = 8
        max_rows = 30
        cols_used = columns[:max_cols]
        data_rows = raw_data[:max_rows]

        table_data = [cols_used]
        for r in data_rows:
            row = r[:max_cols] + ([""] * max(0, max_cols - len(r)))
            table_data.append([str(x) for x in row[:max_cols]])

        t = Table(table_data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E5E7EB")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2*cm))

        if len(columns) > max_cols or len(raw_data) > max_rows:
            story.append(Paragraph(
                f"<i>Note: showing first {min(max_rows, len(raw_data))} rows and first {min(max_cols, len(columns))} columns.</i>",
                styles["Normal"],
            ))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="MIS_AI_Report.pdf", mimetype="application/pdf")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)
