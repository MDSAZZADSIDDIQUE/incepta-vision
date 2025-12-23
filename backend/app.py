import os
import oracledb
import re
import logging
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
import io

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# --- 1. SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [STEP] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen-2.5-32b") 
client = Groq(api_key=GROQ_API_KEY)

try:
    oracle_home = os.getenv("ORACLE_HOME")
    if oracle_home:
        lib_dir = os.path.join(oracle_home, "bin")
        oracledb.init_oracle_client(lib_dir=lib_dir)
        logger.info(f"Oracle Client initialized from: {lib_dir}")
except Exception as e:
    logger.error(f"Warning: Could not init Oracle Client. Error: {e}")

# --- FEW-SHOT EXAMPLES ---
TRAINING_EXAMPLES = """
### 🌟 GOLDEN EXAMPLES: HOW TO THINK & WRITE SQL

1. **Scenario: Stock & Inventory (Crucial: Use DASH_NATIONAL_REPORT)**
   User: "What is the current stock of Tymol in Kushtia depot?"
   <think>
   - Keyword "Stock" -> Table `DASH_NATIONAL_REPORT`.
   - Depot Name "Kushtia" -> Column `NAME`.
   - Product "Tymol" -> Column `DESCRIPTION`.
   </think>
   SQL: ```sql SELECT QTY_STOCK FROM MIS.DASH_NATIONAL_REPORT WHERE UPPER(DESCRIPTION) LIKE '%TYMOL%' AND UPPER(NAME) LIKE '%KUSHTIA%' ```

2. **Scenario: Granular Daily Sales (Crucial: Use DEPOT_SALE)**
   User: "Show me the daily sales for Territory MZR-60."
   <think>
   - Keyword "Daily" + "Territory" -> Table `DEPOT_SALE`.
   - Column `TERR_ID` matches "MZR-60".
   </think>
   SQL: ```sql SELECT REPORT_DATE, TERR_ID, P_CODE, SOLD_VALUE FROM MIS.DEPOT_SALE WHERE TERR_ID = 'MZR-60' ORDER BY REPORT_DATE DESC ```

3. **Scenario: WB03 Specific Request (Crucial: Use _WB03 Tables)**
   User: "What is the Team Target for the WB03 group?"
   <think>
   - Keyword "WB03" + "Team Target" -> Table `YEAR_WISE_TEAM_PERFOR_WB03_B04`.
   </think>
   SQL: ```sql SELECT TEAM, YR_TGTVAL, YR_ACHV FROM MIS.YEAR_WISE_TEAM_PERFOR_WB03_B04 ORDER BY YR_ACHV DESC ```

4. **Scenario: Manager/Leadership Performance**
   User: "How is GM Ashraf performing?"
   <think>
   - Keyword "GM" (General Manager) + Name "Ashraf" -> Table `GM_SM_SALES_ANALYSIS_ACH`.
   </think>
   SQL: ```sql SELECT NAME, TARGET_VALUE, SALES_VALUE, ACHIEVEMENT FROM MIS.GM_SM_SALES_ANALYSIS_ACH WHERE SALES_PERSON = 'GM' AND UPPER(NAME) LIKE '%ASHRAF%' ```

5. **Scenario: Monthly Business Unit Breakdown**
   User: "Show me the monthly sales trend for Animal Health."
   <think>
   - Keyword "Monthly" + "Animal Health" -> Table `COMPANY_WISE_SALES_SUMMARY`.
   - Column `IPL_ANIMAL_HEALTH`.
   </think>
   SQL: ```sql SELECT SALES_MONTH, IPL_ANIMAL_HEALTH FROM MIS.COMPANY_WISE_SALES_SUMMARY ORDER BY SALES_MONTH DESC ```

6. **Scenario: National Quantity Targets**
   User: "What is the target quantity for Cef-3?"
   <think>
   - Keyword "Target" + "Quantity" (implied by unit count) -> Table `NATIONAL_TARGET_SALES_ACHVM`.
   </think>
   SQL: ```sql SELECT P_NAME, PACK_S, N_TGT_QTY, N_SALES_QTY, ACHIEVEMENT FROM MIS.NATIONAL_TARGET_SALES_ACHVM WHERE UPPER(P_NAME) LIKE '%CEF-3%' ```

7. **Scenario: Market Type (Mitford/Retail)**
   User: "How are Mitford sales compared to Retail?"
   <think>
   - Keyword "Mitford" / "Retail" -> Table `DHK_GRP_MKT_WISE_SALES`.
   </think>
   SQL: ```sql SELECT REPORT_DATE, MITFORD, DHAKA_RETAIL FROM MIS.DHK_GRP_MKT_WISE_SALES ORDER BY REPORT_DATE DESC ```
   
8. **Scenario: Monthly Breakdown of Yearly Totals (Crucial)**
   User: "Show total sales summary for 2024 by month."
   <think>
   - Keyword "Monthly Summary" + "Year" -> Table `MONTH_WISE_SALES_REPORT`.
   - This table already has columns JAN, FEB, MAR, etc.
   - Do NOT use UNION ALL or formulas with "...". Select the columns directly.
   </think>
   SQL: ```sql SELECT SALES_YEAR, JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC FROM MIS.MONTH_WISE_SALES_REPORT WHERE SALES_YEAR = 2024 ```
"""

# --- HELPER: SQL SAFETY CHECK ---
def is_safe_sql(sql):
    forbidden_keywords = ["DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", "ALTER", "GRANT", "REVOKE", "CREATE", "REPLACE"]
    sql_upper = sql.upper()
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            logger.warning(f"SECURITY BLOCK: Found forbidden keyword '{kw}' in SQL: {sql}")
            return False
    return True

# --- HELPER: DB CONNECTION ---
def get_db_connection():
    dsn = oracledb.makedsn(
        os.getenv("DB_HOST"),
        os.getenv("DB_PORT"),
        service_name=os.getenv("DB_SERVICE")
    )
    return oracledb.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dsn=dsn
    )

# --- HELPER: SMART SCHEMA LOADER ---
def load_schema_context():
    logger.info("Loading Dictionary Schema...")
    try:
        if not os.path.exists('dictionary.json'):
            logger.error("dictionary.json not found!")
            return "Schema not found."
        with open('dictionary.json', 'r') as f:
            schema_data = json.load(f)
        
        context_str = "### DATABASE SCHEMA (MIS USER)\n\n"
        for table_name, details in schema_data.items():
            clean_name = table_name.upper()
            if not clean_name.startswith("MIS.") and not clean_name.startswith("DWH."):
                clean_name = f"MIS.{clean_name}"
            
            context_str += f"**TABLE: {clean_name}**\n"
            context_str += f"   - Description: {details.get('description', '')}\n"
            if 'columns' in details:
                context_str += "   - Cols: " + ", ".join([f"{k}" for k in details['columns'].keys()]) + "\n"
            context_str += "\n"
        return context_str
    except Exception as e:
        logger.error(f"Error loading dictionary.json: {e}")
        return "Error loading schema dictionary."

SCHEMA_CONTEXT_DYNAMIC = load_schema_context()

# --- MASTER ROUTING PROMPT ---
SYSTEM_PROMPT = f"""
You are an Oracle 11g SQL Expert. Your goal is 100% accuracy in table selection.

{SCHEMA_CONTEXT_DYNAMIC}

{TRAINING_EXAMPLES}

### CRITICAL TABLE SELECTION RULES (Follow this Priority)

1. **"WB03" or "B04" Queries (Top Priority):**
   - If user mentions "WB03", "B04", or "Special Group":
     - Team Target/Achievement -> `YEAR_WISE_TEAM_PERFOR_WB03_B04`
     - General Sales -> Use other `_WB03_B04` tables.

2. **Supply Chain & Procurement (New):**
   - "Blocklist", "Import", "Supplier", "Raw Material", "API", "Manufacturer" -> `SCM_APP_BLOCKLIST`.
   - *Example: "Who supplies Cinnarizine?" -> SCM_APP_BLOCKLIST.*

3. **Daily & Operational Sales (Detailed Tier):**
   - **National Product Target (Quantity):** "Target Qty", "Units Sold", "Pack Size" -> `NATIONAL_TARGET_SALES_ACHVM`.
   - **Stock/Inventory:** "Stock", "Inventory", "Available Qty", "Trade Price" -> `DASH_NATIONAL_REPORT`.
   - **Depot Names:** Refers to Depots by NAME (e.g., "Comilla Sales") -> `DASH_NATIONAL_REPORT`.
   - **Market Type:** "Mitford", "Wholesale", "Retail" -> `DHK_GRP_MKT_WISE_SALES`.
   - **Granular:** "Daily sales for Territory X" or "Product Code Y" -> `DEPOT_SALE`.
   - **Depot Group Level:** "Depot Group Target" or "Sales by Depot ID" -> `DEPOT_GROUP_WISE_SALES`.
   - **Global/MTD:** "Today's Sales", "Global MTD" -> `PRODUCT_GROUP_WISE_SALES`.

4. **Monthly Business Unit & Entity Breakdown:**
   - "Animal Health", "Human Vaccine", "Diaper", "Infusion" Monthly Breakdown -> `COMPANY_WISE_SALES_SUMMARY`.

5. **Entity/Company Queries (IPL, IVL, IHHL):**
   - **Yearly** General Totals -> `ALL_COMPANY_SALES`
   - Company + Channel -> `ALL_COMPANY_SALES_CHANNEL_WISE`

6. **Sales Channel Queries (Depot, Inst, Export):**
   - Annual Totals -> `MONTH_WISE_SALES_REPORT`
   - Sub-groups (B01, A01) -> `SUMMARY_OF_SALES`

7. **Ranking, Performance & Hierarchy:**
   - **Hierarchy/Reporting Lines:** "Who reports to whom?", "ASM Sales", "NSM Performance" -> `GROUP_WISE_SALES_RM_TO_GM`.
   - **Leadership (GM/SM Yearly):** "GM", "SM", "DSM" Yearly Target -> `GM_SM_SALES_ANALYSIS_ACH`.
   - **Field Force (RM Yearly):** "RM Yearly Target/Ach" -> `RM_SALES_ANALYSIS_ACH`.
   - "Top 10" (Monthly) -> `PRODUCT_RANKING_DEPOT_SALES`
   - "Top 10" (Yearly) -> `BRAND_RANKING_YEARLY`
   - "Team Growth" -> `TEAM_WISE_SALES_GROWTH`
   - "Team Target" (General) -> `YEAR_WISE_TEAM_PERFORMANCE`

8. **Detailed Geography & SKU:**
   - "Region/Area/Territory" -> `BRAND_WISE_SALES_DATA`
   - "Brand vs SKU" -> `BRAND_WISE_SALES_REPORT`

9. **Master Data:**
   - Plant Names -> `EXPO_PLANT`
   - User Profiles -> `DASHBOARD_USERS_INFO`

### COMMON ID MAPPINGS (Reference)
- **Regions:** Mymensingh='MZR-00', Sylhet='SYL-00', Comilla='COM-00', Chittagong='CTG-00', Dhaka='DHK-00'.
- **Product Groups:** 'KINETIX', 'ZYMOS', 'CELLBIOTIC', 'ASTER'.
- **Teams:** 'GENERAL-TEAM', 'AST-GYR', 'OPR-XEN'.
- **Reporting:** GM=General Manager, SM=Sales Manager, RM=Regional Manager.

### SYNTAX RULES
- **Limit:** Use `WHERE ROWNUM <= N`. NEVER use `LIMIT`.
- **Dates:** `DASH_NATIONAL_REPORT`, `SCM_APP_BLOCKLIST` use `REPORT_DATE`, `WDATE`, or `BLOCKLIST_DATE`. Others use `REPORT_DATE` or `SALES_MONTH`.
- **Strings:** Use `UPPER(col) LIKE UPPER('%val%')`.
- **Completeness:** NEVER use "..." or placeholders in SQL. Write out the full query.

### OUTPUT FORMAT
Return **ONLY** the SQL query inside ```sql ... ``` tags.
"""

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('message')
    logger.info(f"START: Received Query: {user_query}")

    # --- 1. DYNAMIC DATE INJECTION ---
    today = datetime.now()
    current_date_str = today.strftime("%d-%b-%y").upper()
    current_month_str = today.strftime("%b-%y").upper()
    first_of_this_month = today.replace(day=1)
    last_month_obj = (first_of_this_month - timedelta(days=1))
    last_month_str = last_month_obj.strftime("%b-%y").upper()

    DATE_CONTEXT = f"""
    ### TEMPORAL CONTEXT
    - Today's Date: {current_date_str}
    - Current Month: {current_month_str}
    - Previous Month: {last_month_str}
    - If user asks for "Today", use `REPORT_DATE = '{current_date_str}'` (or WDATE).
    - If user asks for "Last Month", use `SALES_MONTH = '{last_month_str}'`.
    """
    logger.info(f"Date Context: Today={current_date_str}, LastMonth={last_month_str}")

    FULL_SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + DATE_CONTEXT

    # --- 2. GENERATE SQL (FIRST ATTEMPT) ---
    try:
        logger.info("Calling LLM for Initial SQL Generation...")
        completion = client.chat.completions.create(
            model=GROQ_MODEL, 
            messages=[
                {"role": "system", "content": FULL_SYSTEM_PROMPT},
                {"role": "user", "content": f"Request: {user_query}\n\nTrace your logic in <think> tags: 1. Is it Supply Chain? 2. Is it Sales/Target? 3. Select Table. 4. Write SQL."}
            ],
            temperature=0.1 
        )
        raw_content = completion.choices[0].message.content.strip()
        logger.info(f"LLM Response (Raw): {raw_content[:200]}...") # Log first 200 chars
    except Exception as e:
        logger.error(f"LLM API Error: {str(e)}")
        return jsonify({"response": f"LLM Error: {str(e)}", "sql": "", "suggestions": []})

    # --- SQL Extraction ---
    clean_sql = ""
    code_match = re.search(r'```(?:sql)?\s*(.*?)```', raw_content, re.DOTALL | re.IGNORECASE)
    if code_match:
        clean_sql = code_match.group(1).strip()
    else:
        clean_sql = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
        clean_sql = re.sub(r'^.*?SELECT', 'SELECT', clean_sql, flags=re.DOTALL | re.IGNORECASE)

    if clean_sql.endswith(";"):
        clean_sql = clean_sql[:-1]
    
    logger.info(f"Extracted SQL: {clean_sql}")

    # --- 3. SAFETY CHECK ---
    if not is_safe_sql(clean_sql):
        return jsonify({"response": "Security Alert: Unsafe SQL blocked.", "sql": clean_sql})

    # --- 4. EXECUTE SQL (WITH SELF-HEALING) ---
    MAX_RETRIES = 2
    attempt = 0
    results = []
    columns = []

    while attempt < MAX_RETRIES:
        try:
            logger.info(f"DB Execution Attempt {attempt + 1}")
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(clean_sql)
            
            if cursor.description:
                columns = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
                logger.info(f"Rows fetched: {len(rows)}")

                # --- FUZZY RETRY (0 ROWS) ---
                if not rows and "WHERE" in clean_sql.upper() and attempt == 0:
                    logger.warning("0 Rows returned. Triggering Fuzzy Retry...")
                    
                    retry_prompt = f"""
                    The query returned 0 rows. It might be a spelling or case sensitivity issue.
                    
                    Original SQL: {clean_sql}
                    User Query: {user_query}
                    
                    Fix:
                    1. Use `UPPER(col) LIKE UPPER('%val%')` for string comparisons.
                    2. If searching for a name (e.g., 'Kustia'), try correcting it (e.g., 'Kushtia').
                    3. Return ONLY the corrected SQL.
                    """
                    
                    retry_resp = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": retry_prompt}]
                    )
                    new_sql = retry_resp.choices[0].message.content.strip()
                    new_sql = re.sub(r'```(?:sql)?\s*(.*?)```', r'\1', new_sql, flags=re.DOTALL).strip()
                    if new_sql.endswith(";"): new_sql = new_sql[:-1]
                    
                    logger.info(f"Fuzzy SQL Generated: {new_sql}")
                    clean_sql = new_sql # Update for next loop iteration
                    
                    cursor.close()
                    conn.close()
                    attempt += 1
                    continue # Try loop again with new SQL

                # Process Rows
                for row in rows:
                    processed_row = []
                    for item in row:
                        if isinstance(item, oracledb.LOB):
                            processed_row.append(str(item.read()))
                        else:
                            processed_row.append(str(item) if item is not None else "")
                    results.append(processed_row)

            cursor.close()
            conn.close()
            break # Success, break loop

        except oracledb.DatabaseError as e:
            error_obj, = e.args
            error_msg = error_obj.message
            logger.error(f"DB Error on Attempt {attempt+1}: {error_msg}")

            if attempt == MAX_RETRIES - 1:
                return jsonify({"response": f"Database Error: {error_msg}", "sql": clean_sql, "suggestions": []})

            # --- SCHEMA CORRECTION ---
            correction_prompt = f"""
            The SQL query failed with this Oracle Error: {error_msg}
            
            Bad SQL: {clean_sql}
            
            **CRITICAL FIXES:**
            1. Did you use "..." or placeholders? Remove them and write the full logic.
            2. Did you use a column that doesn't exist? Review the schema below.
            
            {SCHEMA_CONTEXT_DYNAMIC}
            
            Return ONLY the fixed SQL.
            """
            
            logger.info("Requesting LLM Schema Fix...")
            fix_resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": correction_prompt}]
            )
            clean_sql = fix_resp.choices[0].message.content.strip()
            clean_sql = re.sub(r'```(?:sql)?\s*(.*?)```', r'\1', clean_sql, flags=re.DOTALL).strip()
            if clean_sql.endswith(";"): clean_sql = clean_sql[:-1]
            logger.info(f"Fixed SQL Generated: {clean_sql}")
            
            attempt += 1

    # --- 5. SUMMARIZE ---
    safe_context_data = str(results[:5])[:2000]
    logger.info("Generating Summary...")
    
    summary_prompt = f"""
    User Request: {user_query}
    SQL Used: {clean_sql}
    Data Sample: {safe_context_data}
    
    Task 1: Provide a 1-sentence answer.
    Task 2: Suggest 3 follow-up questions.
    
    Format:
    [Answer]
    <Answer>
    
    [Suggestions]
    <Q1>|<Q2>|<Q3>
    """
    
    try:
        summary_completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise data assistant."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        full_response = summary_completion.choices[0].message.content.strip()
        
        answer = "Data retrieved successfully."
        suggestions = []
        
        if "[Answer]" in full_response:
            _, relevant_content = full_response.split("[Answer]", 1)
            if "[Suggestions]" in relevant_content:
                ans_part, sugg_part = relevant_content.split("[Suggestions]", 1)
                answer = ans_part.strip()
                suggestions = [s.strip() for s in sugg_part.split('|') if s.strip()]
            else:
                answer = relevant_content.strip()
        else:
            answer = full_response
            
    except Exception as e:
        logger.error(f"Summary Error: {e}")
        answer = "Data retrieved successfully."
        suggestions = ["Show Details", "Export PDF"]

    logger.info("Response Sent.")
    return jsonify({
        "response": answer, 
        "sql": clean_sql, 
        "suggestions": suggestions[:3], 
        "raw_data": results, 
        "columns": columns
    })

@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    logger.info("PDF Export Requested")
    data = request.json
    raw_data = data.get('raw_data', [])
    columns = data.get('columns', [])
    summary = data.get('summary', 'No summary provided.')
    query = data.get('query', 'Report')

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Executive Data Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Query:</b> {query}", styles['Normal']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Summary:</b> {summary}", styles['Normal']))
    elements.append(Spacer(1, 20))

    if raw_data and columns:
        table_data = [columns] + raw_data[:50]
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)

    doc.build(elements)
    buffer.seek(0)
    logger.info("PDF Generated")
    return send_file(buffer, as_attachment=True, download_name='report.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True, port=5000)