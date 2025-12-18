import os
import oracledb
import re
import logging
import json
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

# Setup basic console logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Oracle Initialization
try:
    oracle_home = os.getenv("ORACLE_HOME")
    if oracle_home:
        lib_dir = os.path.join(oracle_home, "bin")
        oracledb.init_oracle_client(lib_dir=lib_dir)
        logger.info(f"Oracle Client initialized from: {lib_dir}")
except Exception as e:
    logger.error(f"Warning: Could not init Oracle Client. Error: {e}")

# --- Load Semantic Layer ---
def load_schema_context():
    try:
        if not os.path.exists('dictionary.json'):
            return "Schema not found."
        with open('dictionary.json', 'r') as f:
            schema_data = json.load(f)
        
        context_str = "### DATABASE SCHEMA (MIS USER)\n\n"
        for table_name, details in schema_data.items():
            context_str += f"**TABLE: MIS.{table_name}**\n"
            context_str += f"   - Description: {details.get('description', '')}\n"
            if 'synonyms' in details:
                context_str += f"   - Keywords: {', '.join(details['synonyms'])}\n"
            if 'columns' in details:
                context_str += "   - Important Columns:\n"
                for col_name, col_desc in details['columns'].items():
                    context_str += f"     - `{col_name}`: {col_desc}\n"
            context_str += "\n"
        return context_str
    except Exception as e:
        logger.error(f"Error loading dictionary.json: {e}")
        return "Error loading schema dictionary."

SCHEMA_CONTEXT_DYNAMIC = load_schema_context()

# --- UPDATED PROMPT WITH ORACLE 11g RULES ---
SYSTEM_PROMPT = f"""
You are an Oracle 11g SQL Expert and Data Analyst. 
{SCHEMA_CONTEXT_DYNAMIC}

### CRITICAL RULES
1. **Date Handling:** ALWAYS use `TO_DATE('YYYY-MM-DD', 'YYYY-MM-DD')` for comparisons.
2. **Text vs Date:** Check the dictionary. If a column is TEXT (e.g., 'JAN-21'), use `LIKE` or String matching.
3. **No LIMIT Clause:** Oracle 11g DOES NOT support `LIMIT`.
   - **Incorrect:** `SELECT ... LIMIT 1`
   - **Correct (Top N):** `SELECT * FROM (SELECT ... ORDER BY ... DESC) WHERE ROWNUM <= 1`
   - **Correct (Sample):** `SELECT ... WHERE ROWNUM <= 5`
4. **Reasoning:** Wrap reasoning in <think> tags.
5. **Output:** Return ONLY the SQL query.
"""

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

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('message')
    logger.info(f"Received query: {user_query}")

    # 1. Generate SQL
    try:
        completion = client.chat.completions.create(
            model="qwen/qwen3-32b", 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Request: {user_query}\n\nThink step-by-step in <think> tags. Then provide SQL."}
            ],
            temperature=0.1 
        )
        raw_content = completion.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"response": f"LLM Error: {str(e)}", "sql": "", "thought_process": ""})

    # --- Process Response ---
    thought_process = "No reasoning provided."
    think_match = re.search(r'<think>(.*?)</think>', raw_content, flags=re.DOTALL)
    if think_match:
        thought_process = think_match.group(1).strip()
    
    clean_sql = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
    clean_sql = re.sub(r'```sql', '', clean_sql, flags=re.IGNORECASE).replace("```", "").strip()
    if clean_sql.endswith(";"):
        clean_sql = clean_sql[:-1]
        
    logger.info(f"Executing SQL: {clean_sql}")

    # 2. Execute SQL
    results = []
    columns = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(clean_sql)
        if cursor.description:
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
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
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return jsonify({"response": f"Database Error: {str(e)}", "sql": clean_sql, "thought_process": thought_process})

    # 3. Summarize
    safe_context_data = str(results[:10])[:3000]
    
    summary_prompt = f"""
    User Request: {user_query}
    SQL Used: {clean_sql}
    Data Sample: {safe_context_data}
    Total Rows Retrieved: {len(results)}

    Write a professional business summary of this data. 
    - Highlight key metrics.
    - Format with Markdown.
    - Keep it concise.
    """
    
    try:
        summary_completion = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {"role": "system", "content": "You are a Senior Data Analyst."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        bot_response = summary_completion.choices[0].message.content
    except Exception as e:
        bot_response = "Data retrieved successfully."

    return jsonify({
        "response": bot_response, 
        "sql": clean_sql, 
        "thought_process": thought_process, 
        "raw_data": results, 
        "columns": columns
    })

@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    data = request.json
    raw_data = data.get('raw_data', [])
    columns = data.get('columns', [])
    summary = data.get('summary', 'No summary provided.')
    query = data.get('query', 'Report')

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Executive Data Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # Query Info
    elements.append(Paragraph(f"<b>Query:</b> {query}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Summary Section
    elements.append(Paragraph("<b>Analysis & Summary:</b>", styles['Heading3']))
    clean_summary = summary.replace("**", "") 
    elements.append(Paragraph(clean_summary, styles['Normal']))
    elements.append(Spacer(1, 20))

    # Data Table
    if raw_data and columns:
        elements.append(Paragraph("<b>Data Detail (First 50 Rows):</b>", styles['Heading3']))
        elements.append(Spacer(1, 10))
        
        table_data = [columns] + raw_data[:50]
        
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)

    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='extensive_report.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True, port=5000)