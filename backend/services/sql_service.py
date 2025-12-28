"""
SQL generation and validation service using LLM.
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from groq import Groq

from config import config

logger = logging.getLogger(__name__)


class SQLService:
    """Service for SQL generation, validation, and safety checks."""
    
    def __init__(self, llm_client: Groq):
        """
        Initialize SQL service.
        
        Args:
            llm_client: Groq LLM client instance
        """
        self.llm = llm_client
    
    def is_safe_sql(self, sql: str) -> bool:
        """
        Check if SQL contains forbidden keywords.
        
        Args:
            sql: SQL query to check
            
        Returns:
            True if safe, False if contains forbidden keywords
        """
        sql_upper = (sql or "").upper()
        for keyword in config.FORBIDDEN_SQL_KEYWORDS:
            if re.search(rf"\b{keyword}\b", sql_upper):
                logger.warning(f"Unsafe SQL detected: contains {keyword}")
                return False
        return True
    
    def normalize_table_name(self, name: str) -> str:
        """
        Normalize table name to uppercase with schema.
        
        Args:
            name: Table name to normalize
            
        Returns:
            Normalized table name (e.g., 'MIS.TABLE_NAME')
        """
        n = (name or "").strip().strip('"').upper()
        if not n:
            return n
        # Keep DB link suffix if present
        if "." not in n and n not in ("DUAL",):
            n = f"MIS.{n}"
        return n
    
    def extract_table_references(self, sql: str) -> List[str]:
        """
        Extract table names referenced in SQL query.
        
        Args:
            sql: SQL query
            
        Returns:
            List of normalized table names
        """
        if not sql:
            return []
        
        # Remove comments and string literals
        s = self._strip_sql_comments(sql)
        s = re.sub(r"'.*?'", "''", s, flags=re.DOTALL)
        su = s.upper()
        
        # Capture chunks after FROM/JOIN
        boundaries = (
            r"\bWHERE\b|\bGROUP\b|\bORDER\b|\bHAVING\b|"
            r"\bUNION\b|\bCONNECT\b|\bSTART\b|\bFETCH\b|"
            r"\bFOR\b|\bMODEL\b|\bQUALIFY\b"
        )
        refs: List[str] = []
        
        for kw in ["FROM", "JOIN"]:
            for m in re.finditer(rf"\b{kw}\b\s+", su):
                start = m.end()
                b = re.search(boundaries, su[start:])
                end = start + (b.start() if b else len(su[start:]))
                chunk = su[start:end].strip()
                
                # Ignore subquery starts
                if chunk.startswith("("):
                    continue
                
                # Split by commas (legacy joins)
                for part in chunk.split(","):
                    token = part.strip().split()[0] if part.strip() else ""
                    if not token or token.startswith("("):
                        continue
                    token = token.rstrip(");")
                    refs.append(token)
        
        # Normalize and deduplicate
        normed = []
        seen = set()
        for r in refs:
            rn = self.normalize_table_name(r)
            if rn and rn not in seen:
                seen.add(rn)
                normed.append(rn)
        return normed
    
    def validate_sql_tables(
        self, 
        sql: str, 
        allowed_tables: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that SQL only references allowed tables.
        
        Args:
            sql: SQL query to validate
            allowed_tables: List of allowed table names
            
        Returns:
            Tuple of (is_valid, disallowed_tables)
        """
        allowed_set = {self.normalize_table_name(t) for t in allowed_tables}
        allowed_set.add("DUAL")  # Always allow Oracle DUAL
        
        used = self.extract_table_references(sql)
        disallowed = [t for t in used if t not in allowed_set]
        
        return (len(disallowed) == 0), disallowed
    
    def ensure_rownum_limit(self, sql: str, max_rows: int = None) -> str:
        """
        Add ROWNUM limit to SQL if not present.
        
        Args:
            sql: SQL query
            max_rows: Maximum rows to return (default from config)
            
        Returns:
            SQL with ROWNUM limit
        """
        if max_rows is None:
            max_rows = config.MAX_ROWS
        
        su = sql.upper()
        if re.search(r"\bROWNUM\b", su) or re.search(r"\bFETCH\s+FIRST\b", su):
            return sql
        
        # Wrap as subquery
        return f"SELECT * FROM (\n{sql}\n) WHERE ROWNUM <= {max_rows}"
    
    def extract_sql_from_llm(self, text: str) -> str:
        """
        Extract SQL from LLM response.
        
        Args:
            text: LLM response text
            
        Returns:
            Extracted SQL query
        """
        if not text:
            return ""
        
        # Preferred: ```sql ... ```
        m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            sql = m.group(1).strip()
            return sql[:-1] if sql.endswith(";") else sql
        
        # Fallback: try to find first SELECT
        m2 = re.search(r"\bSELECT\b.*", text, flags=re.DOTALL | re.IGNORECASE)
        if m2:
            sql = m2.group(0).strip()
            return sql[:-1] if sql.endswith(";") else sql
        
        return ""
    
    def generate_sql(
        self, 
        user_query: str, 
        schema_context: str, 
        allowed_tables: List[str]
    ) -> str:
        """
        Generate SQL query using LLM.
        
        Args:
            user_query: Natural language query from user
            schema_context: Schema information for context
            allowed_tables: List of allowed tables
            
        Returns:
            Generated SQL query
        """
        system_prompt = self._build_system_prompt(schema_context, allowed_tables)
        
        completion = self.llm.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            temperature=0.1,
        )
        
        return self.extract_sql_from_llm(
            completion.choices[0].message.content or ""
        )
    
    def fix_sql(
        self,
        user_query: str,
        bad_sql: str,
        error_msg: str,
        schema_context: str,
        allowed_tables: List[str]
    ) -> str:
        """
        Fix SQL that failed execution.
        
        Args:
            user_query: Original user query
            bad_sql: SQL that failed
            error_msg: Error message from database
            schema_context: Schema information
            allowed_tables: List of allowed tables
            
        Returns:
            Fixed SQL query
        """
        system_prompt = self._build_system_prompt(schema_context, allowed_tables)
        
        fix_prompt = f"""The SQL failed on Oracle with this error:
{error_msg}

Fix the SQL. Keep it Oracle 11g. Use ONLY allowed tables.

If the error is ORA-00904 (invalid identifier) and it is caused by ORDER BY on an alias, 
prefer ORDER BY column position (e.g., ORDER BY 2) or repeat the expression.

User question: {user_query}

Bad SQL:
```sql
{bad_sql}
```
"""
        
        completion = self.llm.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": fix_prompt},
            ],
            temperature=0.1,
        )
        
        return self.extract_sql_from_llm(
            completion.choices[0].message.content or ""
        )
    
    def _strip_sql_comments(self, sql: str) -> str:
        """Remove SQL comments."""
        # Remove /* ... */ and -- ... endline
        s = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
        s = re.sub(r"--.*?$", " ", s, flags=re.MULTILINE)
        return s
    
    def _build_system_prompt(self, schema_context: str, allowed_tables: List[str]) -> str:
        """Build system prompt for SQL generation."""
        from datetime import datetime
        today = datetime.now()
        allowed_lines = "\n".join([f"- {self.normalize_table_name(t)}" for t in allowed_tables])
        
        return f"""You are an Oracle 11g SQL expert. Your goal is 100% accurate table selection and readable outputs.

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
