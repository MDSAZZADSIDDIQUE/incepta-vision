"""
Database service for managing Oracle database connections and operations.
"""
import oracledb
from typing import List, Tuple, Any, Optional
import logging
from contextlib import contextmanager

from config import config

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for database operations."""
    
    def __init__(self):
        """Initialize database service and Oracle client."""
        self._init_oracle_client()
    
    def _init_oracle_client(self) -> None:
        """Initialize Oracle client in thick mode if ORACLE_HOME is set."""
        try:
            if config.ORACLE_HOME:
                oracledb.init_oracle_client(
                    lib_dir=os.path.join(config.ORACLE_HOME, "bin")
                )
                logger.info("Oracle client initialized in thick mode")
        except Exception as e:
            logger.warning(f"Could not init Oracle client (thick mode): {e}")
            logger.info("Using thin mode")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            oracledb.Connection: Database connection
            
        Example:
            with db_service.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        conn = None
        try:
            dsn = oracledb.makedsn(
                config.DB_HOST,
                config.DB_PORT,
                service_name=config.DB_SERVICE,
            )
            conn = oracledb.connect(
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                dsn=dsn,
            )
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
    
    def execute_query(
        self, 
        sql: str, 
        params: Optional[dict] = None
    ) -> Tuple[List[str], List[List[Any]]]:
        """
        Execute a SQL query and return results.
        
        Args:
            sql: SQL query to execute
            params: Optional query parameters
            
        Returns:
            Tuple of (column_names, rows)
            
        Raises:
            oracledb.DatabaseError: If query execution fails
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)
                
                # Get column names
                columns = [d[0] for d in (cursor.description or [])]
                
                # Fetch and process rows
                rows = []
                for row in cursor.fetchall():
                    processed = []
                    for item in row:
                        if item is None:
                            processed.append("")
                        elif isinstance(item, (bytes, bytearray)):
                            processed.append(item.decode(errors="ignore"))
                        else:
                            # Handle CLOB safely
                            try:
                                if hasattr(item, "read"):
                                    processed.append(str(item.read()))
                                else:
                                    processed.append(str(item))
                            except Exception:
                                processed.append(str(item))
                    rows.append(processed)
                
                return columns, rows
                
            finally:
                cursor.close()
    
    def introspect_columns(self, table_name: str) -> List[str]:
        """
        Get column names for a table without fetching data.
        
        Args:
            table_name: Fully qualified table name (e.g., 'MIS.TABLE_NAME')
            
        Returns:
            List of column names
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Use 1=0 to get structure without data
                cursor.execute(f"SELECT * FROM {table_name} WHERE 1=0")
                if cursor.description:
                    return [d[0] for d in cursor.description if d and d[0]]
                return []
            except Exception as e:
                logger.warning(f"Could not introspect columns for {table_name}: {e}")
                return []
            finally:
                cursor.close()


# Singleton instance
db_service = DatabaseService()
