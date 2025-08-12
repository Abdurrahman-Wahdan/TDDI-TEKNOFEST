"""
Database Connection for MCP Client
Step 1: Simple PostgreSQL connection
"""

import psycopg2
import psycopg2.extras
import logging
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Simple PostgreSQL connection manager
    """
    
    def __init__(
        self, 
        host: str = os.getenv("LOCAL_DB_HOST", "localhost"),  
        port: int = os.getenv("LOCAL_DB_PORT", 5432),  
        database: str = os.getenv("LOCAL_DB_NAME","tddi"),                  
        username: str = os.getenv("LOCAL_DB_USERNAME","tddi"),     
        password: str = ""                         # ✅ Empty (no password needed)
    ):
        """
        Initialize database connection parameters.
        
        Args:
            host: Database host (default: localhost)
            port: Database port (default: 5432)
            database: Database name
            username: Database username
            password: Database password
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.connection = None
        
        logger.info(f"Database connection configured for {host}:{port}/{database}")
    
    def connect(self) -> bool:
        """
        Establish database connection.
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                cursor_factory=psycopg2.extras.RealDictCursor  # Returns dict-like rows
            )
            
            # Test connection
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                logger.info(f"Connected to PostgreSQL: {version['version']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results.
        
        Args:
            query: SQL query to execute
            params: Query parameters (optional)
            
        Returns:
            List of dictionaries representing rows
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    def execute_single(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """
        Execute a query and return single result.
        
        Args:
            query: SQL query to execute
            params: Query parameters (optional)
            
        Returns:
            Dictionary representing single row or None
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
                return dict(result) if result else None
                
        except Exception as e:
            logger.error(f"Error executing single query: {e}")
            return None
    
    def is_connected(self) -> bool:
        """Check if database connection is active"""
        try:
            if self.connection:
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT 1;")
                    return True
            return False
        except:
            return False


# Global database instance
db = DatabaseConnection()


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    print("🔌 Testing Database Connection...")
    print("=" * 40)
    
    # YOU NEED TO UPDATE THESE VALUES:
    print("⚠️  BEFORE RUNNING:")
    print("1. Update database name in DatabaseConnection()")
    print("2. Update username if not 'postgres'")
    print("3. Update password")
    print()
    
    # Test connection
    success = db.connect()
    
    if success:
        print("✅ Connection successful!")
        
        # Test basic query
        print("\n📊 Testing basic query...")
        result = db.execute_single("SELECT current_database(), current_user;")
        if result:
            print(f"   Database: {result['current_database']}")
            print(f"   User: {result['current_user']}")
        
        # Test table existence
        print("\n📋 Checking if your tables exist...")
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name IN ('customers', 'plans', 'customer_plans', 'billing', 'technical_appointments')
        ORDER BY table_name;
        """
        
        tables = db.execute_query(tables_query)
        if tables:
            print("   Found tables:")
            for table in tables:
                print(f"   ✅ {table['table_name']}")
        else:
            print("   ⚠️  No expected tables found")
        
        db.disconnect()
        print("\n✅ Test completed successfully!")
        
    else:
        print("❌ Connection failed!")
        print("\n🔧 Check your connection details:")
        print("   - Database name")
        print("   - Username") 
        print("   - Password")
        print("   - PostgreSQL is running")