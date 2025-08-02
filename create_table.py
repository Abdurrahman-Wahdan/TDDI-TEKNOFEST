"""
One-time setup script to create bill_disputes table
Run this once to add the disputes functionality to your database.
"""

import logging
from database import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_disputes_table():
    """
    Create the bill_disputes table for recording customer disputes.
    """
    
    # SQL to create the disputes table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS bill_disputes (
        dispute_id SERIAL PRIMARY KEY,
        customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
        bill_id INTEGER NOT NULL REFERENCES billing(bill_id),
        reason TEXT NOT NULL,
        status VARCHAR(20) DEFAULT 'submitted',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved_at TIMESTAMP NULL,
        resolution_notes TEXT NULL,
        resolved_by VARCHAR(100) NULL,
        
        -- Indexes for better performance
        CONSTRAINT unique_customer_bill_dispute UNIQUE(customer_id, bill_id),
        CONSTRAINT valid_status CHECK (status IN ('submitted', 'investigating', 'resolved', 'rejected'))
    );
    
    -- Add indexes
    CREATE INDEX IF NOT EXISTS idx_disputes_customer_id ON bill_disputes(customer_id);
    CREATE INDEX IF NOT EXISTS idx_disputes_status ON bill_disputes(status);
    CREATE INDEX IF NOT EXISTS idx_disputes_created_at ON bill_disputes(created_at);
    """
    
    try:
        # Connect to database
        if not db.is_connected():
            success = db.connect()
            if not success:
                logger.error("Failed to connect to database")
                return False
        
        # Execute the table creation
        with db.connection.cursor() as cursor:
            cursor.execute(create_table_sql)
            db.connection.commit()
        
        logger.info("✅ bill_disputes table created successfully")
        
        # Verify table was created
        verify_sql = """
        SELECT column_name, data_type, is_nullable 
        FROM information_schema.columns 
        WHERE table_name = 'bill_disputes' 
        ORDER BY ordinal_position;
        """
        
        columns = db.execute_query(verify_sql)
        
        if columns:
            logger.info(f"✅ Table verification: {len(columns)} columns created")
            for col in columns:
                logger.info(f"   - {col['column_name']} ({col['data_type']}) - Nullable: {col['is_nullable']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating disputes table: {e}")
        if db.connection:
            db.connection.rollback()
        return False


def check_table_exists():
    """Check if the disputes table already exists"""
    
    try:
        if not db.is_connected():
            db.connect()
        
        check_sql = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'bill_disputes'
        );
        """
        
        result = db.execute_single(check_sql)
        return result['exists'] if result else False
        
    except Exception as e:
        logger.error(f"Error checking table existence: {e}")
        return False


def main():
    """Main setup function"""
    
    print("🔧 Bill Disputes Table Setup")
    print("=" * 40)
    
    # Check if table already exists
    print("🔍 Checking if bill_disputes table exists...")
    
    if check_table_exists():
        print("⚠️  bill_disputes table already exists!")
        
        choice = input("Do you want to recreate it? (y/N): ").strip().lower()
        if choice == 'y':
            print("🗑️  Dropping existing table...")
            try:
                with db.connection.cursor() as cursor:
                    cursor.execute("DROP TABLE IF EXISTS bill_disputes CASCADE;")
                    db.connection.commit()
                print("✅ Existing table dropped")
            except Exception as e:
                print(f"❌ Error dropping table: {e}")
                return
        else:
            print("👍 Keeping existing table. Setup completed.")
            return
    
    # Create the table
    print("📝 Creating bill_disputes table...")
    
    success = create_disputes_table()
    
    if success:
        print("✅ Setup completed successfully!")
        print("\n📋 What was created:")
        print("   • bill_disputes table")
        print("   • Primary key: dispute_id (auto-increment)")
        print("   • Foreign keys: customer_id, bill_id")
        print("   • Status constraint: submitted/investigating/resolved/rejected")
        print("   • Unique constraint: one dispute per customer+bill")
        print("   • Indexes for performance")
        
        print("\n🎯 Next steps:")
        print("   1. Table is ready for use")
        print("   2. Update billing_service.py to use real database writes")
        print("   3. Test dispute creation functionality")
        
    else:
        print("❌ Setup failed. Check the error messages above.")
    
    # Cleanup
    db.disconnect()


if __name__ == "__main__":
    main()