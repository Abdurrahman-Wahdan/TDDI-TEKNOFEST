"""
Billing Service for MCP Client
Step 2b: Customer billing and payment management
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, date
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import db


logger = logging.getLogger(__name__)


class BillingService:
    """
    Service for managing customer billing and payments
    """
    
    def __init__(self):
        """Initialize billing service"""
        self.db = db
        logger.info("Billing service initialized")
    
    def get_customer_bills(self, customer_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get customer's bills (recent first).
        
        Args:
            customer_id: Customer ID
            limit: Maximum number of bills to return
            
        Returns:
            List of bill dictionaries
        """
        try:
            # Ensure database connection
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    logger.error("Database connection failed")
                    return []
            
            # Query customer bills
            query = """
            SELECT 
                bill_id,
                customer_id,
                amount,
                due_date,
                status,
                last_payment_date
            FROM billing
            WHERE customer_id = %s
            ORDER BY due_date DESC
            LIMIT %s
            """
            
            bills = self.db.execute_query(query, (customer_id, limit))
            
            logger.info(f"Found {len(bills)} bills for customer {customer_id}")
            return bills
            
        except Exception as e:
            logger.error(f"Error getting customer bills: {e}")
            return []
    
    def get_bill_details(self, customer_id: int, bill_id: int) -> Optional[Dict[str, Any]]:
        """
        Get specific bill details for a customer.
        
        Args:
            customer_id: Customer ID (for security)
            bill_id: Bill ID
            
        Returns:
            Bill details dictionary or None
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return None
            
            # Query specific bill (with customer_id for security)
            query = """
            SELECT 
                bill_id,
                customer_id,
                amount,
                due_date,
                status,
                last_payment_date
            FROM billing
            WHERE customer_id = %s AND bill_id = %s
            """
            
            bill = self.db.execute_single(query, (customer_id, bill_id))
            
            if bill:
                logger.info(f"Retrieved bill {bill_id} for customer {customer_id}")
            else:
                logger.warning(f"Bill {bill_id} not found for customer {customer_id}")
            
            return bill
            
        except Exception as e:
            logger.error(f"Error getting bill details: {e}")
            return None
    
    def get_unpaid_bills(self, customer_id: int) -> List[Dict[str, Any]]:
        """
        Get customer's unpaid bills only.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            List of unpaid bills
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return []
            
            query = """
            SELECT 
                bill_id,
                customer_id,
                amount,
                due_date,
                status,
                last_payment_date
            FROM billing
            WHERE customer_id = %s AND status = 'unpaid'
            ORDER BY due_date ASC
            """
            
            unpaid_bills = self.db.execute_query(query, (customer_id,))
            
            logger.info(f"Found {len(unpaid_bills)} unpaid bills for customer {customer_id}")
            return unpaid_bills
            
        except Exception as e:
            logger.error(f"Error getting unpaid bills: {e}")
            return []
    
    def get_overdue_bills(self, customer_id: int) -> List[Dict[str, Any]]:
        """
        Get customer's overdue bills (unpaid + past due date).
        
        Args:
            customer_id: Customer ID
            
        Returns:
            List of overdue bills
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return []
            
            today = date.today()
            
            query = """
            SELECT 
                bill_id,
                customer_id,
                amount,
                due_date,
                status,
                last_payment_date
            FROM billing
            WHERE customer_id = %s 
                AND status = 'unpaid' 
                AND due_date < %s
            ORDER BY due_date ASC
            """
            
            overdue_bills = self.db.execute_query(query, (customer_id, today))
            
            logger.info(f"Found {len(overdue_bills)} overdue bills for customer {customer_id}")
            return overdue_bills
            
        except Exception as e:
            logger.error(f"Error getting overdue bills: {e}")
            return []
    
    def create_bill_dispute(self, customer_id: int, bill_id: int, reason: str) -> Dict[str, Any]:
        """
        Create a bill dispute and store it in the database.
        
        Args:
            customer_id: Customer ID
            bill_id: Bill ID to dispute
            reason: Dispute reason
            
        Returns:
            Dict with dispute result
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return {"success": False, "message": "Database connection failed"}
            
            # First verify the bill exists and belongs to customer
            bill = self.get_bill_details(customer_id, bill_id)
            
            if not bill:
                return {
                    "success": False, 
                    "message": "Bill not found or does not belong to customer"
                }
            
            # Check if dispute already exists for this customer+bill
            existing_dispute_query = """
            SELECT dispute_id, status 
            FROM bill_disputes 
            WHERE customer_id = %s AND bill_id = %s
            """
            
            existing = self.db.execute_single(existing_dispute_query, (customer_id, bill_id))
            
            if existing:
                return {
                    "success": False,
                    "message": f"Dispute already exists for this bill (ID: {existing['dispute_id']}, Status: {existing['status']})"
                }
            
            # Insert new dispute into database
            insert_query = """
            INSERT INTO bill_disputes (customer_id, bill_id, reason, status, created_at)
            VALUES (%s, %s, %s, 'submitted', CURRENT_TIMESTAMP)
            RETURNING dispute_id, created_at
            """
            
            with self.db.connection.cursor() as cursor:
                cursor.execute(insert_query, (customer_id, bill_id, reason))
                result = cursor.fetchone()
                self.db.connection.commit()
                
                dispute_id = result['dispute_id']
                created_at = result['created_at']
            
            logger.info(f"Bill dispute created in database: ID {dispute_id} for customer {customer_id}, bill {bill_id}")
            
            return {
                "success": True,
                "message": "Bill dispute submitted successfully",
                "dispute_id": dispute_id,
                "dispute_info": {
                    "dispute_id": dispute_id,
                    "customer_id": customer_id,
                    "bill_id": bill_id,
                    "bill_amount": bill["amount"],
                    "reason": reason,
                    "status": "submitted",
                    "created_at": created_at.isoformat() if created_at else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating bill dispute: {e}")
            if self.db.connection:
                self.db.connection.rollback()
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def get_customer_disputes(self, customer_id: int) -> List[Dict[str, Any]]:
        """
        Get all disputes for a customer.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            List of customer disputes
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return []
            
            query = """
            SELECT 
                d.dispute_id,
                d.customer_id,
                d.bill_id,
                d.reason,
                d.status,
                d.created_at,
                d.resolved_at,
                d.resolution_notes,
                b.amount as bill_amount,
                b.due_date as bill_due_date
            FROM bill_disputes d
            JOIN billing b ON d.bill_id = b.bill_id
            WHERE d.customer_id = %s
            ORDER BY d.created_at DESC
            """
            
            disputes = self.db.execute_query(query, (customer_id,))
            
            logger.info(f"Found {len(disputes)} disputes for customer {customer_id}")
            return disputes
            
        except Exception as e:
            logger.error(f"Error getting customer disputes: {e}")
            return []
    
    def get_billing_summary(self, customer_id: int) -> Dict[str, Any]:
        """
        Get comprehensive billing summary for customer.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Dict with billing statistics
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return {"error": "Database connection failed"}
            
            # Get comprehensive billing statistics
            query = """
            SELECT 
                COUNT(*) as total_bills,
                COUNT(CASE WHEN status = 'paid' THEN 1 END) as paid_bills,
                COUNT(CASE WHEN status = 'unpaid' THEN 1 END) as unpaid_bills,
                SUM(amount) as total_billed,
                SUM(CASE WHEN status = 'paid' THEN amount ELSE 0 END) as total_paid,
                SUM(CASE WHEN status = 'unpaid' THEN amount ELSE 0 END) as outstanding_amount,
                MIN(due_date) as earliest_due_date,
                MAX(due_date) as latest_due_date,
                MAX(last_payment_date) as last_payment_date
            FROM billing 
            WHERE customer_id = %s
            """
            
            summary = self.db.execute_single(query, (customer_id,))
            
            if summary:
                # Add calculated fields
                summary["payment_rate"] = (
                    (summary["paid_bills"] / summary["total_bills"] * 100) 
                    if summary["total_bills"] > 0 else 0
                )
                
                # Get overdue count
                overdue_bills = self.get_overdue_bills(customer_id)
                summary["overdue_bills"] = len(overdue_bills)
                summary["overdue_amount"] = sum(bill["amount"] for bill in overdue_bills)
            
            logger.info(f"Generated billing summary for customer {customer_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Error getting billing summary: {e}")
            return {"error": f"Failed to get billing summary: {str(e)}"}


# Global billing service instance
billing_service = BillingService()


def get_customer_bills(customer_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """Simple function to get customer bills"""
    return billing_service.get_customer_bills(customer_id, limit)


def get_bill_details(customer_id: int, bill_id: int) -> Optional[Dict[str, Any]]:
    """Simple function to get bill details"""
    return billing_service.get_bill_details(customer_id, bill_id)


def create_bill_dispute(customer_id: int, bill_id: int, reason: str) -> Dict[str, Any]:
    """Simple function to create bill dispute"""
    return billing_service.create_bill_dispute(customer_id, bill_id, reason)


def get_customer_disputes(customer_id: int) -> List[Dict[str, Any]]:
    """Simple function to get customer disputes"""
    return billing_service.get_customer_disputes(customer_id)


if __name__ == "__main__":
    # Test billing service
    logging.basicConfig(level=logging.INFO)
    
    print("💰 Testing Billing Service")
    print("=" * 40)
    
    # Get test customer ID
    test_customer_id = input("Enter a test customer ID from your database: ").strip()
    
    try:
        test_customer_id = int(test_customer_id)
    except ValueError:
        print("❌ Invalid customer ID")
        exit(1)
    
    print(f"\n🧪 Testing billing functions for customer ID: {test_customer_id}")
    print("-" * 50)
    
    # Test 1: Get customer bills
    print("1️⃣ Testing get_customer_bills()")
    bills = get_customer_bills(test_customer_id, 5)
    
    if bills:
        print(f"   ✅ Found {len(bills)} bills:")
        for bill in bills:
            status_icon = "✅" if bill['status'] == 'paid' else "⏳"
            print(f"      {status_icon} Bill #{bill['bill_id']} - {bill['amount']}₺ - Due: {bill['due_date']} - Status: {bill['status']}")
    else:
        print("   ⚠️ No bills found")
    
    # Test 2: Get unpaid bills
    print(f"\n2️⃣ Testing get_unpaid_bills()")
    unpaid = billing_service.get_unpaid_bills(test_customer_id)
    
    if unpaid:
        print(f"   ⏳ Found {len(unpaid)} unpaid bills:")
        for bill in unpaid:
            print(f"      • Bill #{bill['bill_id']} - {bill['amount']}₺ - Due: {bill['due_date']}")
    else:
        print("   ✅ No unpaid bills")
    
    # Test 3: Get overdue bills
    print(f"\n3️⃣ Testing get_overdue_bills()")
    overdue = billing_service.get_overdue_bills(test_customer_id)
    
    if overdue:
        print(f"   ⚠️ Found {len(overdue)} overdue bills:")
        for bill in overdue:
            print(f"      • Bill #{bill['bill_id']} - {bill['amount']}₺ - Due: {bill['due_date']}")
    else:
        print("   ✅ No overdue bills")
    
    # Test 4: Get billing summary
    print(f"\n4️⃣ Testing get_billing_summary()")
    summary = billing_service.get_billing_summary(test_customer_id)
    
    if summary and "error" not in summary:
        print(f"   📊 Billing Summary:")
        print(f"      Total Bills: {summary['total_bills']}")
        print(f"      Paid: {summary['paid_bills']} | Unpaid: {summary['unpaid_bills']}")
        print(f"      Total Billed: {summary['total_billed']}₺")
        print(f"      Outstanding: {summary['outstanding_amount']}₺")
        print(f"      Payment Rate: {summary['payment_rate']:.1f}%")
        print(f"      Overdue: {summary['overdue_bills']} bills ({summary['overdue_amount']}₺)")
    else:
        print(f"   ❌ Error: {summary.get('error', 'Unknown error')}")
    
    # Test 5: Test bill dispute (if there are bills)
    if bills:
        print(f"\n5️⃣ Testing create_bill_dispute()")
        test_bill_id = bills[0]['bill_id']
        dispute_result = create_bill_dispute(
            test_customer_id, 
            test_bill_id, 
            "Test dispute - bill amount seems incorrect"
        )
        
        if dispute_result["success"]:
            print(f"   ✅ Dispute created: ID {dispute_result['dispute_id']}")
            print(f"      Status: {dispute_result['dispute_info']['status']}")
        else:
            print(f"   ❌ Dispute failed: {dispute_result['message']}")
    
    # Test 6: View customer disputes
    print(f"\n6️⃣ Testing get_customer_disputes()")
    disputes = get_customer_disputes(test_customer_id)
    
    if disputes:
        print(f"   📋 Found {len(disputes)} disputes:")
        for dispute in disputes:
            print(f"      • Dispute #{dispute['dispute_id']} - Bill #{dispute['bill_id']}")
            print(f"        Amount: {dispute['bill_amount']}₺ - Status: {dispute['status']}")
            print(f"        Reason: {dispute['reason'][:50]}...")
            print(f"        Created: {dispute['created_at']}")
    else:
        print("   ✅ No disputes found")
    
    print("\n✅ Billing service test completed!")