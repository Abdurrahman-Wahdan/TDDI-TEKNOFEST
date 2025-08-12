"""
Registration Service for MCP Client
Step 2e: New customer registration and account creation
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, date
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import db


logger = logging.getLogger(__name__)


class RegistrationService:
    """
    Service for registering new customers
    """
    
    def __init__(self):
        """Initialize registration service"""
        self.db = db
        logger.info("Registration service initialized")
    
    def check_tc_kimlik_exists(self, tc_kimlik_no: str) -> bool:
        """
        Check if TC kimlik number already exists in the system.
        
        Args:
            tc_kimlik_no: Turkish ID number
            
        Returns:
            bool: True if TC kimlik already exists
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return False
            
            query = """
            SELECT COUNT(*) as count
            FROM customers
            WHERE tc_kimlik_no = %s
            """
            
            result = self.db.execute_single(query, (tc_kimlik_no,))
            
            if result and result['count'] > 0:
                logger.info(f"TC kimlik {tc_kimlik_no} already exists")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking TC kimlik existence: {e}")
            return False
    
    def create_new_customer(
        self, 
        tc_kimlik_no: str, 
        first_name: str,
        last_name: str,
        phone_number: str,
        email: str,
        city: str,
        district: str = "",
        initial_plan_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a new customer account.
        
        Args:
            tc_kimlik_no: Turkish ID number
            first_name: Customer's first name
            last_name: Customer's last name
            phone_number: Phone number
            email: Email address
            city: City
            district: District (optional)
            initial_plan_id: Optional plan to assign immediately
            
        Returns:
            Dict with creation result
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return {"success": False, "message": "Database connection failed"}
            
            # Check if TC kimlik already exists
            if self.check_tc_kimlik_exists(tc_kimlik_no):
                return {
                    "success": False,
                    "message": "Customer with this TC kimlik number already exists"
                }
            
            # Validate initial plan if provided
            initial_plan = None
            if initial_plan_id:
                plan_query = """
                SELECT plan_id, plan_name, plan_type, monthly_fee, quota_gb
                FROM plans
                WHERE plan_id = %s
                """
                
                initial_plan = self.db.execute_single(plan_query, (initial_plan_id,))
                
                if not initial_plan:
                    return {
                        "success": False,
                        "message": f"Initial plan with ID {initial_plan_id} not found"
                    }
            
            # Begin transaction
            with self.db.connection.cursor() as cursor:
                # 1. Create customer record
                customer_insert = """
                INSERT INTO customers (
                    tc_kimlik_no, first_name, last_name, phone_number, 
                    email, city, district, customer_since, customer_status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'active')
                RETURNING customer_id
                """
                
                cursor.execute(customer_insert, (
                    tc_kimlik_no, first_name, last_name, phone_number,
                    email, city, district, date.today()
                ))
                
                customer_result = cursor.fetchone()
                customer_id = customer_result['customer_id']
                
                # 2. Assign initial plan if provided
                if initial_plan_id:
                    plan_insert = """
                    INSERT INTO customer_plans (customer_id, plan_id, is_active)
                    VALUES (%s, %s, true)
                    """
                    
                    cursor.execute(plan_insert, (customer_id, initial_plan_id))
                
                self.db.connection.commit()
            
            logger.info(f"New customer created: ID {customer_id}, TC: {tc_kimlik_no}")
            
            # Prepare response
            result = {
                "success": True,
                "message": "Customer account created successfully",
                "customer_id": customer_id,
                "customer_data": {
                    "customer_id": customer_id,
                    "tc_kimlik_no": tc_kimlik_no,
                    "first_name": first_name,
                    "last_name": last_name,
                    "phone_number": phone_number,
                    "email": email,
                    "city": city,
                    "district": district,
                    "customer_since": date.today().isoformat(),
                    "customer_status": "active"
                }
            }
            
            if initial_plan:
                result["initial_plan"] = {
                    "plan_id": initial_plan_id,
                    "plan_name": initial_plan['plan_name'],
                    "plan_type": initial_plan['plan_type'],
                    "monthly_fee": initial_plan['monthly_fee'],
                    "quota_gb": initial_plan['quota_gb']
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating new customer: {e}")
            if self.db.connection:
                self.db.connection.rollback()
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def get_registration_stats(self) -> Dict[str, Any]:
        """
        Get registration statistics.
        
        Returns:
            Dict with registration stats
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return {"error": "Database connection failed"}
            
            stats_query = """
            SELECT 
                COUNT(*) as total_customers,
                COUNT(CASE WHEN customer_status = 'active' THEN 1 END) as active_customers,
                COUNT(CASE WHEN customer_since >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as new_this_month,
                COUNT(CASE WHEN customer_since >= CURRENT_DATE - INTERVAL '7 days' THEN 1 END) as new_this_week
            FROM customers
            """
            
            stats = self.db.execute_single(stats_query)
            
            if stats:
                logger.info("Registration statistics retrieved")
                return stats
            else:
                return {"error": "Failed to retrieve statistics"}
                
        except Exception as e:
            logger.error(f"Error getting registration stats: {e}")
            return {"error": f"Error: {str(e)}"}


# Global registration service instance
registration_service = RegistrationService()


def create_new_customer(
    tc_kimlik_no: str, 
    first_name: str,
    last_name: str,
    phone_number: str,
    email: str,
    city: str,
    district: str = "",
    initial_plan_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Simple function to create new customer.
    
    Args:
        tc_kimlik_no: Turkish ID number
        first_name: First name
        last_name: Last name
        phone_number: Phone number
        email: Email address
        city: City
        district: District (optional)
        initial_plan_id: Optional initial plan
        
    Returns:
        Dict with creation result
    """
    return registration_service.create_new_customer(
        tc_kimlik_no, first_name, last_name, phone_number,
        email, city, district, initial_plan_id
    )


def check_tc_kimlik_exists(tc_kimlik_no: str) -> bool:
    """Simple function to check TC kimlik existence"""
    return registration_service.check_tc_kimlik_exists(tc_kimlik_no)


if __name__ == "__main__":
    # Test registration service
    logging.basicConfig(level=logging.INFO)
    
    print("📝 Testing Registration Service")
    print("=" * 40)
    
    # Test 1: Get registration stats
    print("1️⃣ Testing get_registration_stats()")
    stats = registration_service.get_registration_stats()
    
    if "error" not in stats:
        print(f"   📊 Registration Statistics:")
        print(f"      Total Customers: {stats['total_customers']}")
        print(f"      Active Customers: {stats['active_customers']}")
        print(f"      New This Month: {stats['new_this_month']}")
        print(f"      New This Week: {stats['new_this_week']}")
    else:
        print(f"   ❌ Error: {stats['error']}")
    
    # Test 2: Check if TC kimlik exists (test with existing one)
    print(f"\n2️⃣ Testing check_tc_kimlik_exists()")
    test_tc = input("Enter a TC kimlik to check (or press Enter to skip): ").strip()
    
    if test_tc:
        exists = check_tc_kimlik_exists(test_tc)
        print(f"   TC kimlik {test_tc} exists: {exists}")
    else:
        print("   ⏭️ Skipped TC kimlik check")
    
    # Test 3: Create new customer (interactive)
    print(f"\n3️⃣ Testing create_new_customer()")
    
    create_test = input("Do you want to test customer creation? (y/N): ").strip().lower()
    
    if create_test == 'y':
        print("Enter new customer details:")
        
        new_tc = input("TC Kimlik No: ").strip()
        first_name = input("First Name: ").strip()
        last_name = input("Last Name: ").strip()
        phone = input("Phone Number: ").strip()
        email = input("Email: ").strip()
        city = input("City: ").strip()
        district = input("District (optional): ").strip()
        
        if new_tc and first_name and last_name and phone and email and city:
            print(f"\n   Creating customer: {first_name} {last_name}")
            
            result = create_new_customer(
                new_tc, first_name, last_name, phone, email, city, district
            )
            
            if result["success"]:
                print(f"   ✅ Customer created successfully!")
                print(f"      Customer ID: {result['customer_id']}")
                print(f"      Name: {result['customer_data']['first_name']} {result['customer_data']['last_name']}")
                print(f"      TC: {result['customer_data']['tc_kimlik_no']}")
                print(f"      Phone: {result['customer_data']['phone_number']}")
            else:
                print(f"   ❌ Creation failed: {result['message']}")
        else:
            print("   ⚠️ Missing required fields")
    else:
        print("   ⏭️ Skipped customer creation test")
    
    print("\n✅ Registration service test completed!")