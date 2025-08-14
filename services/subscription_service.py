# Copyright 2025 kermits
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Subscription Service for MCP Client
Step 2a: Customer subscription and plan management
"""

import logging
from typing import Dict, Any, Optional, List
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import db


logger = logging.getLogger(__name__)


class SubscriptionService:
    """
    Service for managing customer subscriptions and plans
    """
    
    def __init__(self):
        """Initialize subscription service"""
        self.db = db
        logger.info("Subscription service initialized")
    
    def get_customer_active_plans(self, customer_id: int) -> List[Dict[str, Any]]:
        """
        Get customer's active plans only.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            List of active plan dictionaries
        """
        try:
            # Ensure database connection
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    logger.error("Database connection failed")
                    return []
            
            # Query active plans
            query = """
            SELECT 
                p.plan_id,
                p.plan_type,
                p.plan_name,
                p.monthly_fee,
                p.quota_gb,
                p.contract_end_date,
                cp.is_active
            FROM customer_plans cp
            JOIN plans p ON cp.plan_id = p.plan_id  
            WHERE cp.customer_id = %s AND cp.is_active = true
            ORDER BY p.plan_name
            """
            
            active_plans = self.db.execute_query(query, (customer_id,))
            
            logger.info(f"Found {len(active_plans)} active plans for customer {customer_id}")
            return active_plans
            
        except Exception as e:
            logger.error(f"Error getting active plans: {e}")
            return []
    
    def get_customer_subscription_info(self, customer_id: int) -> Dict[str, Any]:
        """
        Get comprehensive customer subscription information.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Dict with customer info, all plans, and billing summary
        """
        try:
            # Ensure database connection
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    logger.error("Database connection failed")
                    return {"error": "Database connection failed"}
            
            # 1. Get customer basic info
            customer_query = """
            SELECT 
                customer_id,
                first_name,
                last_name,
                phone_number,
                email,
                city,
                customer_since,
                customer_status
            FROM customers 
            WHERE customer_id = %s
            """
            
            customer_info = self.db.execute_single(customer_query, (customer_id,))
            
            if not customer_info:
                return {"error": "Customer not found"}
            
            # 2. Get all customer plans (active and inactive)
            plans_query = """
            SELECT 
                p.plan_id,
                p.plan_type,
                p.plan_name,
                p.monthly_fee,
                p.quota_gb,
                p.contract_end_date,
                cp.is_active
            FROM customer_plans cp
            JOIN plans p ON cp.plan_id = p.plan_id
            WHERE cp.customer_id = %s
            ORDER BY cp.is_active DESC, p.plan_name
            """
            
            all_plans = self.db.execute_query(plans_query, (customer_id,))
            
            # 3. Get billing summary
            billing_query = """
            SELECT 
                COUNT(*) as total_bills,
                COUNT(CASE WHEN status = 'paid' THEN 1 END) as paid_bills,
                COUNT(CASE WHEN status = 'unpaid' THEN 1 END) as unpaid_bills,
                SUM(amount) as total_amount,
                SUM(CASE WHEN status = 'unpaid' THEN amount ELSE 0 END) as outstanding_amount,
                MAX(due_date) as latest_due_date
            FROM billing 
            WHERE customer_id = %s
            """
            
            billing_summary = self.db.execute_single(billing_query, (customer_id,))
            
            # 4. Separate active and inactive plans
            active_plans = [plan for plan in all_plans if plan['is_active']]
            inactive_plans = [plan for plan in all_plans if not plan['is_active']]
            
            # 5. Build comprehensive response
            subscription_info = {
                "customer_info": customer_info,
                "active_plans": active_plans,
                "inactive_plans": inactive_plans,
                "plans_summary": {
                    "total_plans": len(all_plans),
                    "active_count": len(active_plans),
                    "inactive_count": len(inactive_plans)
                },
                "billing_summary": billing_summary,
                "success": True
            }
            
            logger.info(f"Retrieved subscription info for customer {customer_id}: {len(active_plans)} active plans")
            return subscription_info
            
        except Exception as e:
            logger.error(f"Error getting subscription info: {e}")
            return {"error": f"Failed to retrieve subscription info: {str(e)}", "success": False}
    
    def change_customer_plan(self, customer_id: int, old_plan_id: int, new_plan_id: int) -> Dict[str, Any]:
        """
        Change customer's plan (deactivate old, activate new).
        
        Args:
            customer_id: Customer ID
            old_plan_id: Current plan ID to deactivate
            new_plan_id: New plan ID to activate
            
        Returns:
            Dict with change result
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return {"success": False, "message": "Database connection failed"}
            
            # Verify customer has the old plan active
            verify_query = """
            SELECT COUNT(*) as count
            FROM customer_plans
            WHERE customer_id = %s AND plan_id = %s AND is_active = true
            """
            
            has_old_plan = self.db.execute_single(verify_query, (customer_id, old_plan_id))
            
            if not has_old_plan or has_old_plan['count'] == 0:
                return {
                    "success": False,
                    "message": "Customer does not have the specified active plan"
                }
            
            # Check if new plan exists
            new_plan_query = """
            SELECT plan_id, plan_name, plan_type, monthly_fee, quota_gb
            FROM plans
            WHERE plan_id = %s
            """
            
            new_plan = self.db.execute_single(new_plan_query, (new_plan_id,))
            
            if not new_plan:
                return {
                    "success": False,
                    "message": "New plan not found"
                }
            
            # Check if customer already has the new plan
            has_new_plan_query = """
            SELECT COUNT(*) as count
            FROM customer_plans
            WHERE customer_id = %s AND plan_id = %s
            """
            
            existing_new_plan = self.db.execute_single(has_new_plan_query, (customer_id, new_plan_id))
            
            # Begin transaction
            with self.db.connection.cursor() as cursor:
                # 1. Deactivate old plan
                cursor.execute("""
                    UPDATE customer_plans 
                    SET is_active = false 
                    WHERE customer_id = %s AND plan_id = %s
                """, (customer_id, old_plan_id))
                
                # 2. Activate new plan (insert or update)
                if existing_new_plan and existing_new_plan['count'] > 0:
                    # Update existing record
                    cursor.execute("""
                        UPDATE customer_plans 
                        SET is_active = true 
                        WHERE customer_id = %s AND plan_id = %s
                    """, (customer_id, new_plan_id))
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO customer_plans (customer_id, plan_id, is_active)
                        VALUES (%s, %s, true)
                    """, (customer_id, new_plan_id))
                
                self.db.connection.commit()
            
            logger.info(f"Plan changed for customer {customer_id}: {old_plan_id} → {new_plan_id}")
            
            return {
                "success": True,
                "message": "Plan changed successfully",
                "old_plan_id": old_plan_id,
                "new_plan_id": new_plan_id,
                "new_plan_details": new_plan
            }
            
        except Exception as e:
            logger.error(f"Error changing customer plan: {e}")
            if self.db.connection:
                self.db.connection.rollback()
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def get_all_available_plans(self) -> List[Dict[str, Any]]:
        """
        Get all available plans for plan changes.
        
        Returns:
            List of all available plans
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return []
            
            query = """
            SELECT 
                plan_id,
                plan_type,
                plan_name,
                monthly_fee,
                quota_gb,
                contract_end_date
            FROM plans
            ORDER BY plan_type, monthly_fee
            """
            
            plans = self.db.execute_query(query)
            logger.info(f"Retrieved {len(plans)} available plans")
            return plans
            
        except Exception as e:
            logger.error(f"Error getting available plans: {e}")
            return []
    
    def get_available_plans(self) -> List[Dict[str, Any]]:
        """
        Get all available plans for plan changes (alias for get_all_available_plans).
        
        Returns:
            List of all available plans
        """
        return self.get_all_available_plans()


# Global subscription service instance
subscription_service = SubscriptionService()


def get_customer_active_plans(customer_id: int) -> List[Dict[str, Any]]:
    """
    Simple function to get customer's active plans.
    
    Args:
        customer_id: Customer ID
        
    Returns:
        List of active plans
    """
    return subscription_service.get_customer_active_plans(customer_id)


def get_customer_subscription_info(customer_id: int) -> Dict[str, Any]:
    """
    Simple function to get comprehensive subscription info.
    
    Args:
        customer_id: Customer ID
        
    Returns:
        Dict with comprehensive subscription information
    """
    return subscription_service.get_customer_subscription_info(customer_id)


def get_available_plans() -> List[Dict[str, Any]]:
    """
    Simple function to get all available plans.
    
    Returns:
        List of available plans
    """
    return subscription_service.get_available_plans()


def change_customer_plan(customer_id: int, old_plan_id: int, new_plan_id: int) -> Dict[str, Any]:
    """
    Simple function to change customer plan.
    
    Args:
        customer_id: Customer ID
        old_plan_id: Current plan ID
        new_plan_id: New plan ID
        
    Returns:
        Dict with change result
    """
    return subscription_service.change_customer_plan(customer_id, old_plan_id, new_plan_id)


if __name__ == "__main__":
    # Test subscription service
    logging.basicConfig(level=logging.INFO)
    
    print("📋 Testing Subscription Service")
    print("=" * 40)
    
    # Get test customer ID
    test_customer_id = input("Enter a test customer ID from your database: ").strip()
    
    try:
        test_customer_id = int(test_customer_id)
    except ValueError:
        print("❌ Invalid customer ID")
        exit(1)
    
    print(f"\n🧪 Testing subscription functions for customer ID: {test_customer_id}")
    print("-" * 50)
    
    # Test 1: Get active plans
    print("1️⃣ Testing get_customer_active_plans()")
    active_plans = get_customer_active_plans(test_customer_id)
    
    if active_plans:
        print(f"   ✅ Found {len(active_plans)} active plans:")
        for plan in active_plans:
            print(f"      • {plan['plan_name']} ({plan['plan_type']}) - {plan['monthly_fee']}₺/month - {plan['quota_gb']}GB")
    else:
        print("   ⚠️ No active plans found")
    
    # Test 2: Get comprehensive subscription info
    print(f"\n2️⃣ Testing get_customer_subscription_info()")
    subscription_info = get_customer_subscription_info(test_customer_id)
    
    if subscription_info.get("success"):
        customer = subscription_info["customer_info"]
        print(f"   ✅ Customer: {customer['first_name']} {customer['last_name']}")
        print(f"      Phone: {customer['phone_number']}")
        print(f"      Status: {customer['customer_status']}")
        print(f"      Customer Since: {customer['customer_since']}")
        
        print(f"\n   📊 Plans Summary:")
        summary = subscription_info["plans_summary"]
        print(f"      Total Plans: {summary['total_plans']}")
        print(f"      Active: {summary['active_count']}")
        print(f"      Inactive: {summary['inactive_count']}")
        
        print(f"\n   💰 Billing Summary:")
        billing = subscription_info["billing_summary"]
        print(f"      Total Bills: {billing['total_bills']}")
        print(f"      Paid: {billing['paid_bills']}")
        print(f"      Unpaid: {billing['unpaid_bills']}")
        print(f"      Outstanding Amount: {billing['outstanding_amount']}₺")
        
    else:
        print(f"   ❌ Error: {subscription_info.get('error')}")
    
    # Test 3: Get all available plans
    print(f"\n3️⃣ Testing get_available_plans()")
    all_plans = get_available_plans()
    
    if all_plans:
        print(f"   ✅ Found {len(all_plans)} available plans:")
        for plan in all_plans[:3]:  # Show first 3
            print(f"      • {plan['plan_name']} - {plan['monthly_fee']}₺ - {plan['quota_gb']}GB")
        if len(all_plans) > 3:
            print(f"      ... and {len(all_plans) - 3} more")
    else:
        print("   ⚠️ No plans found")
    
    # Test 4: Test plan change (if customer has active plans and other plans available)
    if active_plans and all_plans and len(all_plans) > 1:
        print(f"\n4️⃣ Testing change_customer_plan()")
        
        current_plan = active_plans[0]
        
        # Find a different plan to change to
        new_plan = None
        for plan in all_plans:
            if plan['plan_id'] != current_plan['plan_id']:
                new_plan = plan
                break
        
        if new_plan:
            print(f"   🔄 Changing from '{current_plan['plan_name']}' to '{new_plan['plan_name']}'")
            
            change_result = change_customer_plan(
                test_customer_id,
                current_plan['plan_id'],
                new_plan['plan_id']
            )
            
            if change_result["success"]:
                print(f"   ✅ Plan changed successfully!")
                print(f"      Old: {current_plan['plan_name']}")
                print(f"      New: {change_result['new_plan_details']['plan_name']}")
            else:
                print(f"   ❌ Plan change failed: {change_result['message']}")
        else:
            print("   ⚠️ No alternative plans available for testing")
    else:
        print(f"\n4️⃣ Skipping plan change test (insufficient data)")
    
    print("\n✅ Subscription service test completed!")