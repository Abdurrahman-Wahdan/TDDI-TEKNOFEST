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
Authentication Service for MCP Client
Step 1b: Customer authentication by TC kimlik number
"""

import logging
from typing import Dict, Any, Optional

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import db

logger = logging.getLogger(__name__)


class AuthService:
    """
    Authentication service for customer verification
    """
    
    def __init__(self):
        """Initialize authentication service"""
        self.db = db
        logger.info("Authentication service initialized")
    
    def authenticate_customer(self, tc_kimlik_no: str) -> Dict[str, Any]:
        """
        Authenticate customer by TC kimlik number.
        
        Args:
            tc_kimlik_no: Turkish ID number
            
        Returns:
            Dict: {
                "exists": bool,
                "is_active": bool, 
                "customer_id": int or None,
                "customer_data": dict or None,
                "message": str
            }
        """
        try:
            # Ensure database connection
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return {
                        "exists": False,
                        "is_active": False,
                        "customer_id": None,
                        "customer_data": None,
                        "message": "Database connection failed"
                    }
            
            # Query customer by TC kimlik
            query = """
            SELECT 
                customer_id,
                tc_kimlik_no,
                first_name,
                last_name,
                phone_number,
                email,
                city,
                district,
                customer_since,
                customer_status
            FROM customers 
            WHERE tc_kimlik_no = %s
            """
            
            customer = self.db.execute_single(query, (tc_kimlik_no,))
            
            if not customer:
                logger.info(f"Customer not found for TC: {tc_kimlik_no}")
                return {
                    "exists": False,
                    "is_active": False,
                    "customer_id": None,
                    "customer_data": None,
                    "message": "Customer not found"
                }
            
            # Check if customer is active using customer_status
            is_active = customer['customer_status'].lower() == 'active'
            
            if not is_active:
                logger.info(f"Customer {customer['customer_id']} is not active: {customer['customer_status']}")
                return {
                    "exists": True,
                    "is_active": False,
                    "customer_id": customer['customer_id'],
                    "customer_data": customer,
                    "message": f"Customer account is {customer['customer_status']}"
                }
            
            # Customer exists and is active
            logger.info(f"Customer {customer['customer_id']} authenticated successfully")
            return {
                "exists": True,
                "is_active": True,
                "customer_id": customer['customer_id'],
                "customer_data": customer,
                "message": "Authentication successful"
            }
            
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return {
                "exists": False,
                "is_active": False,
                "customer_id": None,
                "customer_data": None,
                "message": f"Authentication error: {str(e)}"
            }
    
    def get_customer_summary(self, customer_id: int) -> Optional[Dict[str, Any]]:
        """
        Get customer summary information.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Dict with customer summary or None
        """
        try:
            query = """
            SELECT 
                c.customer_id,
                c.first_name,
                c.last_name,
                c.phone_number,
                c.customer_since,
                c.customer_status,
                COUNT(cp.plan_id) as active_plans
            FROM customers c
            LEFT JOIN customer_plans cp ON c.customer_id = cp.customer_id AND cp.is_active = true
            WHERE c.customer_id = %s
            GROUP BY c.customer_id, c.first_name, c.last_name, c.phone_number, c.customer_since, c.customer_status
            """
            
            return self.db.execute_single(query, (customer_id,))
            
        except Exception as e:
            logger.error(f"Error getting customer summary: {e}")
            return None


# Global auth service instance
auth_service = AuthService()


def authenticate_customer(tc_kimlik_no: str) -> Dict[str, Any]:
    """
    Simple function to authenticate customer.
    
    Args:
        tc_kimlik_no: Turkish ID number
        
    Returns:
        Authentication result dictionary
    """
    return auth_service.authenticate_customer(tc_kimlik_no)


if __name__ == "__main__":
    # Test authentication service
    logging.basicConfig(level=logging.INFO)
    
    print("🔐 Testing Authentication Service")
    print("=" * 40)
    
    # You need to provide a sample TC kimlik from your test data
    test_tc_kimlik = input("Enter a test TC kimlik number from your database: ").strip()
    
    if not test_tc_kimlik:
        print("❌ No TC kimlik provided")
        exit(1)
    
    print(f"\n🧪 Testing authentication for: {test_tc_kimlik}")
    print("-" * 40)
    
    # Test authentication
    result = authenticate_customer(test_tc_kimlik)
    
    print(f"📊 Authentication Result:")
    print(f"   Exists: {result['exists']}")
    print(f"   Active: {result['is_active']}")
    print(f"   Customer ID: {result['customer_id']}")
    print(f"   Message: {result['message']}")
    
    if result['customer_data']:
        customer = result['customer_data']
        print(f"\n👤 Customer Details:")
        print(f"   Name: {customer['first_name']} {customer['last_name']}")
        print(f"   Phone: {customer['phone_number']}")
        print(f"   Email: {customer['email']}")
        print(f"   City: {customer['city']}")
        print(f"   Status: {customer['customer_status']}")
        print(f"   Customer Since: {customer['customer_since']}")
    
    # Test customer summary
    if result['customer_id']:
        print(f"\n📋 Testing customer summary...")
        summary = auth_service.get_customer_summary(result['customer_id'])
        if summary:
            print(f"   Active Plans: {summary['active_plans']}")
    
    print("\n✅ Authentication test completed!")