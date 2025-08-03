"""
Complete MCP Client for Turkcell Customer Service System

This client provides a unified interface to all backend services:
- Authentication
- Subscription Management  
- Billing Operations
- Technical Support
- Customer Registration

All operations return consistent JSON responses for easy agent integration.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import date, datetime

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all service modules
from services.auth_service import auth_service
from services.subscription_service import subscription_service
from services.billing_service import billing_service
from services.technical_service import technical_service
from services.registration_service import registration_service

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Unified MCP Client for all Turkcell customer service operations.
    
    Provides a single interface for agents to access all backend services
    with consistent error handling and response formats.
    """
    
    def __init__(self):
        """Initialize MCP client with all services"""
        self.auth_service = auth_service
        self.subscription_service = subscription_service
        self.billing_service = billing_service
        self.technical_service = technical_service
        self.registration_service = registration_service
        
        logger.info("MCP Client initialized with all services")
    
    # ==================== AUTHENTICATION OPERATIONS ====================
    
    def authenticate_customer(self, tc_kimlik_no: str) -> Dict[str, Any]:
        """
        Authenticate customer by TC kimlik number.
        
        Args:
            tc_kimlik_no: Turkish ID number
            
        Returns:
            Dict: {
                "success": bool,
                "exists": bool,
                "is_active": bool,
                "customer_id": int or None,
                "customer_data": dict or None,
                "message": str
            }
        """
        try:
            result = self.auth_service.authenticate_customer(tc_kimlik_no)
            
            return {
                "success": True,
                "exists": result["exists"],
                "is_active": result["is_active"],
                "customer_id": result["customer_id"],
                "customer_data": result["customer_data"],
                "message": result["message"]
            }
            
        except Exception as e:
            logger.error(f"MCP authentication error: {e}")
            return {
                "success": False,
                "exists": False,
                "is_active": False,
                "customer_id": None,
                "customer_data": None,
                "message": f"Authentication service error: {str(e)}"
            }
    
    # ==================== SUBSCRIPTION OPERATIONS ====================
    
    def get_customer_active_plans(self, customer_id: int) -> Dict[str, Any]:
        """
        Get customer's active subscription plans.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Dict: {"success": bool, "plans": list, "count": int, "message": str}
        """
        try:
            plans = self.subscription_service.get_customer_active_plans(customer_id)
            
            return {
                "success": True,
                "plans": plans,
                "count": len(plans),
                "message": f"Found {len(plans)} active plans"
            }
            
        except Exception as e:
            logger.error(f"MCP get active plans error: {e}")
            return {
                "success": False,
                "plans": [],
                "count": 0,
                "message": f"Error retrieving plans: {str(e)}"
            }
    
    def get_customer_subscription_info(self, customer_id: int) -> Dict[str, Any]:
        """
        Get comprehensive customer subscription information.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Dict: Complete subscription info with success flag
        """
        try:
            info = self.subscription_service.get_customer_subscription_info(customer_id)
            
            if info.get("success", True):  # Default to True if not specified
                return {
                    "success": True,
                    "data": info,
                    "message": "Subscription info retrieved successfully"
                }
            else:
                return {
                    "success": False,
                    "data": None,
                    "message": info.get("error", "Failed to retrieve subscription info")
                }
                
        except Exception as e:
            logger.error(f"MCP get subscription info error: {e}")
            return {
                "success": False,
                "data": None,
                "message": f"Error retrieving subscription info: {str(e)}"
            }
    
    def get_available_plans(self) -> Dict[str, Any]:
        """
        Get all available plans for plan changes.
        
        Returns:
            Dict: {"success": bool, "plans": list, "count": int, "message": str}
        """
        try:
            plans = self.subscription_service.get_available_plans()
            
            return {
                "success": True,
                "plans": plans,
                "count": len(plans),
                "message": f"Found {len(plans)} available plans"
            }
            
        except Exception as e:
            logger.error(f"MCP get available plans error: {e}")
            return {
                "success": False,
                "plans": [],
                "count": 0,
                "message": f"Error retrieving available plans: {str(e)}"
            }
    
    def change_customer_plan(self, customer_id: int, old_plan_id: int, new_plan_id: int) -> Dict[str, Any]:
        """
        Change customer's subscription plan.
        
        Args:
            customer_id: Customer ID
            old_plan_id: Current plan ID to deactivate
            new_plan_id: New plan ID to activate
            
        Returns:
            Dict: Change operation result
        """
        try:
            result = self.subscription_service.change_customer_plan(customer_id, old_plan_id, new_plan_id)
            return result
            
        except Exception as e:
            logger.error(f"MCP plan change error: {e}")
            return {
                "success": False,
                "message": f"Error changing plan: {str(e)}"
            }
    
    # ==================== BILLING OPERATIONS ====================
    
    def get_customer_bills(self, customer_id: int, limit: int = 10) -> Dict[str, Any]:
        """
        Get customer's recent bills.
        
        Args:
            customer_id: Customer ID
            limit: Number of bills to return
            
        Returns:
            Dict: {"success": bool, "bills": list, "count": int, "message": str}
        """
        try:
            bills = self.billing_service.get_customer_bills(customer_id, limit)
            
            return {
                "success": True,
                "bills": bills,
                "count": len(bills),
                "message": f"Found {len(bills)} bills"
            }
            
        except Exception as e:
            logger.error(f"MCP get bills error: {e}")
            return {
                "success": False,
                "bills": [],
                "count": 0,
                "message": f"Error retrieving bills: {str(e)}"
            }
    
    def get_unpaid_bills(self, customer_id: int) -> Dict[str, Any]:
        """
        Get customer's unpaid bills.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Dict: {"success": bool, "bills": list, "count": int, "total_amount": float}
        """
        try:
            bills = self.billing_service.get_unpaid_bills(customer_id)
            total_amount = sum(bill['amount'] for bill in bills)
            
            return {
                "success": True,
                "bills": bills,
                "count": len(bills),
                "total_amount": total_amount,
                "message": f"Found {len(bills)} unpaid bills totaling {total_amount}₺"
            }
            
        except Exception as e:
            logger.error(f"MCP get unpaid bills error: {e}")
            return {
                "success": False,
                "bills": [],
                "count": 0,
                "total_amount": 0,
                "message": f"Error retrieving unpaid bills: {str(e)}"
            }
    
    def create_bill_dispute(self, customer_id: int, bill_id: int, reason: str) -> Dict[str, Any]:
        """
        Create a bill dispute.
        
        Args:
            customer_id: Customer ID
            bill_id: Bill ID to dispute
            reason: Dispute reason
            
        Returns:
            Dict: Dispute creation result
        """
        try:
            result = self.billing_service.create_bill_dispute(customer_id, bill_id, reason)
            return result
            
        except Exception as e:
            logger.error(f"MCP bill dispute error: {e}")
            return {
                "success": False,
                "message": f"Error creating dispute: {str(e)}"
            }
    
    def get_billing_summary(self, customer_id: int) -> Dict[str, Any]:
        """
        Get comprehensive billing summary.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Dict: Billing summary with success flag
        """
        try:
            summary = self.billing_service.get_billing_summary(customer_id)
            
            if "error" not in summary:
                return {
                    "success": True,
                    "summary": summary,
                    "message": "Billing summary retrieved successfully"
                }
            else:
                return {
                    "success": False,
                    "summary": None,
                    "message": summary["error"]
                }
                
        except Exception as e:
            logger.error(f"MCP billing summary error: {e}")
            return {
                "success": False,
                "summary": None,
                "message": f"Error retrieving billing summary: {str(e)}"
            }
    
    # ==================== TECHNICAL SUPPORT OPERATIONS ====================
    
    def get_customer_active_appointment(self, customer_id: int) -> Dict[str, Any]:
        """
        Check if customer has an active appointment.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Dict: {"success": bool, "has_active": bool, "appointment": dict or None}
        """
        try:
            result = self.technical_service.get_customer_active_appointment(customer_id)
            
            return {
                "success": True,
                "has_active": result["has_active"],
                "appointment": result["appointment"],
                "message": "Active appointment check completed"
            }
            
        except Exception as e:
            logger.error(f"MCP active appointment error: {e}")
            return {
                "success": False,
                "has_active": False,
                "appointment": None,
                "message": f"Error checking active appointment: {str(e)}"
            }
    
    def get_available_appointment_slots(self, days_ahead: int = 14) -> Dict[str, Any]:
        """
        Get available appointment time slots.
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            Dict: {"success": bool, "slots": list, "count": int}
        """
        try:
            slots = self.technical_service.get_available_appointment_slots(days_ahead)
            
            return {
                "success": True,
                "slots": slots,
                "count": len(slots),
                "message": f"Found {len(slots)} available appointment slots"
            }
            
        except Exception as e:
            logger.error(f"MCP available slots error: {e}")
            return {
                "success": False,
                "slots": [],
                "count": 0,
                "message": f"Error retrieving available slots: {str(e)}"
            }
    
    def create_appointment(self, customer_id: int, appointment_date: date, appointment_time: str, team_name: str, notes: str = "") -> Dict[str, Any]:
        """
        Create a new technical appointment.
        
        Args:
            customer_id: Customer ID
            appointment_date: Appointment date
            appointment_time: Appointment time (HH:MM)
            team_name: Team name
            notes: Optional notes
            
        Returns:
            Dict: Appointment creation result
        """
        try:
            result = self.technical_service.create_new_appointment(
                customer_id, appointment_date, appointment_time, team_name, notes
            )
            return result
            
        except Exception as e:
            logger.error(f"MCP create appointment error: {e}")
            return {
                "success": False,
                "message": f"Error creating appointment: {str(e)}"
            }
    
    def reschedule_appointment(self, appointment_id: int, customer_id: int, new_date: date, new_time: str, new_team: str = None) -> Dict[str, Any]:
        """
        Reschedule an existing appointment.
        
        Args:
            appointment_id: Appointment ID
            customer_id: Customer ID (for security)
            new_date: New appointment date
            new_time: New appointment time
            new_team: New team (optional)
            
        Returns:
            Dict: Reschedule result
        """
        try:
            result = self.technical_service.update_appointment(
                appointment_id, customer_id, new_date, new_time, new_team
            )
            return result
            
        except Exception as e:
            logger.error(f"MCP reschedule appointment error: {e}")
            return {
                "success": False,
                "message": f"Error rescheduling appointment: {str(e)}"
            }
    
    # ==================== REGISTRATION OPERATIONS ====================
    
    def check_tc_kimlik_exists(self, tc_kimlik_no: str) -> Dict[str, Any]:
        """
        Check if TC kimlik number already exists.
        
        Args:
            tc_kimlik_no: Turkish ID number
            
        Returns:
            Dict: {"success": bool, "exists": bool, "message": str}
        """
        try:
            exists = self.registration_service.check_tc_kimlik_exists(tc_kimlik_no)
            
            return {
                "success": True,
                "exists": exists,
                "message": f"TC kimlik exists: {exists}"
            }
            
        except Exception as e:
            logger.error(f"MCP check TC kimlik error: {e}")
            return {
                "success": False,
                "exists": False,
                "message": f"Error checking TC kimlik: {str(e)}"
            }
    
    def register_new_customer(self, tc_kimlik_no: str, first_name: str, last_name: str, phone_number: str, email: str, city: str, district: str = "", initial_plan_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Register a new customer.
        
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
            Dict: Registration result
        """
        try:
            result = self.registration_service.create_new_customer(
                tc_kimlik_no, first_name, last_name, phone_number,
                email, city, district, initial_plan_id
            )
            return result
            
        except Exception as e:
            logger.error(f"MCP register customer error: {e}")
            return {
                "success": False,
                "message": f"Error registering customer: {str(e)}"
            }


# Global MCP client instance
mcp_client = MCPClient()


# ==================== CONVENIENCE FUNCTIONS ====================

def authenticate_customer(tc_kimlik_no: str) -> Dict[str, Any]:
    """Simple authentication function"""
    return mcp_client.authenticate_customer(tc_kimlik_no)


def get_customer_info(customer_id: int) -> Dict[str, Any]:
    """Get comprehensive customer information"""
    return mcp_client.get_customer_subscription_info(customer_id)


def get_customer_bills(customer_id: int, limit: int = 10) -> Dict[str, Any]:
    """Get customer bills"""
    return mcp_client.get_customer_bills(customer_id, limit)


def create_appointment(customer_id: int, appointment_date: date, appointment_time: str, team_name: str, notes: str = "") -> Dict[str, Any]:
    """Create technical appointment"""
    return mcp_client.create_appointment(customer_id, appointment_date, appointment_time, team_name, notes)


if __name__ == "__main__":
    """Test MCP Client comprehensive functionality"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🔌 Testing Complete MCP Client")
    print("=" * 50)
    
    # Get test customer data
    test_tc = input("Enter a test TC kimlik number: ").strip()
    
    if not test_tc:
        print("❌ No TC kimlik provided")
        exit(1)
    
    print(f"\n🧪 Testing MCP Client with TC: {test_tc}")
    print("-" * 50)
    
    # Test 1: Authentication
    print("1️⃣ Testing Authentication")
    auth_result = mcp_client.authenticate_customer(test_tc)
    
    if auth_result["success"] and auth_result["is_active"]:
        customer_id = auth_result["customer_id"]
        customer_name = f"{auth_result['customer_data']['first_name']} {auth_result['customer_data']['last_name']}"
        print(f"   ✅ Customer authenticated: {customer_name} (ID: {customer_id})")
        
        # Test 2: Get subscription info
        print(f"\n2️⃣ Testing Subscription Info")
        sub_result = mcp_client.get_customer_subscription_info(customer_id)
        
        if sub_result["success"]:
            data = sub_result["data"]
            active_plans = data["active_plans"]
            print(f"   ✅ Subscription info retrieved")
            print(f"      Active Plans: {len(active_plans)}")
            if active_plans:
                for plan in active_plans[:2]:
                    print(f"        • {plan['plan_name']} - {plan['monthly_fee']}₺")
        
        # Test 3: Get billing summary
        print(f"\n3️⃣ Testing Billing Summary")
        billing_result = mcp_client.get_billing_summary(customer_id)
        
        if billing_result["success"]:
            summary = billing_result["summary"]
            print(f"   ✅ Billing summary retrieved")
            print(f"      Total Bills: {summary['total_bills']}")
            print(f"      Outstanding: {summary['outstanding_amount']}₺")
        
        # Test 4: Check active appointment
        print(f"\n4️⃣ Testing Active Appointment Check")
        apt_result = mcp_client.get_customer_active_appointment(customer_id)
        
        if apt_result["success"]:
            if apt_result["has_active"]:
                apt = apt_result["appointment"]
                print(f"   📅 Active appointment found:")
                print(f"      Date: {apt['appointment_date']} at {apt['appointment_hour']}")
            else:
                print(f"   ✅ No active appointment")
        
        # Test 5: Get available plans
        print(f"\n5️⃣ Testing Available Plans")
        plans_result = mcp_client.get_available_plans()
        
        if plans_result["success"]:
            print(f"   ✅ Available plans retrieved: {plans_result['count']} plans")
        
    elif auth_result["success"] and auth_result["exists"] and not auth_result["is_active"]:
        print(f"   ⚠️ Customer exists but is inactive")
        
    elif auth_result["success"] and not auth_result["exists"]:
        print(f"   ❌ Customer not found")
        
        # Test registration for non-customer
        print(f"\n6️⃣ Testing Registration Check")
        tc_check = mcp_client.check_tc_kimlik_exists(test_tc)
        if tc_check["success"]:
            print(f"   TC kimlik check: {tc_check['message']}")
        
    else:
        print(f"   ❌ Authentication failed: {auth_result['message']}")
    
    print("\n✅ MCP Client test completed!")