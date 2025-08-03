"""
MCP Operations Configuration

This module defines all available MCP operations and their mappings
for easy reference by agents and documentation.
"""

from enum import Enum
from typing import Dict, List, Any


class MCPOperations:
    """
    Centralized definition of all MCP operations available in the system.
    """
    
    # ==================== AUTHENTICATION OPERATIONS ====================
    AUTHENTICATION = {
        "authenticate_customer": {
            "description": "Authenticate customer by TC kimlik number",
            "parameters": ["tc_kimlik_no: str"],
            "returns": "Dict with authentication result",
            "example": "authenticate_customer('12345678901')"
        }
    }
    
    # ==================== SUBSCRIPTION OPERATIONS ====================
    SUBSCRIPTION = {
        "get_customer_active_plans": {
            "description": "Get customer's currently active subscription plans",
            "parameters": ["customer_id: int"],
            "returns": "Dict with active plans list",
            "example": "get_customer_active_plans(123)"
        },
        "get_customer_subscription_info": {
            "description": "Get comprehensive subscription information",
            "parameters": ["customer_id: int"],
            "returns": "Dict with complete subscription data",
            "example": "get_customer_subscription_info(123)"
        },
        "get_available_plans": {
            "description": "Get all available plans for plan changes",
            "parameters": [],
            "returns": "Dict with available plans list",
            "example": "get_available_plans()"
        },
        "change_customer_plan": {
            "description": "Change customer's subscription plan",
            "parameters": ["customer_id: int", "old_plan_id: int", "new_plan_id: int"],
            "returns": "Dict with change operation result",
            "example": "change_customer_plan(123, 1, 2)"
        }
    }
    
    # ==================== BILLING OPERATIONS ====================
    BILLING = {
        "get_customer_bills": {
            "description": "Get customer's recent bills",
            "parameters": ["customer_id: int", "limit: int = 10"],
            "returns": "Dict with bills list",
            "example": "get_customer_bills(123, 5)"
        },
        "get_unpaid_bills": {
            "description": "Get customer's unpaid bills only",
            "parameters": ["customer_id: int"],
            "returns": "Dict with unpaid bills and total amount",
            "example": "get_unpaid_bills(123)"
        },
        "get_billing_summary": {
            "description": "Get comprehensive billing statistics",
            "parameters": ["customer_id: int"],
            "returns": "Dict with billing summary",
            "example": "get_billing_summary(123)"
        },
        "create_bill_dispute": {
            "description": "Create a dispute for a specific bill",
            "parameters": ["customer_id: int", "bill_id: int", "reason: str"],
            "returns": "Dict with dispute creation result",
            "example": "create_bill_dispute(123, 456, 'Amount seems incorrect')"
        }
    }
    
    # ==================== TECHNICAL SUPPORT OPERATIONS ====================
    TECHNICAL_SUPPORT = {
        "get_customer_active_appointment": {
            "description": "Check if customer has an active technical appointment",
            "parameters": ["customer_id: int"],
            "returns": "Dict with active appointment info",
            "example": "get_customer_active_appointment(123)"
        },
        "get_available_appointment_slots": {
            "description": "Get available appointment time slots",
            "parameters": ["days_ahead: int = 14"],
            "returns": "Dict with available slots",
            "example": "get_available_appointment_slots(7)"
        },
        "create_appointment": {
            "description": "Create a new technical support appointment",
            "parameters": ["customer_id: int", "appointment_date: date", "appointment_time: str", "team_name: str", "notes: str = ''"],
            "returns": "Dict with appointment creation result",
            "example": "create_appointment(123, date(2024,2,15), '14:00', 'Technical Team A', 'Internet issues')"
        },
        "reschedule_appointment": {
            "description": "Reschedule an existing appointment",
            "parameters": ["appointment_id: int", "customer_id: int", "new_date: date", "new_time: str", "new_team: str = None"],
            "returns": "Dict with reschedule result",
            "example": "reschedule_appointment(789, 123, date(2024,2,16), '10:00')"
        }
    }
    
    # ==================== REGISTRATION OPERATIONS ====================
    REGISTRATION = {
        "check_tc_kimlik_exists": {
            "description": "Check if TC kimlik number already exists in system",
            "parameters": ["tc_kimlik_no: str"],
            "returns": "Dict with existence check result",
            "example": "check_tc_kimlik_exists('12345678901')"
        },
        "register_new_customer": {
            "description": "Register a new customer account",
            "parameters": ["tc_kimlik_no: str", "first_name: str", "last_name: str", "phone_number: str", "email: str", "city: str", "district: str = ''", "initial_plan_id: int = None"],
            "returns": "Dict with registration result",
            "example": "register_new_customer('12345678901', 'John', 'Doe', '+905551234567', 'john@email.com', 'Istanbul')"
        }
    }
    
    @classmethod
    def get_all_operations(cls) -> Dict[str, Dict]:
        """Get all operations organized by category"""
        return {
            "authentication": cls.AUTHENTICATION,
            "subscription": cls.SUBSCRIPTION,
            "billing": cls.BILLING,
            "technical_support": cls.TECHNICAL_SUPPORT,
            "registration": cls.REGISTRATION
        }
    
    @classmethod
    def get_operation_list(cls) -> List[str]:
        """Get flat list of all operation names"""
        operations = []
        for category in cls.get_all_operations().values():
            operations.extend(category.keys())
        return operations
    
    @classmethod
    def find_operation(cls, operation_name: str) -> Dict[str, Any]:
        """Find operation details by name"""
        for category_name, category in cls.get_all_operations().items():
            if operation_name in category:
                result = category[operation_name].copy()
                result["category"] = category_name
                return result
        return None


# Customer Service Operation Mapping (based on your 5 operations)
class CustomerServiceOperations:
    """
    Maps your 5 customer service operations to MCP functions.
    """
    
    OPERATION_1_SUBSCRIPTION_CHANGES = [
        "get_customer_active_plans",
        "get_available_plans", 
        "change_customer_plan"
    ]
    
    OPERATION_2_TECHNICAL_SUPPORT = [
        "get_customer_active_appointment",
        "get_available_appointment_slots",
        "create_appointment",
        "reschedule_appointment"
    ]
    
    OPERATION_3_SUBSCRIPTION_INFO = [
        "get_customer_active_plans",
        "get_customer_subscription_info"
    ]
    
    OPERATION_4_BILL_DISPUTES = [
        "get_customer_bills",
        "get_unpaid_bills",
        "get_billing_summary",
        "create_bill_dispute"
    ]
    
    OPERATION_5_FAQ = [
        # Handled by SSS Agent with vector database
        # No MCP operations needed
    ]
    
    OPERATION_6_NEW_CUSTOMER = [
        "check_tc_kimlik_exists",
        "register_new_customer"
    ]
    
    @classmethod
    def get_operations_for_service(cls, operation_number: int) -> List[str]:
        """Get MCP operations needed for a specific customer service operation"""
        mapping = {
            1: cls.OPERATION_1_SUBSCRIPTION_CHANGES,
            2: cls.OPERATION_2_TECHNICAL_SUPPORT,
            3: cls.OPERATION_3_SUBSCRIPTION_INFO,
            4: cls.OPERATION_4_BILL_DISPUTES,
            5: [],  # FAQ handled by vector database
            6: cls.OPERATION_6_NEW_CUSTOMER
        }
        return mapping.get(operation_number, [])


if __name__ == "__main__":
    """Display all available MCP operations"""
    
    print("🔌 MCP Operations Reference")
    print("=" * 60)
    
    all_ops = MCPOperations.get_all_operations()
    
    for category_name, operations in all_ops.items():
        print(f"\n📋 {category_name.upper()} OPERATIONS:")
        print("-" * 40)
        
        for op_name, details in operations.items():
            print(f"• {op_name}")
            print(f"  Description: {details['description']}")
            print(f"  Parameters: {', '.join(details['parameters'])}")
            print(f"  Example: {details['example']}")
            print()
    
    print(f"\n📊 SUMMARY:")
    print(f"Total Operations: {len(MCPOperations.get_operation_list())}")
    print(f"Categories: {len(all_ops)}")
    
    print(f"\n🎯 CUSTOMER SERVICE MAPPING:")
    print("-" * 40)
    
    for i in range(1, 7):
        ops = CustomerServiceOperations.get_operations_for_service(i)
        operation_names = {
            1: "Subscription & Tariff Changes",
            2: "Technical Support - Internet", 
            3: "Current Subscription Info Query",
            4: "Bill Disputes & Understanding",
            5: "FAQ (Vector Database)",
            6: "New Customer Registration"
        }
        
        print(f"{i}. {operation_names[i]}")
        if ops:
            for op in ops:
                print(f"   • {op}")
        else:
            print(f"   • Handled by SSS Agent (Vector Database)")
        print()