"""
MCP Client Demo - How Agents Use the MCP Client

This demo shows how your agents (Auth, Classifier, Operation, etc.) 
will use the MCP Client to perform various operations.
"""

import logging
from datetime import date, timedelta

from mcp_client import mcp_client

logging.basicConfig(level=logging.INFO)


def demo_customer_authentication_flow():
    """Demo: Customer authentication flow"""
    
    print("\n🔐 DEMO: Customer Authentication Flow")
    print("=" * 50)
    
    # Simulate user input
    tc_kimlik = input("Enter TC kimlik for demo (or press Enter for simulation): ").strip()
    
    if not tc_kimlik:
        tc_kimlik = "12345678901"  # Demo TC
        print(f"Using demo TC kimlik: {tc_kimlik}")
    
    # 1. Authentication Agent would call this
    print(f"\n1️⃣ Authenticating customer with TC: {tc_kimlik}")
    auth_result = mcp_client.authenticate_customer(tc_kimlik)
    
    if auth_result["success"] and auth_result["is_active"]:
        customer_id = auth_result["customer_id"]
        customer_data = auth_result["customer_data"]
        
        print(f"   ✅ Customer authenticated successfully!")
        print(f"   Customer ID: {customer_id}")
        print(f"   Name: {customer_data['first_name']} {customer_data['last_name']}")
        print(f"   Phone: {customer_data['phone_number']}")
        print(f"   Status: {customer_data['customer_status']}")
        
        return customer_id, customer_data
        
    elif auth_result["success"] and not auth_result["exists"]:
        print(f"   ❌ Customer not found - would route to registration")
        return None, None
        
    else:
        print(f"   ❌ Authentication failed: {auth_result['message']}")
        return None, None


def demo_subscription_management(customer_id: int):
    """Demo: Subscription management operations"""
    
    print(f"\n📋 DEMO: Subscription Management for Customer {customer_id}")
    print("=" * 50)
    
    # 1. Get current subscription info
    print("1️⃣ Getting current subscription info...")
    sub_info = mcp_client.get_customer_subscription_info(customer_id)
    
    if sub_info["success"]:
        data = sub_info["data"]
        active_plans = data["active_plans"]
        
        print(f"   ✅ Found {len(active_plans)} active plans:")
        for plan in active_plans:
            print(f"      • {plan['plan_name']} - {plan['monthly_fee']}₺/month - {plan['quota_gb']}GB")
        
        # 2. Show available plans for changes
        print(f"\n2️⃣ Getting available plans for changes...")
        available_plans = mcp_client.get_available_plans()
        
        if available_plans["success"]:
            plans = available_plans["plans"]
            print(f"   ✅ Found {len(plans)} available plans:")
            for i, plan in enumerate(plans[:3], 1):
                print(f"      {i}. {plan['plan_name']} - {plan['monthly_fee']}₺ - {plan['quota_gb']}GB")
            
            # 3. Demo plan change (if customer has plans and other options exist)
            if active_plans and len(plans) > 1:
                current_plan = active_plans[0]
                
                # Find a different plan
                new_plan = None
                for plan in plans:
                    if plan['plan_id'] != current_plan['plan_id']:
                        new_plan = plan
                        break
                
                if new_plan:
                    print(f"\n3️⃣ Demo plan change simulation:")
                    print(f"   Current: {current_plan['plan_name']} (ID: {current_plan['plan_id']})")
                    print(f"   Would change to: {new_plan['plan_name']} (ID: {new_plan['plan_id']})")
                    
                    # Uncomment to actually perform the change:
                    # change_result = mcp_client.change_customer_plan(customer_id, current_plan['plan_id'], new_plan['plan_id'])
                    # print(f"   Change result: {change_result['message']}")


def demo_billing_operations(customer_id: int):
    """Demo: Billing operations"""
    
    print(f"\n💰 DEMO: Billing Operations for Customer {customer_id}")
    print("=" * 50)
    
    # 1. Get billing summary
    print("1️⃣ Getting billing summary...")
    billing_summary = mcp_client.get_billing_summary(customer_id)
    
    if billing_summary["success"]:
        summary = billing_summary["summary"]
        print(f"   ✅ Billing Summary:")
        print(f"      Total Bills: {summary['total_bills']}")
        print(f"      Paid Bills: {summary['paid_bills']}")
        print(f"      Unpaid Bills: {summary['unpaid_bills']}")
        print(f"      Outstanding Amount: {summary['outstanding_amount']}₺")
        print(f"      Payment Rate: {summary['payment_rate']:.1f}%")
    
    # 2. Get recent bills
    print(f"\n2️⃣ Getting recent bills...")
    recent_bills = mcp_client.get_customer_bills(customer_id, 5)
    
    if recent_bills["success"]:
        bills = recent_bills["bills"]
        print(f"   ✅ Found {len(bills)} recent bills:")
        for bill in bills:
            status_icon = "✅" if bill['status'] == 'paid' else "⏳"
            print(f"      {status_icon} Bill #{bill['bill_id']} - {bill['amount']}₺ - Due: {bill['due_date']} - {bill['status']}")
    
    # 3. Check unpaid bills
    print(f"\n3️⃣ Checking unpaid bills...")
    unpaid_bills = mcp_client.get_unpaid_bills(customer_id)
    
    if unpaid_bills["success"]:
        if unpaid_bills["count"] > 0:
            print(f"   ⏳ Found {unpaid_bills['count']} unpaid bills totaling {unpaid_bills['total_amount']}₺")
            
            # Demo dispute creation (simulation)
            if unpaid_bills["bills"]:
                first_unpaid = unpaid_bills["bills"][0]
                print(f"\n4️⃣ Demo dispute creation for Bill #{first_unpaid['bill_id']}:")
                print(f"   Would create dispute with reason: 'Amount seems higher than expected'")
                
                # Uncomment to actually create dispute:
                # dispute_result = mcp_client.create_bill_dispute(customer_id, first_unpaid['bill_id'], "Amount seems higher than expected")
                # print(f"   Dispute result: {dispute_result['message']}")
        else:
            print(f"   ✅ No unpaid bills found")


def demo_technical_support(customer_id: int):
    """Demo: Technical support operations"""
    
    print(f"\n🔧 DEMO: Technical Support for Customer {customer_id}")
    print("=" * 50)
    
    # 1. Check if customer has active appointment
    print("1️⃣ Checking for active appointments...")
    active_apt = mcp_client.get_customer_active_appointment(customer_id)
    
    if active_apt["success"]:
        if active_apt["has_active"]:
            apt = active_apt["appointment"]
            print(f"   📅 Active appointment found:")
            print(f"      ID: {apt['appointment_id']}")
            print(f"      Date: {apt['appointment_date']} at {apt['appointment_hour']}")
            print(f"      Team: {apt['team_name']}")
            print(f"      Status: {apt['appointment_status']}")
            
            # Demo reschedule option
            print(f"\n2️⃣ Demo reschedule option:")
            tomorrow = date.today() + timedelta(days=1)
            print(f"   Would offer to reschedule to {tomorrow} at 10:00")
            
        else:
            print(f"   ✅ No active appointment found")
            
            # 2. Show available slots for new appointment
            print(f"\n2️⃣ Getting available appointment slots...")
            available_slots = mcp_client.get_available_appointment_slots(7)
            
            if available_slots["success"]:
                slots = available_slots["slots"]
                print(f"   ✅ Found {len(slots)} available slots (showing first 3):")
                for i, slot in enumerate(slots[:3], 1):
                    print(f"      {i}. {slot['date']} ({slot['day_name']}) at {slot['time']} - {slot['team']}")
                
                # Demo appointment creation
                if slots:
                    first_slot = slots[0]
                    print(f"\n3️⃣ Demo appointment creation:")
                    print(f"   Would create appointment for {first_slot['date']} at {first_slot['time']}")
                    print(f"   Team: {first_slot['team']}")
                    print(f"   Notes: 'Internet connection problems'")
                    
                    # Uncomment to actually create appointment:
                    # apt_result = mcp_client.create_appointment(
                    #     customer_id, first_slot['date'], first_slot['time'], 
                    #     first_slot['team'], "Internet connection problems"
                    # )
                    # print(f"   Creation result: {apt_result['message']}")


def demo_registration_flow():
    """Demo: New customer registration flow"""
    
    print(f"\n📝 DEMO: New Customer Registration Flow")
    print("=" * 50)
    
    # Demo TC kimlik that doesn't exist
    demo_tc = "99999999999"
    
    print(f"1️⃣ Checking if TC kimlik {demo_tc} exists...")
    tc_check = mcp_client.check_tc_kimlik_exists(demo_tc)
    
    if tc_check["success"]:
        if not tc_check["exists"]:
            print(f"   ✅ TC kimlik not found - can proceed with registration")
            
            print(f"\n2️⃣ Demo registration process:")
            print(f"   Would collect: First name, Last name, Phone, Email, City")
            print(f"   Would offer initial plan selection")
            print(f"   Would create new customer account")
            
            # Demo registration (simulation)
            print(f"\n3️⃣ Registration simulation:")
            print(f"   TC: {demo_tc}")
            print(f"   Name: Demo Customer")
            print(f"   Phone: +905551234567")
            print(f"   Email: demo@example.com")
            print(f"   City: Istanbul")
            
            # Uncomment to actually create customer:
            # reg_result = mcp_client.register_new_customer(
            #     demo_tc, "Demo", "Customer", "+905551234567", 
            #     "demo@example.com", "Istanbul"
            # )
            # print(f"   Registration result: {reg_result['message']}")
            
        else:
            print(f"   ⚠️ TC kimlik already exists - cannot register")


def main():
    """Main demo function"""
    
    print("🔌 MCP Client Demo - Agent Usage Examples")
    print("=" * 60)
    print("This demo shows how your agents will use the MCP Client")
    print("to perform various customer service operations.")
    
    # Demo 1: Authentication Flow
    customer_id, customer_data = demo_customer_authentication_flow()
    
    if customer_id:
        # Customer exists and is active - show customer operations
        demo_subscription_management(customer_id)
        demo_billing_operations(customer_id)
        demo_technical_support(customer_id)
    else:
        # Customer doesn't exist - show registration flow
        demo_registration_flow()
    
    print(f"\n✅ MCP Client Demo Completed!")
    print("=" * 60)
    print("🎯 Key Takeaways:")
    print("• All operations return consistent JSON responses")
    print("• Error handling is built-in to all functions")
    print("• Agents can easily call any operation they need")
    print("• Authentication provides customer_id for other operations")
    print("• Registration flow handles new customers")


if __name__ == "__main__":
    main()