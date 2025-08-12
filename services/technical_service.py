"""
Technical Support Service for MCP Client
Step 2c: Technical appointment management and scheduling
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, date, timedelta, time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import db


logger = logging.getLogger(__name__)


class TechnicalService:
    """
    Service for managing technical support appointments
    """
    
    def __init__(self):
        """Initialize technical service"""
        self.db = db
        logger.info("Technical support service initialized")
    
    def get_customer_active_appointment(self, customer_id: int) -> Dict[str, Any]:
        """
        Check if customer has an active (pending/scheduled) appointment.
        
        Args:
            customer_id: Customer ID
            
        Returns:
            Dict: {"has_active": bool, "appointment": dict or None}
        """
        try:
            # Ensure database connection
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    logger.error("Database connection failed")
                    return {"has_active": False, "appointment": None}
            
            # Query for active appointments (not completed/cancelled)
            query = """
            SELECT 
                appointment_id,
                customer_id,
                team_name,
                appointment_date,
                appointment_hour,
                appointment_status,
                notes
            FROM technical_appointments
            WHERE customer_id = %s 
                AND appointment_status IN ('scheduled', 'pending', 'confirmed')
                AND appointment_date >= CURRENT_DATE
            ORDER BY appointment_date ASC, appointment_hour ASC
            LIMIT 1
            """
            
            appointment = self.db.execute_single(query, (customer_id,))
            
            if appointment:
                logger.info(f"Found active appointment {appointment['appointment_id']} for customer {customer_id}")
                return {"has_active": True, "appointment": appointment}
            else:
                logger.info(f"No active appointment found for customer {customer_id}")
                return {"has_active": False, "appointment": None}
            
        except Exception as e:
            logger.error(f"Error checking active appointments: {e}")
            return {"has_active": False, "appointment": None}
    
    def get_available_appointment_slots(self, days_ahead: int = 14) -> List[Dict[str, Any]]:
        """
        Get available appointment slots for the next N days.
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of available time slots
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return []
            
            # Define working hours and teams
            working_hours = [
                "09:00", "10:00", "11:00", "14:00", "15:00", "16:00", "17:00"
            ]
            teams = ["Technical Team A", "Technical Team B", "Field Support"]
            
            # Get existing appointments for the date range
            start_date = date.today()
            end_date = start_date + timedelta(days=days_ahead)
            
            existing_query = """
            SELECT appointment_date, appointment_hour, team_name, COUNT(*) as bookings
            FROM technical_appointments
            WHERE appointment_date BETWEEN %s AND %s
                AND appointment_status IN ('scheduled', 'pending', 'confirmed')
            GROUP BY appointment_date, appointment_hour, team_name
            """
            
            existing_appointments = self.db.execute_query(existing_query, (start_date, end_date))
            
            # Create a set of booked slots for quick lookup
            booked_slots = set()
            for apt in existing_appointments:
                slot_key = f"{apt['appointment_date']}_{apt['appointment_hour']}_{apt['team_name']}"
                booked_slots.add(slot_key)
            
            # Generate available slots
            available_slots = []
            current_date = start_date
            
            while current_date <= end_date:
                # Skip weekends (optional)
                if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                    
                    for hour in working_hours:
                        for team in teams:
                            slot_key = f"{current_date}_{hour}_{team}"
                            
                            # If slot is not booked, add it to available
                            if slot_key not in booked_slots:
                                available_slots.append({
                                    "date": current_date,
                                    "time": hour,
                                    "team": team,
                                    "datetime_str": f"{current_date} {hour}",
                                    "day_name": current_date.strftime("%A")
                                })
                
                current_date += timedelta(days=1)
            
            logger.info(f"Found {len(available_slots)} available appointment slots")
            return available_slots[:20]  # Limit to first 20 slots
            
        except Exception as e:
            logger.error(f"Error getting available slots: {e}")
            return []
    
    def create_new_appointment(
        self, 
        customer_id: int, 
        appointment_date: date, 
        appointment_time: str, 
        team_name: str,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Create a new technical appointment.
        
        Args:
            customer_id: Customer ID
            appointment_date: Appointment date
            appointment_time: Appointment time (HH:MM format)
            team_name: Assigned team name
            notes: Optional notes about the issue
            
        Returns:
            Dict with creation result
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return {"success": False, "message": "Database connection failed"}
            
            # Check if slot is still available
            conflict_query = """
            SELECT COUNT(*) as conflicts
            FROM technical_appointments
            WHERE appointment_date = %s 
                AND appointment_hour = %s 
                AND team_name = %s
                AND appointment_status IN ('scheduled', 'pending', 'confirmed')
            """
            
            conflict = self.db.execute_single(conflict_query, (appointment_date, appointment_time, team_name))
            
            if conflict and conflict['conflicts'] > 0:
                return {
                    "success": False,
                    "message": "Selected time slot is no longer available"
                }
            
            # Check if customer already has an active appointment
            active_check = self.get_customer_active_appointment(customer_id)
            if active_check["has_active"]:
                return {
                    "success": False,
                    "message": f"Customer already has an active appointment (ID: {active_check['appointment']['appointment_id']})"
                }
            
            # Create the appointment
            insert_query = """
            INSERT INTO technical_appointments (
                customer_id, team_name, appointment_date, appointment_hour, 
                appointment_status, notes
            )
            VALUES (%s, %s, %s, %s, 'scheduled', %s)
            RETURNING appointment_id
            """
            
            with self.db.connection.cursor() as cursor:
                cursor.execute(insert_query, (
                    customer_id, team_name, appointment_date, appointment_time, notes
                ))
                result = cursor.fetchone()
                self.db.connection.commit()
                
                appointment_id = result['appointment_id']
            
            logger.info(f"Created appointment {appointment_id} for customer {customer_id}")
            
            return {
                "success": True,
                "message": "Appointment scheduled successfully",
                "appointment_id": appointment_id,
                "appointment_details": {
                    "appointment_id": appointment_id,
                    "customer_id": customer_id,
                    "date": appointment_date.isoformat(),
                    "time": appointment_time,
                    "team": team_name,
                    "status": "scheduled",
                    "notes": notes
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating appointment: {e}")
            if self.db.connection:
                self.db.connection.rollback()
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def update_appointment(
        self, 
        appointment_id: int, 
        customer_id: int,
        new_date: date, 
        new_time: str, 
        new_team: str = None
    ) -> Dict[str, Any]:
        """
        Update an existing appointment (reschedule).
        
        Args:
            appointment_id: Appointment ID to update
            customer_id: Customer ID (for security)
            new_date: New appointment date
            new_time: New appointment time
            new_team: New team (optional)
            
        Returns:
            Dict with update result
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return {"success": False, "message": "Database connection failed"}
            
            # Verify appointment belongs to customer
            verify_query = """
            SELECT appointment_id, team_name, appointment_status
            FROM technical_appointments
            WHERE appointment_id = %s AND customer_id = %s
            """
            
            existing = self.db.execute_single(verify_query, (appointment_id, customer_id))
            
            if not existing:
                return {
                    "success": False,
                    "message": "Appointment not found or does not belong to customer"
                }
            
            # Use existing team if new_team not specified
            if not new_team:
                new_team = existing['team_name']
            
            # Check if new slot is available
            conflict_query = """
            SELECT COUNT(*) as conflicts
            FROM technical_appointments
            WHERE appointment_date = %s 
                AND appointment_hour = %s 
                AND team_name = %s
                AND appointment_status IN ('scheduled', 'pending', 'confirmed')
                AND appointment_id != %s
            """
            
            conflict = self.db.execute_single(conflict_query, (new_date, new_time, new_team, appointment_id))
            
            if conflict and conflict['conflicts'] > 0:
                return {
                    "success": False,
                    "message": "New time slot is not available"
                }
            
            # Update the appointment
            update_query = """
            UPDATE technical_appointments
            SET appointment_date = %s,
                appointment_hour = %s,
                team_name = %s,
                appointment_status = 'scheduled'
            WHERE appointment_id = %s AND customer_id = %s
            """
            
            with self.db.connection.cursor() as cursor:
                cursor.execute(update_query, (new_date, new_time, new_team, appointment_id, customer_id))
                rows_updated = cursor.rowcount
                self.db.connection.commit()
            
            if rows_updated > 0:
                logger.info(f"Updated appointment {appointment_id} for customer {customer_id}")
                
                return {
                    "success": True,
                    "message": "Appointment rescheduled successfully",
                    "appointment_id": appointment_id,
                    "new_details": {
                        "date": new_date.isoformat(),
                        "time": new_time,
                        "team": new_team
                    }
                }
            else:
                return {"success": False, "message": "Failed to update appointment"}
            
        except Exception as e:
            logger.error(f"Error updating appointment: {e}")
            if self.db.connection:
                self.db.connection.rollback()
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def get_customer_appointment_history(self, customer_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get customer's appointment history.
        
        Args:
            customer_id: Customer ID
            limit: Number of appointments to return
            
        Returns:
            List of appointment history
        """
        try:
            if not self.db.is_connected():
                success = self.db.connect()
                if not success:
                    return []
            
            query = """
            SELECT 
                appointment_id,
                customer_id,
                team_name,
                appointment_date,
                appointment_hour,
                appointment_status,
                notes
            FROM technical_appointments
            WHERE customer_id = %s
            ORDER BY appointment_date DESC, appointment_hour DESC
            LIMIT %s
            """
            
            history = self.db.execute_query(query, (customer_id, limit))
            
            logger.info(f"Retrieved {len(history)} appointment records for customer {customer_id}")
            return history
            
        except Exception as e:
            logger.error(f"Error getting appointment history: {e}")
            return []


# Global technical service instance
technical_service = TechnicalService()


def get_customer_active_appointment(customer_id: int) -> Dict[str, Any]:
    """Simple function to check active appointment"""
    return technical_service.get_customer_active_appointment(customer_id)


def get_available_appointment_slots(days_ahead: int = 14) -> List[Dict[str, Any]]:
    """Simple function to get available slots"""
    return technical_service.get_available_appointment_slots(days_ahead)


def create_new_appointment(customer_id: int, appointment_date: date, appointment_time: str, team_name: str, notes: str = "") -> Dict[str, Any]:
    """Simple function to create appointment"""
    return technical_service.create_new_appointment(customer_id, appointment_date, appointment_time, team_name, notes)


def update_appointment(appointment_id: int, customer_id: int, new_date: date, new_time: str, new_team: str = None) -> Dict[str, Any]:
    """Simple function to update appointment"""
    return technical_service.update_appointment(appointment_id, customer_id, new_date, new_time, new_team)


if __name__ == "__main__":
    # Test technical service
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 Testing Technical Support Service")
    print("=" * 40)
    
    # Get test customer ID
    test_customer_id = input("Enter a test customer ID from your database: ").strip()
    
    try:
        test_customer_id = int(test_customer_id)
    except ValueError:
        print("❌ Invalid customer ID")
        exit(1)
    
    print(f"\n🧪 Testing technical support functions for customer ID: {test_customer_id}")
    print("-" * 50)
    
    # Test 1: Check active appointment
    print("1️⃣ Testing get_customer_active_appointment()")
    active_result = get_customer_active_appointment(test_customer_id)
    
    if active_result["has_active"]:
        apt = active_result["appointment"]
        print(f"   📅 Active appointment found:")
        print(f"      ID: {apt['appointment_id']}")
        print(f"      Date: {apt['appointment_date']} at {apt['appointment_hour']}")
        print(f"      Team: {apt['team_name']}")
        print(f"      Status: {apt['appointment_status']}")
        print(f"      Notes: {apt['notes'][:50]}...")
    else:
        print("   ✅ No active appointment found")
    
    # Test 2: Get available slots
    print(f"\n2️⃣ Testing get_available_appointment_slots()")
    available_slots = get_available_appointment_slots(7)  # Next 7 days
    
    if available_slots:
        print(f"   📋 Found {len(available_slots)} available slots (showing first 5):")
        for i, slot in enumerate(available_slots[:5], 1):
            print(f"      {i}. {slot['date']} ({slot['day_name']}) at {slot['time']} - {slot['team']}")
    else:
        print("   ⚠️ No available slots found")
    
    # Test 3: Create appointment (if no active appointment and slots available)
    if not active_result["has_active"] and available_slots:
        print(f"\n3️⃣ Testing create_new_appointment()")
        
        # Use first available slot
        first_slot = available_slots[0]
        
        create_result = create_new_appointment(
            test_customer_id,
            first_slot['date'],
            first_slot['time'],
            first_slot['team'],
            "Test appointment - internet connection issues"
        )
        
        if create_result["success"]:
            print(f"   ✅ Appointment created: ID {create_result['appointment_id']}")
            print(f"      Date: {create_result['appointment_details']['date']}")
            print(f"      Time: {create_result['appointment_details']['time']}")
        else:
            print(f"   ❌ Creation failed: {create_result['message']}")
    
    # Test 4: Get appointment history
    print(f"\n4️⃣ Testing get_customer_appointment_history()")
    history = technical_service.get_customer_appointment_history(test_customer_id)
    
    if history:
        print(f"   📜 Found {len(history)} appointment records:")
        for apt in history:
            status_icon = "✅" if apt['appointment_status'] == 'completed' else "📅"
            print(f"      {status_icon} {apt['appointment_date']} at {apt['appointment_hour']} - {apt['appointment_status']}")
    else:
        print("   ✅ No appointment history")
    
    print("\n✅ Technical support service test completed!")