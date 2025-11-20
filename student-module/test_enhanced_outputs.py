"""
Test script for enhanced output system
This will test the enhanced outputs without running the full camera system
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List

# Mock Stats class to test
@dataclass
class MockStats:
    """Mock session statistics for testing."""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    alerts: List[Dict] = field(default_factory=list)
    concentration_scores: List[float] = field(default_factory=list)
    student_name: str = "Test Student"
    status_durations: Dict[str, float] = field(default_factory=lambda: {
        "Highly Engaged": 120.0,  # 2 minutes
        "Engaged": 180.0,         # 3 minutes
        "Distracted": 60.0        # 1 minute
    })
    last_status_change: float = field(default_factory=time.time)
    current_status: str = "Engaged"
    yawn_count: int = 3
    blink_count: int = 45
    face_absence_count: int = 2
    face_absence_duration: float = 30.0

def test_enhanced_outputs():
    """Test the enhanced output system."""
    print("🧪 Testing Enhanced Output System...")
    
    try:
        from enhanced_outputs import integrate_with_existing_system
        print("✅ Enhanced outputs module imported successfully")
        
        # Create mock stats
        mock_stats = MockStats()
        
        # Add some mock alerts
        current_time = time.time()
        mock_stats.alerts = [
            {"time": current_time - 120, "message": "Yawning detected"},
            {"time": current_time - 60, "message": "Looking down"},
            {"time": current_time - 30, "message": "Slouching detected"}
        ]
        
        # Add some concentration scores
        mock_stats.concentration_scores = [1.0, 0.5, 0.5, 1.0, 0.0, 0.5, 1.0]
        
        print(f"📊 Testing with mock data:")
        print(f"   Student: {mock_stats.student_name}")
        print(f"   Current Status: {mock_stats.current_status}")
        print(f"   Yawn Count: {mock_stats.yawn_count}")
        print(f"   Alerts: {len(mock_stats.alerts)}")
        print(f"   Session Duration: {time.time() - mock_stats.start_time:.1f} seconds")
        
        # Test the integration function
        print("\n🔄 Generating enhanced outputs...")
        student_outputs, teacher_outputs = integrate_with_existing_system(mock_stats)
        
        # Display results
        print("\n📱 STUDENT INTERFACE OUTPUTS:")
        print("=" * 50)
        
        # Live status
        live_status = student_outputs.get('live_status', {})
        print(f"📊 Live Status:")
        print(f"   • Current Engagement: {live_status.get('current_engagement', 'N/A')}")
        print(f"   • Session Time: {live_status.get('session_time', 'N/A')}")
        print(f"   • Face Detected: {live_status.get('face_detected', False)}")
        print(f"   • Identity Verified: {live_status.get('identity_verified', False)}")
        
        # Warning system
        warning_system = student_outputs.get('warning_system', {})
        print(f"\n⚠️ Warning System:")
        print(f"   • Level: {warning_system.get('level', 0)}")
        print(f"   • Message: {warning_system.get('message', 'None')}")
        print(f"   • Sound Enabled: {warning_system.get('sound_enabled', False)}")
        
        # Personal metrics
        personal_metrics = student_outputs.get('personal_metrics', {})
        print(f"\n📈 Personal Metrics:")
        print(f"   • Engagement Score: {personal_metrics.get('engagement_score', 0)}%")
        print(f"   • Attention Trend: {personal_metrics.get('attention_trend', 'N/A')}")
        
        behavioral = personal_metrics.get('behavioral_indicators', {})
        print(f"   • Yawn Count: {behavioral.get('yawn_count', 0)}")
        print(f"   • Blink Rate: {behavioral.get('blink_rate', 0)}")
        print(f"   • Head Movements: {behavioral.get('head_movements', 0)}")
        print(f"   • Posture Score: {behavioral.get('posture_score', 0)}")
        
        print("\n👩‍🏫 TEACHER INTERFACE OUTPUTS:")
        print("=" * 50)
        
        # Class overview
        class_overview = teacher_outputs.get('class_overview', {})
        print(f"📊 Class Overview:")
        print(f"   • Total Students: {class_overview.get('total_students', 0)}")
        print(f"   • Active Students: {class_overview.get('active_students', 0)}")
        print(f"   • Average Engagement: {class_overview.get('average_engagement', 0)}%")
        print(f"   • Active Alerts: {class_overview.get('alerts_active', 0)}")
        
        # Student report
        student_report = teacher_outputs.get('student_report', {})
        session_summary = student_report.get('session_summary', {})
        print(f"\n📋 Student Report:")
        print(f"   • Student: {student_report.get('name', 'N/A')}")
        print(f"   • Overall Score: {session_summary.get('overall_score', 0)}%")
        print(f"   • Grade: {session_summary.get('grade', 'N/A')}")
        print(f"   • Trend: {session_summary.get('trend', 'N/A')}")
        
        # Active alerts
        active_alerts = teacher_outputs.get('active_alerts', [])
        print(f"\n🚨 Active Alerts: {len(active_alerts)}")
        for i, alert in enumerate(active_alerts[:3]):  # Show first 3
            print(f"   {i+1}. [{alert.get('alert_level', 'unknown')}] {alert.get('message', 'No message')}")
        
        print(f"\n✅ Enhanced outputs generated successfully!")
        print(f"📁 Check the 'frontend_outputs' directory for JSON files")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing enhanced outputs: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Enhanced Output Test")
    print("-" * 50)
    
    success = test_enhanced_outputs()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Test completed successfully!")
        print("💡 You can now run 'python run_local_cv.py' to see enhanced outputs in action")
    else:
        print("🔧 Test failed - please check the error messages above")
    
    print("\n🔍 Next steps:")
    print("1. Run this test to verify outputs work")
    print("2. Run 'python run_local_cv.py' for full system")
    print("3. Check 'frontend_outputs/' directory for JSON files")
    print("4. Build your frontend to consume these JSON outputs") 