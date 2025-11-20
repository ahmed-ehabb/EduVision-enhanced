"""
Teacher Session Example
Demonstrates complete teacher workflow with unified API
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.util.auth_client import TeacherAuthClient
import requests

def print_header(title: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_success(message: str):
    """Print success message"""
    print(f"✓ {message}")


def print_error(message: str):
    """Print error message"""
    print(f"✗ {message}")


def print_info(label: str, value: str):
    """Print information"""
    print(f"  {label}: {value}")


class TeacherSessionManager:
    """
    Manages complete teacher workflow with unified API
    """

    def __init__(self, api_url: str = "http://localhost:8002"):
        """
        Initialize teacher session manager

        Args:
            api_url: Base URL of unified API
        """
        self.auth = TeacherAuthClient(api_url)
        self.current_lecture = None
        self.current_session = None

    def login(self, email: str, password: str) -> bool:
        """
        Login as teacher

        Args:
            email: Teacher email
            password: Teacher password

        Returns:
            True if login successful
        """
        try:
            result = self.auth.login(email, password)
            user = result['user']

            print_success(f"Logged in as: {user['full_name']}")
            print_info("Email", user['email'])
            print_info("Role", user['role'])

            return True
        except requests.HTTPError as e:
            try:
                error_detail = e.response.json().get('detail', str(e))
            except:
                error_detail = str(e)
            print_error(f"Login failed: {error_detail}")
            return False

    def register(self, email: str, password: str, full_name: str, department: str) -> bool:
        """
        Register new teacher account

        Args:
            email: Teacher email
            password: Password
            full_name: Full name
            department: Department

        Returns:
            True if registration successful
        """
        try:
            result = self.auth.register(email, password, full_name, department)
            user = result['user']

            print_success(f"Registered: {user['full_name']}")
            print_info("Email", user['email'])
            print_info("Department", department)

            return True
        except requests.HTTPError as e:
            try:
                error_detail = e.response.json().get('detail', str(e))
            except:
                error_detail = str(e)
            print_error(f"Registration failed: {error_detail}")
            return False

    def create_lecture(
        self,
        title: str,
        description: str,
        course_code: str,
        course_name: str
    ) -> bool:
        """
        Create a new lecture

        Args:
            title: Lecture title
            description: Description
            course_code: Course code
            course_name: Course name

        Returns:
            True if creation successful
        """
        try:
            lecture = self.auth.create_lecture(
                title=title,
                description=description,
                course_code=course_code,
                course_name=course_name
            )

            self.current_lecture = lecture
            print_success(f"Created lecture: {lecture['title']}")
            print_info("Lecture ID", lecture['lecture_id'])
            print_info("Course", f"{lecture['course_code']} - {lecture['course_name']}")

            return True
        except requests.HTTPError as e:
            try:
                error_detail = e.response.json().get('detail', str(e))
            except:
                error_detail = str(e)
            print_error(f"Failed to create lecture: {error_detail}")
            return False

    def list_my_lectures(self):
        """List all lectures for this teacher"""
        try:
            lectures = self.auth.list_lectures()

            print_success(f"Found {len(lectures)} lecture(s)")
            for lecture in lectures:
                print(f"\n  - {lecture['title']}")
                print(f"    ID: {lecture['lecture_id']}")
                print(f"    Course: {lecture.get('course_code', 'N/A')} - {lecture.get('course_name', 'N/A')}")
                print(f"    Created: {lecture['created_at']}")

            return lectures
        except requests.HTTPError as e:
            print_error(f"Failed to list lectures: {e}")
            return []

    def create_session_for_lecture(self, lecture_id: str = None) -> bool:
        """
        Create a new session

        Args:
            lecture_id: Lecture ID (uses current lecture if not provided)

        Returns:
            True if creation successful
        """
        if not lecture_id and not self.current_lecture:
            print_error("No lecture specified. Create a lecture first.")
            return False

        lecture_id = lecture_id or self.current_lecture['lecture_id']

        try:
            session = self.auth.create_session(lecture_id)

            self.current_session = session
            print_success(f"Created session")
            print_info("Session Code", session['session_code'])
            print_info("Session ID", session['session_id'])
            print_info("Status", session['status'])

            return True
        except requests.HTTPError as e:
            try:
                error_detail = e.response.json().get('detail', str(e))
            except:
                error_detail = str(e)
            print_error(f"Failed to create session: {error_detail}")
            return False

    def start_current_session(self) -> bool:
        """Start the current session"""
        if not self.current_session:
            print_error("No session to start. Create a session first.")
            return False

        try:
            session_code = self.current_session['session_code']
            result = self.auth.start_session(session_code)

            self.current_session = result
            print_success(f"Started session: {session_code}")
            print_info("Status", result['status'])
            print_info("Started at", result['started_at'])

            print(f"\n  📋 Share this code with students: {session_code}")

            return True
        except requests.HTTPError as e:
            try:
                error_detail = e.response.json().get('detail', str(e))
            except:
                error_detail = str(e)
            print_error(f"Failed to start session: {error_detail}")
            return False

    def end_current_session(self) -> bool:
        """End the current session"""
        if not self.current_session:
            print_error("No session to end.")
            return False

        try:
            session_code = self.current_session['session_code']
            result = self.auth.end_session(session_code)

            print_success(f"Ended session: {session_code}")
            print_info("Status", result['status'])
            print_info("Duration", f"{result['duration']} seconds")

            return True
        except requests.HTTPError as e:
            try:
                error_detail = e.response.json().get('detail', str(e))
            except:
                error_detail = str(e)
            print_error(f"Failed to end session: {error_detail}")
            return False

    def get_session_analytics(self) -> bool:
        """Get analytics for current session"""
        if not self.current_session:
            print_error("No session selected.")
            return False

        try:
            session_id = self.current_session['session_id']
            analytics = self.auth.get_session_analytics(session_id)

            print_success("Session Analytics")

            if 'message' in analytics:
                print_info("Status", analytics['message'])
            else:
                print_info("Total Participants", str(analytics.get('total_participants', 0)))
                print_info("Avg Engagement Score", str(analytics.get('avg_engagement_score', 'N/A')))
                print_info("Distraction Events", str(analytics.get('total_distraction_events', 0)))
                if analytics.get('dominant_emotion'):
                    print_info("Dominant Emotion", analytics['dominant_emotion'])

            return True
        except requests.HTTPError as e:
            try:
                error_detail = e.response.json().get('detail', str(e))
            except:
                error_detail = str(e)
            print_error(f"Failed to get analytics: {error_detail}")
            return False


def main():
    """
    Main demonstration function
    """
    print("=" * 70)
    print("  EduVision - Teacher Session Manager")
    print("=" * 70)

    # Initialize manager
    manager = TeacherSessionManager()

    # Step 1: Authentication
    print_header("STEP 1: Teacher Login")
    if not manager.login("integration.teacher@eduvision.com", "TeacherPass123!"):
        print("\nTry registering first or check credentials")
        return

    # Step 2: Create Lecture
    print_header("STEP 2: Create Lecture")
    if not manager.create_lecture(
        title="Advanced Machine Learning",
        description="Deep dive into ML algorithms and applications",
        course_code="CS501",
        course_name="Advanced ML"
    ):
        return

    # Step 3: List Lectures
    print_header("STEP 3: List My Lectures")
    manager.list_my_lectures()

    # Step 4: Create Session
    print_header("STEP 4: Create Session")
    if not manager.create_session_for_lecture():
        return

    # Step 5: Start Session
    print_header("STEP 5: Start Session")
    if not manager.start_current_session():
        return

    # Step 6: Simulate session time
    print_header("STEP 6: Session Active")
    print("\n  Session is now active! Students can join using the session code.")
    print("  In a real scenario, this is where:")
    print("    - Students would join the session")
    print("    - Engagement data would be collected")
    print("    - Real-time monitoring would occur")

    input("\n  Press Enter to end the session...")

    # Step 7: End Session
    print_header("STEP 7: End Session")
    if not manager.end_current_session():
        return

    # Step 8: Get Analytics
    print_header("STEP 8: Get Session Analytics")
    manager.get_session_analytics()

    print("\n" + "=" * 70)
    print("  Teacher Session Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
