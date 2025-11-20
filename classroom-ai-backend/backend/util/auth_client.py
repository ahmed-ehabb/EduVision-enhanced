"""
Authentication Client for Teacher Module
Handles authentication with the unified EduVision API
"""

import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import os

class TeacherAuthClient:
    """Client for handling teacher authentication with the unified API"""

    def __init__(self, api_base_url: str = "http://localhost:8002"):
        """
        Initialize the authentication client

        Args:
            api_base_url: Base URL of the unified API server
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.user_info: Optional[Dict[str, Any]] = None

    def register(
        self,
        email: str,
        password: str,
        full_name: str,
        department: str
    ) -> Dict[str, Any]:
        """
        Register a new teacher user

        Args:
            email: Teacher email address
            password: Password (min 8 chars, 1 uppercase, 1 lowercase, 1 digit)
            full_name: Teacher's full name
            department: Teacher's department

        Returns:
            Registration response with tokens and user info

        Raises:
            requests.HTTPError: If registration fails
        """
        url = f"{self.api_base_url}/auth/register"
        data = {
            "email": email,
            "password": password,
            "full_name": full_name,
            "role": "teacher",
            "department": department
        }

        response = requests.post(url, json=data)
        response.raise_for_status()

        result = response.json()
        self._store_tokens(result)

        return result

    def login(self, email: str, password: str) -> Dict[str, Any]:
        """
        Login with email and password

        Args:
            email: User email
            password: User password

        Returns:
            Login response with tokens and user info

        Raises:
            requests.HTTPError: If login fails
        """
        url = f"{self.api_base_url}/auth/login"
        data = {
            "email": email,
            "password": password
        }

        response = requests.post(url, json=data)
        response.raise_for_status()

        result = response.json()
        self._store_tokens(result)

        return result

    def logout(self) -> bool:
        """
        Logout and revoke refresh token

        Returns:
            True if logout was successful
        """
        if not self.refresh_token:
            return False

        url = f"{self.api_base_url}/auth/logout"
        headers = self._get_auth_headers()

        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()
            self._clear_tokens()
            return True
        except requests.HTTPError:
            return False

    def refresh_access_token(self) -> bool:
        """
        Refresh the access token using refresh token

        Returns:
            True if refresh was successful
        """
        if not self.refresh_token:
            return False

        url = f"{self.api_base_url}/auth/refresh"
        headers = {"Authorization": f"Bearer {self.refresh_token}"}

        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()

            result = response.json()
            self.access_token = result["access_token"]
            self.token_expires_at = datetime.now() + timedelta(seconds=result.get("expires_in", 86400))

            return True
        except requests.HTTPError:
            return False

    # ============================================================================
    # LECTURE MANAGEMENT
    # ============================================================================

    def create_lecture(
        self,
        title: str,
        description: Optional[str] = None,
        course_code: Optional[str] = None,
        course_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new lecture

        Args:
            title: Lecture title
            description: Lecture description
            course_code: Course code (e.g., "CS401")
            course_name: Course name (e.g., "Machine Learning")

        Returns:
            Created lecture information

        Raises:
            requests.HTTPError: If creation fails
        """
        self._ensure_valid_token()

        url = f"{self.api_base_url}/api/lectures"
        headers = self._get_auth_headers()
        data = {
            "title": title,
            "description": description,
            "course_code": course_code,
            "course_name": course_name
        }

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()

        return response.json()

    def list_lectures(self) -> List[Dict[str, Any]]:
        """
        List all lectures for the authenticated teacher

        Returns:
            List of lectures

        Raises:
            requests.HTTPError: If request fails
        """
        self._ensure_valid_token()

        url = f"{self.api_base_url}/api/lectures"
        headers = self._get_auth_headers()

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        return response.json()

    def get_lecture(self, lecture_id: str) -> Dict[str, Any]:
        """
        Get lecture details

        Args:
            lecture_id: Lecture UUID

        Returns:
            Lecture information

        Raises:
            requests.HTTPError: If request fails
        """
        self._ensure_valid_token()

        url = f"{self.api_base_url}/api/lectures/{lecture_id}"
        headers = self._get_auth_headers()

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        return response.json()

    # ============================================================================
    # SESSION MANAGEMENT
    # ============================================================================

    def create_session(
        self,
        lecture_id: str,
        scheduled_at: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new session for a lecture

        Args:
            lecture_id: Lecture UUID
            scheduled_at: Optional scheduled start time (ISO format)

        Returns:
            Created session with session_code

        Raises:
            requests.HTTPError: If creation fails
        """
        self._ensure_valid_token()

        url = f"{self.api_base_url}/api/sessions"
        headers = self._get_auth_headers()
        data = {
            "lecture_id": lecture_id,
            "scheduled_at": scheduled_at
        }

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()

        return response.json()

    def start_session(self, session_code: str) -> Dict[str, Any]:
        """
        Start a session

        Args:
            session_code: Session code (e.g., "ABC123")

        Returns:
            Updated session information

        Raises:
            requests.HTTPError: If request fails
        """
        self._ensure_valid_token()

        url = f"{self.api_base_url}/api/sessions/{session_code}/start"
        headers = self._get_auth_headers()

        response = requests.post(url, headers=headers)
        response.raise_for_status()

        return response.json()

    def end_session(self, session_code: str) -> Dict[str, Any]:
        """
        End a session

        Args:
            session_code: Session code (e.g., "ABC123")

        Returns:
            Updated session with duration

        Raises:
            requests.HTTPError: If request fails
        """
        self._ensure_valid_token()

        url = f"{self.api_base_url}/api/sessions/{session_code}/end"
        headers = self._get_auth_headers()

        response = requests.post(url, headers=headers)
        response.raise_for_status()

        return response.json()

    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """
        Get analytics for a session

        Args:
            session_id: Session UUID

        Returns:
            Session analytics data

        Raises:
            requests.HTTPError: If request fails
        """
        self._ensure_valid_token()

        url = f"{self.api_base_url}/api/sessions/{session_id}/analytics"
        headers = self._get_auth_headers()

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        return response.json()

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        return self.access_token is not None

    def _store_tokens(self, auth_response: Dict[str, Any]) -> None:
        """Store tokens from authentication response"""
        self.access_token = auth_response.get("access_token")
        self.refresh_token = auth_response.get("refresh_token")
        self.user_info = auth_response.get("user")

        expires_in = auth_response.get("expires_in", 86400)  # Default 24h
        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

    def _clear_tokens(self) -> None:
        """Clear stored tokens"""
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        self.user_info = None

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers for API requests"""
        if not self.access_token:
            raise ValueError("Not authenticated. Please login first.")

        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }

    def _ensure_valid_token(self) -> None:
        """Ensure access token is valid, refresh if needed"""
        if not self.access_token:
            raise ValueError("Not authenticated. Please login first.")

        # Check if token is expired or about to expire (within 5 minutes)
        if self.token_expires_at and datetime.now() >= (self.token_expires_at - timedelta(minutes=5)):
            if not self.refresh_access_token():
                raise ValueError("Token expired and refresh failed. Please login again.")

    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Get current user information"""
        return self.user_info

    def save_session(self, filepath: str = "teacher_session.json") -> None:
        """
        Save current session to file

        Args:
            filepath: Path to save session file
        """
        if not self.is_authenticated():
            return

        session_data = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_expires_at": self.token_expires_at.isoformat() if self.token_expires_at else None,
            "user_info": self.user_info
        }

        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)

    def load_session(self, filepath: str = "teacher_session.json") -> bool:
        """
        Load session from file

        Args:
            filepath: Path to session file

        Returns:
            True if session loaded successfully
        """
        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)

            self.access_token = session_data.get("access_token")
            self.refresh_token = session_data.get("refresh_token")
            self.user_info = session_data.get("user_info")

            expires_str = session_data.get("token_expires_at")
            if expires_str:
                self.token_expires_at = datetime.fromisoformat(expires_str)

            # Try to refresh if token is expired
            if self.token_expires_at and datetime.now() >= self.token_expires_at:
                return self.refresh_access_token()

            return True
        except Exception:
            return False


# Example usage
if __name__ == "__main__":
    # Create client
    client = TeacherAuthClient()

    # Example registration
    try:
        result = client.register(
            email="teacher@example.com",
            password="SecurePass123",
            full_name="Dr. Jane Smith",
            department="Computer Science"
        )
        print(f"Registered successfully: {result['user']['email']}")
    except requests.HTTPError as e:
        print(f"Registration failed: {e}")

    # Example login
    try:
        result = client.login("teacher@example.com", "SecurePass123")
        print(f"Logged in as: {result['user']['email']}")
        print(f"Role: {result['user']['role']}")
    except requests.HTTPError as e:
        print(f"Login failed: {e}")

    # Example lecture creation
    if client.is_authenticated():
        try:
            lecture = client.create_lecture(
                title="Introduction to AI",
                description="Fundamentals of Artificial Intelligence",
                course_code="CS401",
                course_name="Artificial Intelligence"
            )
            print(f"Created lecture: {lecture['title']} (ID: {lecture['lecture_id']})")

            # Create session
            session = client.create_session(lecture['lecture_id'])
            print(f"Created session: Code={session['session_code']}, ID={session['session_id']}")

            # Start session
            started = client.start_session(session['session_code'])
            print(f"Started session: Status={started['status']}")

        except requests.HTTPError as e:
            print(f"Failed: {e}")
