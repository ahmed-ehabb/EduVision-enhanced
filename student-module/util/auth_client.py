"""
Authentication Client for Student Module
Handles authentication with the unified EduVision API
"""

import requests
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import json
import os

class AuthClient:
    """Client for handling authentication with the unified API"""

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
        student_number: str,
        major: str,
        year_of_study: int
    ) -> Dict[str, Any]:
        """
        Register a new student user

        Args:
            email: Student email address
            password: Password (min 8 chars, 1 uppercase, 1 lowercase, 1 digit)
            full_name: Student's full name
            student_number: Student ID number
            major: Student's major/program
            year_of_study: Current year of study

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
            "role": "student",
            "student_number": student_number,
            "major": major,
            "year_of_study": year_of_study
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

    def join_session(self, session_code: str) -> Dict[str, Any]:
        """
        Join a lecture session using session code

        Args:
            session_code: The session code provided by teacher

        Returns:
            Session join response

        Raises:
            requests.HTTPError: If join fails
        """
        self._ensure_valid_token()

        url = f"{self.api_base_url}/api/sessions/{session_code}/join"
        headers = self._get_auth_headers()

        response = requests.post(url, headers=headers)
        response.raise_for_status()

        return response.json()

    def log_engagement_events(
        self,
        session_id: str,
        events: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Batch log engagement events for a session

        Args:
            session_id: UUID of the session
            events: List of engagement event dictionaries

        Returns:
            Response with count of logged events

        Raises:
            requests.HTTPError: If logging fails
        """
        self._ensure_valid_token()

        url = f"{self.api_base_url}/api/sessions/{session_id}/engagement"
        headers = self._get_auth_headers()

        response = requests.post(url, json=events, headers=headers)
        response.raise_for_status()

        return response.json()

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

    def save_session(self, filepath: str = "student_session.json") -> None:
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

    def load_session(self, filepath: str = "student_session.json") -> bool:
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
    client = AuthClient()

    # Example registration
    try:
        result = client.register(
            email="student@example.com",
            password="SecurePass123",
            full_name="John Doe",
            student_number="STU2025001",
            major="Computer Science",
            year_of_study=3
        )
        print(f"Registered successfully: {result['user']['email']}")
    except requests.HTTPError as e:
        print(f"Registration failed: {e}")

    # Example login
    try:
        result = client.login("student@example.com", "SecurePass123")
        print(f"Logged in as: {result['user']['email']}")
        print(f"Role: {result['user']['role']}")
    except requests.HTTPError as e:
        print(f"Login failed: {e}")

    # Example joining session
    if client.is_authenticated():
        try:
            session_result = client.join_session("ABC123")
            print(f"Joined session: {session_result['lecture_title']}")
        except requests.HTTPError as e:
            print(f"Failed to join session: {e}")
