"""
Authentication Module for EduVision
Handles JWT tokens, password hashing, and authentication logic
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from dotenv import load_dotenv
import secrets
import hashlib

# Load environment variables
load_dotenv(dotenv_path="../database/.env")

# ============================================================================
# Configuration
# ============================================================================

# JWT Settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing
BCRYPT_ROUNDS = int(os.getenv("BCRYPT_ROUNDS", "12"))
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ============================================================================
# Password Hashing
# ============================================================================

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt

    Args:
        password: Plain text password

    Returns:
        Hashed password string
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash

    Args:
        plain_password: Plain text password to verify
        hashed_password: Stored hash to verify against

    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


# ============================================================================
# JWT Token Generation
# ============================================================================

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token

    Args:
        data: Data to encode in token (should include user_id, email, role)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string

    Example:
        token = create_access_token(
            data={"user_id": str(user.user_id), "email": user.email, "role": user.role}
        )
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT refresh token

    Args:
        data: Data to encode in token (should include user_id)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT refresh token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token

    Args:
        token: JWT token string

    Returns:
        Decoded token payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def verify_token(token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
    """
    Verify a JWT token and check its type

    Args:
        token: JWT token string
        token_type: Expected token type ('access' or 'refresh')

    Returns:
        Token payload if valid and correct type, None otherwise
    """
    payload = decode_token(token)

    if payload is None:
        return None

    # Check token type
    if payload.get("type") != token_type:
        return None

    # Check expiration
    exp = payload.get("exp")
    if exp is None or datetime.utcnow() > datetime.fromtimestamp(exp):
        return None

    return payload


# ============================================================================
# Refresh Token Hashing
# ============================================================================

def hash_refresh_token(token: str) -> str:
    """
    Create a SHA256 hash of a refresh token for storage

    Args:
        token: Refresh token string

    Returns:
        SHA256 hash of the token
    """
    return hashlib.sha256(token.encode()).hexdigest()


# ============================================================================
# Token Pair Generation
# ============================================================================

def create_token_pair(user_id: str, email: str, role: str) -> Dict[str, str]:
    """
    Create both access and refresh tokens for a user

    Args:
        user_id: User's UUID as string
        email: User's email
        role: User's role (admin/teacher/student)

    Returns:
        Dictionary with access_token and refresh_token

    Example:
        tokens = create_token_pair(
            user_id=str(user.user_id),
            email=user.email,
            role=user.role.value
        )
    """
    # Data to encode in access token
    access_data = {
        "user_id": user_id,
        "email": email,
        "role": role
    }

    # Data to encode in refresh token (minimal)
    refresh_data = {
        "user_id": user_id
    }

    access_token = create_access_token(data=access_data)
    refresh_token = create_refresh_token(data=refresh_data)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
    }


# ============================================================================
# Token Validation for API Routes
# ============================================================================

def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Extract user information from access token

    Args:
        token: JWT access token

    Returns:
        User data dict if valid, None otherwise

    Returns dict with:
        - user_id: UUID string
        - email: User email
        - role: User role
    """
    payload = verify_token(token, token_type="access")

    if payload is None:
        return None

    return {
        "user_id": payload.get("user_id"),
        "email": payload.get("email"),
        "role": payload.get("role")
    }


# ============================================================================
# Password Strength Validation
# ============================================================================

def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Validate password meets security requirements

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, error_message)

    Requirements:
        - At least 8 characters
        - Contains uppercase letter
        - Contains lowercase letter
        - Contains digit
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"

    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"

    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"

    return True, ""


# ============================================================================
# Email Validation
# ============================================================================

def validate_email(email: str) -> bool:
    """
    Basic email validation

    Args:
        email: Email address to validate

    Returns:
        True if valid email format, False otherwise
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


# ============================================================================
# Testing & Utilities
# ============================================================================

def generate_session_code(length: int = 8) -> str:
    """
    Generate a random session code for lecture sessions

    Args:
        length: Length of code (default 8)

    Returns:
        Random uppercase alphanumeric code

    Example: 'A7K9M2P4'
    """
    import random
    import string
    # Use only uppercase letters and digits, exclude similar chars (O/0, I/1, etc.)
    chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
    return ''.join(random.choice(chars) for _ in range(length))


def get_password_hash_info(hashed_password: str) -> Dict[str, Any]:
    """
    Get information about a hashed password

    Args:
        hashed_password: Bcrypt hashed password

    Returns:
        Dictionary with hash information
    """
    try:
        # Bcrypt hash format: $2b$rounds$salt+hash
        parts = hashed_password.split('$')
        if len(parts) >= 4:
            return {
                "algorithm": parts[1],  # e.g., '2b'
                "rounds": int(parts[2]),  # cost factor
                "valid_format": True
            }
    except:
        pass

    return {"valid_format": False}


# ============================================================================
# Main Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EduVision Authentication Module Test")
    print("=" * 70)

    # Test password hashing
    print("\n1. Password Hashing Test")
    test_password = "TestPassword123"
    hashed = hash_password(test_password)
    print(f"   Original: {test_password}")
    print(f"   Hashed: {hashed[:60]}...")
    print(f"   Verify (correct): {verify_password(test_password, hashed)}")
    print(f"   Verify (wrong): {verify_password('WrongPassword', hashed)}")

    # Test password strength validation
    print("\n2. Password Strength Validation")
    test_passwords = [
        ("weak", False),
        ("StrongPass123", True),
        ("noupperca5e", False),
        ("NOLOWERCASE5", False),
        ("NoDigits", False)
    ]
    for pwd, expected in test_passwords:
        valid, msg = validate_password_strength(pwd)
        status = "[OK]" if valid == expected else "[FAIL]"
        print(f"   {status} '{pwd}': {valid} - {msg if not valid else 'Valid'}")

    # Test JWT token generation
    print("\n3. JWT Token Generation")
    tokens = create_token_pair(
        user_id="123e4567-e89b-12d3-a456-426614174000",
        email="test@eduvision.com",
        role="student"
    )
    print(f"   Access Token: {tokens['access_token'][:50]}...")
    print(f"   Refresh Token: {tokens['refresh_token'][:50]}...")
    print(f"   Token Type: {tokens['token_type']}")
    print(f"   Expires In: {tokens['expires_in']} seconds")

    # Test token verification
    print("\n4. Token Verification")
    user_data = get_user_from_token(tokens['access_token'])
    if user_data:
        print(f"   [OK] Token valid")
        print(f"   User ID: {user_data['user_id']}")
        print(f"   Email: {user_data['email']}")
        print(f"   Role: {user_data['role']}")
    else:
        print(f"   [FAIL] Token invalid")

    # Test invalid token
    print("\n5. Invalid Token Test")
    invalid_user = get_user_from_token("invalid.token.here")
    print(f"   [{'OK' if invalid_user is None else 'FAIL'}] Invalid token rejected: {invalid_user is None}")

    # Test session code generation
    print("\n6. Session Code Generation")
    for i in range(5):
        code = generate_session_code()
        print(f"   Session Code {i+1}: {code}")

    # Test email validation
    print("\n7. Email Validation")
    test_emails = [
        ("valid@example.com", True),
        ("user.name+tag@example.co.uk", True),
        ("invalid.email", False),
        ("@example.com", False),
        ("user@", False)
    ]
    for email, expected in test_emails:
        valid = validate_email(email)
        status = "[OK]" if valid == expected else "[FAIL]"
        print(f"   {status} '{email}': {valid}")

    print("\n" + "=" * 70)
    print("Authentication module test complete!")
    print("=" * 70)
