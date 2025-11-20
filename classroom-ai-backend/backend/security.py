import os
from cryptography.fernet import Fernet
import logging
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from calendar import timegm
from zoneinfo import ZoneInfo

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-for-testing")  # Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.
    
    Args:
        data: Dictionary containing claims to encode in the token
        expires_delta: Optional timedelta for token expiration. If not provided,
                      defaults to ACCESS_TOKEN_EXPIRE_MINUTES
    
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    now = datetime.now(ZoneInfo("UTC"))
    
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
    # Convert to UTC timestamp for JWT
    to_encode.update({"exp": timegm(expire.utctimetuple())})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# It is critical that this environment variable is set.
# Generate one using: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")


class DataEncryptor:
    def __init__(self, key: str):
        if not key:
            logging.warning(
                "[WARNING] ENCRYPTION_KEY is not set. Data at rest will not be encrypted."
            )
            self.fernet = None
        else:
            try:
                self.fernet = Fernet(key.encode())
                logging.info(
                    "[OK] DataEncryptor initialized successfully. Sensitive data will be encrypted at rest."
                )
            except Exception as e:
                logging.error(
                    f"[ERROR] Failed to initialize DataEncryptor. Key might be invalid. Error: {e}"
                )
                self.fernet = None

    def encrypt(self, data: str) -> str:
        """Encrypts a string. Returns the original string if encryption is not available."""
        if not self.fernet or not data:
            return data
        try:
            return self.fernet.encrypt(data.encode()).decode()
        except Exception:
            # If encryption fails, return the original data to avoid breaking functionality
            return data

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypts a string. Returns the encrypted string if decryption is not available or fails."""
        if not self.fernet or not encrypted_data:
            return encrypted_data
        try:
            # The data from DB might be already decrypted or not encrypted at all
            return self.fernet.decrypt(encrypted_data.encode()).decode()
        except Exception:
            # If decryption fails, it's likely the data was not encrypted in the first place.
            return encrypted_data


# Singleton instance will be managed by the application lifecycle
# encryptor = DataEncryptor(ENCRYPTION_KEY)
