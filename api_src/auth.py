"""Authentication and authorization module."""

import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import HTTPException, Security, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import structlog

from .config import config

logger = structlog.get_logger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
security_bearer = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name=config.security.api_key.header_name, auto_error=False)


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    scopes: List[str] = []
    exp: Optional[datetime] = None


class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    scopes: List[str] = []


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


# Mock user database (replace with real database in production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("admin123"),
        "disabled": False,
        "scopes": ["read", "write", "admin"]
    },
    "user": {
        "username": "user",
        "full_name": "Regular User", 
        "email": "user@example.com",
        "hashed_password": pwd_context.hash("user123"),
        "disabled": False,
        "scopes": ["read"]
    }
}


class AuthenticationError(HTTPException):
    """Custom authentication error."""
    
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(HTTPException):
    """Custom authorization error."""
    
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database.
    
    Args:
        username: Username to look up
        
    Returns:
        User object if found, None otherwise
    """
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user with username and password.
    
    Args:
        username: Username
        password: Password
        
    Returns:
        User object if authenticated, None otherwise
    """
    user = get_user(username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.
    
    Args:
        data: Data to encode in token
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config.security.oauth2.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode, 
        config.security.oauth2.secret_key, 
        algorithm=config.security.oauth2.algorithm
    )
    
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create a JWT refresh token.
    
    Args:
        data: Data to encode in token
        
    Returns:
        Encoded JWT refresh token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=config.security.oauth2.refresh_token_expire_days)
    to_encode.update({"exp": expire, "type": "refresh"})
    
    encoded_jwt = jwt.encode(
        to_encode,
        config.security.oauth2.secret_key,
        algorithm=config.security.oauth2.algorithm
    )
    
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """Verify and decode a JWT token.
    
    Args:
        token: JWT token to verify
        
    Returns:
        Token data
        
    Raises:
        AuthenticationError: If token is invalid
    """
    try:
        payload = jwt.decode(
            token, 
            config.security.oauth2.secret_key, 
            algorithms=[config.security.oauth2.algorithm]
        )
        
        username: str = payload.get("sub")
        if username is None:
            raise AuthenticationError("Invalid token: missing subject")
        
        scopes: List[str] = payload.get("scopes", [])
        exp_timestamp: int = payload.get("exp")
        exp = datetime.fromtimestamp(exp_timestamp) if exp_timestamp else None
        
        token_data = TokenData(username=username, scopes=scopes, exp=exp)
        return token_data
        
    except JWTError as e:
        logger.warning("JWT verification failed", error=str(e))
        raise AuthenticationError("Invalid token")


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """Verify API key authentication.
    
    Args:
        api_key: API key from header
        
    Returns:
        API key if valid, None otherwise
        
    Raises:
        AuthenticationError: If API key authentication is enabled but key is invalid
    """
    if not config.security.api_key.enabled:
        return None
    
    if not api_key:
        if config.security.oauth2.enabled:
            return None  # Fall back to OAuth2
        raise AuthenticationError("API key required")
    
    if api_key not in config.security.api_key.valid_keys:
        logger.warning("Invalid API key attempted", api_key=api_key[:8] + "...")
        raise AuthenticationError("Invalid API key")
    
    logger.info("API key authentication successful")
    return api_key


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    api_key: Optional[str] = Depends(verify_api_key)
) -> Optional[User]:
    """Get current authenticated user.
    
    Args:
        credentials: Bearer token credentials
        api_key: API key (if authenticated via API key)
        
    Returns:
        Current user if authenticated
        
    Raises:
        AuthenticationError: If authentication fails
    """
    # If API key authentication succeeded, return a generic API user
    if api_key:
        return User(
            username="api_user",
            full_name="API User",
            scopes=["read", "write"]  # Default scopes for API key users
        )
    
    # Try OAuth2 authentication
    if not config.security.oauth2.enabled:
        if config.security.api_key.enabled:
            raise AuthenticationError("Authentication required")
        return None  # No authentication required
    
    if not credentials:
        raise AuthenticationError("Bearer token required")
    
    # Verify token
    token_data = verify_token(credentials.credentials)
    
    # Get user from database
    user = get_user(token_data.username)
    if user is None:
        raise AuthenticationError("User not found")
    
    if user.disabled:
        raise AuthenticationError("User account disabled")
    
    # Convert to User model (without password hash)
    return User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        disabled=user.disabled,
        scopes=user.scopes
    )


def require_scopes(required_scopes: List[str]):
    """Dependency to require specific scopes.
    
    Args:
        required_scopes: List of required scopes
        
    Returns:
        Dependency function
    """
    def check_scopes(current_user: User = Depends(get_current_user)) -> User:
        if not current_user:
            raise AuthenticationError("Authentication required")
        
        # Check if user has required scopes
        user_scopes = set(current_user.scopes)
        required_scopes_set = set(required_scopes)
        
        if not required_scopes_set.issubset(user_scopes):
            missing_scopes = required_scopes_set - user_scopes
            logger.warning(
                "Insufficient permissions",
                user=current_user.username,
                required_scopes=required_scopes,
                user_scopes=current_user.scopes,
                missing_scopes=list(missing_scopes)
            )
            raise AuthorizationError(
                f"Insufficient permissions. Required scopes: {required_scopes}"
            )
        
        return current_user
    
    return check_scopes


# Common permission dependencies
require_read = require_scopes(["read"])
require_write = require_scopes(["write"])
require_admin = require_scopes(["admin"])