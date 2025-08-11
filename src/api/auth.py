"""
Authentication and security module for API
"""
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from src.utils.config import config_manager
from src.utils.logging import get_logger
from src.utils.exceptions import AuthenticationException

logger = get_logger()

# Load API configuration
api_config = config_manager.get_api_config()['api']
security_config = api_config['security']

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# JWT settings
SECRET_KEY = security_config['secret_key']
ALGORITHM = security_config['algorithm']
ACCESS_TOKEN_EXPIRE_MINUTES = security_config['access_token_expire_minutes']
API_KEYS = security_config['api_keys']

class Token(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Token data model"""
    username: Optional[str] = None

class User(BaseModel):
    """User model"""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> TokenData:
    """
    Verify and decode a JWT token
    
    Args:
        token: JWT token to verify
        
    Returns:
        Token data
        
    Raises:
        AuthenticationException: If token is invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise AuthenticationException("Invalid token")
        token_data = TokenData(username=username)
        return token_data
    except JWTError:
        raise AuthenticationException("Invalid token")

def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """
    Verify API key authentication
    
    Args:
        api_key: API key from header
        
    Returns:
        True if valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if api_key not in API_KEYS:
        logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    logger.info(f"Valid API key authentication: {api_key[:10]}...")
    return True

def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> User:
    """
    Verify JWT token authentication
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        User object
        
    Raises:
        HTTPException: If token is invalid
    """
    try:
        token_data = verify_token(credentials.credentials)
        user = User(username=token_data.username)
        logger.info(f"Valid JWT authentication for user: {user.username}")
        return user
    except AuthenticationException as e:
        logger.warning(f"Invalid JWT token: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Dependency for API key or JWT authentication
def get_current_user(
    api_key: Optional[str] = Security(api_key_header),
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> User:
    """
    Get current authenticated user (supports both API key and JWT)
    
    Args:
        api_key: API key from header
        credentials: JWT credentials
        
    Returns:
        User object
        
    Raises:
        HTTPException: If authentication fails
    """
    # Try API key authentication first
    if api_key and api_key in API_KEYS:
        logger.info(f"API key authentication successful: {api_key[:10]}...")
        return User(username="api_user")
    
    # Try JWT authentication
    if credentials:
        try:
            token_data = verify_token(credentials.credentials)
            user = User(username=token_data.username)
            logger.info(f"JWT authentication successful for user: {user.username}")
            return user
        except AuthenticationException:
            pass
    
    # If neither authentication method works
    logger.warning("Authentication failed - no valid API key or JWT token")
    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials. Provide either X-API-Key header or Bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )