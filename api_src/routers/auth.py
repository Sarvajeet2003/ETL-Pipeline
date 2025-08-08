"""Authentication endpoints."""

from datetime import timedelta
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
import structlog

from ..config import config
from ..auth import (
    authenticate_user, create_access_token, create_refresh_token,
    verify_token, get_user, Token, User
)
from ..models import LoginRequest, APIResponse, ErrorResponse

logger = structlog.get_logger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


@router.post(
    "/login",
    response_model=APIResponse,
    summary="User login",
    description="Authenticate user and return access token",
    responses={
        200: {"description": "Login successful"},
        401: {"description": "Invalid credentials"},
        501: {"description": "OAuth2 authentication not enabled"}
    }
)
async def login(
    request: Request,
    login_request: LoginRequest
) -> APIResponse:
    """User login endpoint.
    
    Args:
        request: FastAPI request object
        login_request: Login credentials
        
    Returns:
        API response with access token
    """
    # Check if OAuth2 is enabled
    if not config.security.oauth2.enabled:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="OAuth2 authentication is not enabled"
        )
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.info("Login attempt",
               request_id=request_id,
               username=login_request.username)
    
    try:
        # Authenticate user
        user = authenticate_user(login_request.username, login_request.password)
        
        if not user:
            logger.warning("Login failed - invalid credentials",
                          request_id=request_id,
                          username=login_request.username)
            
            return APIResponse(
                success=False,
                error=ErrorResponse(
                    error="AuthenticationError",
                    message="Invalid username or password",
                    request_id=request_id
                )
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=config.security.oauth2.access_token_expire_minutes)
        access_token = create_access_token(
            data={"sub": user.username, "scopes": user.scopes},
            expires_delta=access_token_expires
        )
        
        # Create refresh token
        refresh_token = create_refresh_token(
            data={"sub": user.username, "scopes": user.scopes}
        )
        
        # Create token response
        token_data = Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=config.security.oauth2.access_token_expire_minutes * 60,
            refresh_token=refresh_token
        )
        
        logger.info("Login successful",
                   request_id=request_id,
                   username=user.username)
        
        return APIResponse(
            success=True,
            data=token_data,
            metadata={
                "user": {
                    "username": user.username,
                    "full_name": user.full_name,
                    "scopes": user.scopes
                }
            }
        )
        
    except Exception as e:
        logger.error("Login error",
                    request_id=request_id,
                    username=login_request.username,
                    error=str(e),
                    error_type=type(e).__name__)
        
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="InternalError",
                message="An error occurred during login",
                request_id=request_id
            )
        )


@router.post(
    "/token",
    response_model=Token,
    summary="OAuth2 token endpoint",
    description="OAuth2 compatible token endpoint for authentication",
    responses={
        200: {"description": "Token issued successfully"},
        401: {"description": "Invalid credentials"},
        501: {"description": "OAuth2 authentication not enabled"}
    }
)
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Token:
    """OAuth2 compatible token endpoint.
    
    Args:
        request: FastAPI request object
        form_data: OAuth2 form data
        
    Returns:
        Access token
    """
    # Check if OAuth2 is enabled
    if not config.security.oauth2.enabled:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="OAuth2 authentication is not enabled"
        )
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.info("OAuth2 token request",
               request_id=request_id,
               username=form_data.username)
    
    # Authenticate user
    user = authenticate_user(form_data.username, form_data.password)
    
    if not user:
        logger.warning("OAuth2 token request failed - invalid credentials",
                      request_id=request_id,
                      username=form_data.username)
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=config.security.oauth2.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.scopes},
        expires_delta=access_token_expires
    )
    
    # Create refresh token
    refresh_token = create_refresh_token(
        data={"sub": user.username, "scopes": user.scopes}
    )
    
    logger.info("OAuth2 token issued successfully",
               request_id=request_id,
               username=user.username)
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=config.security.oauth2.access_token_expire_minutes * 60,
        refresh_token=refresh_token
    )


@router.post(
    "/refresh",
    response_model=APIResponse,
    summary="Refresh access token",
    description="Refresh an expired access token using a refresh token",
    responses={
        200: {"description": "Token refreshed successfully"},
        401: {"description": "Invalid refresh token"},
        501: {"description": "OAuth2 authentication not enabled"}
    }
)
async def refresh_token(
    request: Request,
    refresh_token: str
) -> APIResponse:
    """Refresh access token.
    
    Args:
        request: FastAPI request object
        refresh_token: Refresh token
        
    Returns:
        New access token
    """
    # Check if OAuth2 is enabled
    if not config.security.oauth2.enabled:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="OAuth2 authentication is not enabled"
        )
    
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.info("Token refresh request", request_id=request_id)
    
    try:
        # Verify refresh token
        token_data = verify_token(refresh_token)
        
        # Get user
        user = get_user(token_data.username)
        if not user:
            logger.warning("Token refresh failed - user not found",
                          request_id=request_id,
                          username=token_data.username)
            
            return APIResponse(
                success=False,
                error=ErrorResponse(
                    error="AuthenticationError",
                    message="User not found",
                    request_id=request_id
                )
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=config.security.oauth2.access_token_expire_minutes)
        new_access_token = create_access_token(
            data={"sub": user.username, "scopes": user.scopes},
            expires_delta=access_token_expires
        )
        
        # Create new refresh token
        new_refresh_token = create_refresh_token(
            data={"sub": user.username, "scopes": user.scopes}
        )
        
        # Create token response
        token_data = Token(
            access_token=new_access_token,
            token_type="bearer",
            expires_in=config.security.oauth2.access_token_expire_minutes * 60,
            refresh_token=new_refresh_token
        )
        
        logger.info("Token refreshed successfully",
                   request_id=request_id,
                   username=user.username)
        
        return APIResponse(
            success=True,
            data=token_data
        )
        
    except Exception as e:
        logger.warning("Token refresh failed",
                      request_id=request_id,
                      error=str(e))
        
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="AuthenticationError",
                message="Invalid refresh token",
                request_id=request_id
            )
        )


@router.get(
    "/me",
    response_model=APIResponse,
    summary="Get current user",
    description="Get information about the currently authenticated user",
    responses={
        200: {"description": "User information retrieved successfully"},
        401: {"description": "Authentication required"}
    }
)
async def get_current_user_info(
    request: Request,
    current_user: User = Depends(get_current_user)
) -> APIResponse:
    """Get current user information.
    
    Args:
        request: FastAPI request object
        current_user: Current authenticated user
        
    Returns:
        Current user information
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    if not current_user:
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="AuthenticationError",
                message="Authentication required",
                request_id=request_id
            )
        )
    
    logger.info("User info requested",
               request_id=request_id,
               username=current_user.username)
    
    return APIResponse(
        success=True,
        data={
            "username": current_user.username,
            "email": current_user.email,
            "full_name": current_user.full_name,
            "scopes": current_user.scopes,
            "disabled": current_user.disabled
        }
    )


@router.post(
    "/logout",
    response_model=APIResponse,
    summary="User logout",
    description="Logout the current user (token invalidation)",
    responses={
        200: {"description": "Logout successful"},
        401: {"description": "Authentication required"}
    }
)
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user)
) -> APIResponse:
    """User logout endpoint.
    
    Note: In a production system, you would typically maintain a blacklist
    of invalidated tokens or use short-lived tokens with refresh tokens.
    
    Args:
        request: FastAPI request object
        current_user: Current authenticated user
        
    Returns:
        Logout confirmation
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    if not current_user:
        return APIResponse(
            success=False,
            error=ErrorResponse(
                error="AuthenticationError",
                message="Authentication required",
                request_id=request_id
            )
        )
    
    logger.info("User logout",
               request_id=request_id,
               username=current_user.username)
    
    # In a real implementation, you would:
    # 1. Add the token to a blacklist
    # 2. Remove refresh tokens from storage
    # 3. Clear any user sessions
    
    return APIResponse(
        success=True,
        data={
            "message": "Logout successful",
            "username": current_user.username
        }
    )