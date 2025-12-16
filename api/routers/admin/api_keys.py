from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel

from api import auth
from api.services.api_key_service import api_key_service

router = APIRouter()

class CreateApiKeyRequest(BaseModel):
    name: str
    role: str = "external"
    metadata: Optional[dict] = None

class ApiKeyResponse(BaseModel):
    id: str
    name: str
    role: str
    created_at: str
    key_masked: Optional[str] = None
    key: Optional[str] = None # Only returned on creation

@router.get("", response_model=List[ApiKeyResponse])
async def list_api_keys(_token: str = Depends(auth.require_admin)):
    """
    List all API keys.
    Requires Admin privileges.
    """
    return api_key_service.list_api_keys()

@router.post("", response_model=ApiKeyResponse)
async def create_api_key(
    request: CreateApiKeyRequest,
    _token: str = Depends(auth.require_admin)
):
    """
    Create a new API key.
    Requires Admin privileges.
    """
    return api_key_service.create_api_key(
        name=request.name,
        role=request.role,
        metadata=request.metadata
    )

@router.delete("/{key_id}")
async def revoke_api_key(
    key_id: str,
    _token: str = Depends(auth.require_admin)
):
    """
    Revoke an API key.
    Requires Admin privileges.
    """
    success = api_key_service.revoke_api_key(key_id)
    if not success:
        raise HTTPException(status_code=404, detail="API Key not found")
    return {"status": "success"}
