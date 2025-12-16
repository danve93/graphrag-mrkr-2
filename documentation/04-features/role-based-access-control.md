# Role-Based Access Control (RBAC)

## Overview

Amber implements a comprehensive Role-Based Access Control system to manage access to features and data based on user roles. This enables secure multi-user environments with distinct permission levels.

## User Roles

### Admin
- **Full System Access**: Complete access to all features and data
- **API Key Management**: Create, revoke, and manage API keys for external integrations
- **Shared Chat Management**: View and manage chats shared by external users
- **User Management**: Access to user administration features
- **All Standard Features**: Chat, Database, Graph, Categories, etc.

### External
- **Limited Chat Access**: Minimal chat-only interface
- **No Administrative Features**: Cannot access admin panels or settings
- **API Key Authentication**: Must authenticate using a valid API key
- **Chat Sharing**: Can share chat sessions with administrators

### User (Default)
- **Standard Access**: Access to core chat and knowledge base features
- **No Administrative Features**: Cannot access admin panels
- **Username-based Authentication**: Simple token-based authentication

## Authentication Flow

### Admin Authentication
1. Navigate to `/admin`
2. Enter admin password
3. Receive admin token with `role: 'admin'`
4. Token stored in localStorage as `chatUser`
5. Full dashboard access granted

### External User Authentication
1. Application provides API key via identify endpoint
2. Backend validates API key
3. External user created/retrieved with `role: 'external'`
4. Token issued with external role
5. Minimal chat interface displayed

### Standard User Authentication
1. Navigate to main page `/`
2. Username entered (or anonymous)
3. User created/retrieved from database
4. Token issued with user's stored role
5. **Important**: Role from database is preserved in token

## Backend Implementation

### User Service
Location: `api/services/user_service.py`

```python
def get_or_create_user(self, username: Optional[str] = None, role: str = "user"):
    """Get existing user OR create new one with specified role"""
    if username:
        result = graph_db.driver.execute_query(
            "MATCH (u:User {username: $username}) RETURN u",
            username=username
        )
        if result and result.records:
            return dict(result.records[0]["u"])  # Preserves existing role
    
    return self.create_user(username=username, role=role)
```

**Key Point**: Existing users retain their stored role (e.g., 'admin') when re-authenticating.

### Token Service
Location: `api/services/token_service.py`

Tokens include role metadata:
```python
token_data = token_service.create_token(
    user_id=user["id"],
    metadata={"role": user.get("role", "user")}
)
```

### Authentication Middleware
Location: `api/auth.py`

```python
async def require_admin(user_id: str = Depends(get_current_user)):
    """Dependency to enforce admin-only endpoints"""
    user = user_service.get_user(user_id)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user_id
```

## Frontend Implementation

### Main Page Router
Location: `frontend/src/app/page.tsx`

The main page implements role-based routing:

```typescript
// Check authentication on mount
useEffect(() => {
    if (!user) {
        window.location.href = '/admin'  // Redirect unauthenticated
    }
}, [user])

// Role-based UI rendering
if (user.role === 'external') {
    return <ExternalChatBubble />  // Minimal chat interface
}

if (user.role !== 'admin') {
    return <ExternalChatBubble />  // Default to restricted view
}

// Admin gets full dashboard
return <FullDashboard />
```

### User State Management
Location: `frontend/src/store/chatStore.ts`

```typescript
interface User {
    id: string
    token: string
    role: string  // 'admin' | 'external' | 'user'
}

// Stored in localStorage and Zustand state
const identifyUser = async (username?: string) => {
    const { user_id, token, role } = await api.identifyUser(username)
    const user = { id: user_id, token, role }
    set({ user })
    localStorage.setItem('chatUser', JSON.stringify(user))
}
```

### Conditional UI Elements
Location: `frontend/src/components/Sidebar/SidebarUser.tsx`

```typescript
{user?.role === 'admin' && (
    <div className="flex gap-2">
        <button onClick={() => setActiveView('adminApiKeys')}>
            API Keys
        </button>
        <button onClick={() => setActiveView('adminSharedChats')}>
            Shared Chats
        </button>
    </div>
)}
```

## Security Considerations

### Token Validation
- All API requests include `Authorization: Bearer <token>` header
- Backend validates token and extracts user_id
- Role checked from database for admin-only endpoints

### Role Persistence
- User roles stored in Neo4j User nodes
- Roles persist across sessions
- Token generation uses stored role, not default

### Admin Endpoints Protection
Protected endpoints using `require_admin` dependency:
- `/api/admin/api-keys` (GET, POST)
- `/api/admin/api-keys/{key_id}` (DELETE)
- `/api/admin/shared-chats` (GET)

## Common Issues & Solutions

### Issue: Admin user gets 403 errors
**Cause**: Old token still has `role: 'user'`  
**Solution**: 
1. Click "Logout" button in sidebar
2. Log back in via `/admin`
3. New token will have correct `role: 'admin'`

### Issue: User role not updating after backend change
**Cause**: Token issued before role was changed in database  
**Solution**: Clear localStorage and re-authenticate

### Issue: External user sees full dashboard
**Cause**: Frontend role check logic issue  
**Solution**: Verify `user.role` is correctly set in store after identify

## Related Documentation

- [API Key Management](./api-key-management.md)
- [External User Integration](./external-user-integration.md)
- [Authentication](../06-api-reference/authentication.md)
