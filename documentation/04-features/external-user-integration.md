# External User Integration

## Overview

External User Integration provides a minimal, chat-only interface for non-administrative users and external applications. This enables embedded chat functionality for third-party applications while maintaining system security and isolation.

## User Experience

### Access Method
External users access Amber through:
1. **API Key Authentication**: Application authenticates user with API key
2. **Automatic Role Assignment**: User receives `external` role
3. **Minimal Interface**: Clean, chat-only bubble interface

### Testing Access
For development and testing purposes, the external interface can be forced without authentication:
- **URL**: `http://localhost:3000/?view=external`
- **Behavior**: Bypasses authentication checks and renders the `ExternalChatBubble` component directly.
- **Note**: Functional chat interactions still require a valid session, but UI elements (like the Share button) can be verified visually.

### Interface Characteristics
- **No Navigation**: No sidebar, no dock, no additional panels
- **Chat Only**: Simple message input and display
- **Responsive**: Mobile-friendly bubble design
- **Share Capability**: Can share conversations with administrators

## Component Implementation

### ExternalChatBubble
Location: `frontend/src/components/Chat/ExternalChatBubble.tsx`

A lightweight chat interface with minimal UI chrome.

**Features**:
- Simple header showing "Assistant"
- **Share Button**: One-click sharing with administrators
- Scrollable message history
- Text input with send button
- Streaming response support
- ReactMarkdown rendering for messages

**Key Differences from Full Chat**:
- No context document selection
- No retrieval mode toggles
- No source citations display
- No quality scoring UI
- No follow-up questions
- Simplified styling

### Code Structure

```typescript
export default function ExternalChatBubble() {
    const { messages, addMessage, updateLastMessage, user } = useChatStore()
    const [input, setInput] = useState('')
    const [sending, setSending] = useState(false)
    const [shared, setShared] = useState(false)

    const handleSend = async (e: FormEvent) => {
        // ...
    }

    const handleShare = async () => {
        // ... calls api.shareSession()
    }

    return (
        <div className="flex flex-col h-screen">
            {/* Header with Share Button */}
            <div className="header">
                <div>Assistant</div>
                <button onClick={handleShare}>
                    <ShareIcon />
                </button>
            </div>
            
            {/* ... */}
        </div>
    )
}
```

## Routing Logic

### Main Page Router
Location: `frontend/src/app/page.tsx`

The main page implements role-based routing to determine which interface to show:

```typescript
export default function Home() {
    const user = useChatStore((state) => state.user)
    const [isAuthChecking, setIsAuthChecking] = useState(true)

    // Check authentication
    useEffect(() => {
        const storedUser = localStorage.getItem('chatUser')
        if (!storedUser) {
            window.location.href = '/admin'  // Redirect to login
        }
        setIsAuthChecking(false)
    }, [user])

    if (isAuthChecking) {
        return <div>Loading Amber...</div>
    }

    if (!user) {
        return null  // Redirecting...
    }

    // Show minimal interface for external users
    if (user.role === 'external') {
        return <ExternalChatBubble />
    }

    // Default non-admin users to minimal interface
    if (user.role !== 'admin') {
        return <ExternalChatBubble />
    }

    // Admin users get full dashboard
    return <FullDashboardWithSidebar />
}
```

## Authentication Flow

### Step-by-Step Process

**1. External App Initializes**
```typescript
// In external application code
const API_KEY = process.env.AMBER_API_KEY
const username = `user-${userId}`  // Unique per user
```

**2. Identify User**
```typescript
const identifyResponse = await fetch('/api/users/identify', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        username: username,
        api_key: API_KEY
    })
})

const {user_id, token, role} = await identifyResponse.json()
// role === 'external'
```

**3. Store Token**
```typescript
localStorage.setItem('chatUser', JSON.stringify({
    id: user_id,
    token: token,
    role: role
}))
```

**4. Make Chat Requests**
```typescript
await fetch('/api/chat/query', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        message: userInput,
        session_id: currentSessionId
    })
})
```

## Backend Handling

### User Creation
Location: `api/services/user_service.py`

External users are created with specific constraints:

```python
def get_or_create_external_user(self, username: str, api_key_id: str):
    """
    Get or create an external user authenticated via API Key.
    Scoped to API Key for tenant isolation.
    """
    # Check for existing user with this key
    query = """
    MATCH (u:User {username: $username})-[:AUTHENTICATED_WITH]->(k:ApiKey {id: $api_key_id})
    RETURN u
    """
    
    result = graph_db.driver.execute_query(query, username=username, api_key_id=api_key_id)
    
    if result and result.records:
        return dict(result.records[0]["u"])
    
    # Create new external user
    return self.create_user(
        username=username,
        role="external",
        api_key_id=api_key_id,
        metadata={"source": "api_key_integration"}
    )
```

### Token Generation
```python
# In identify_user endpoint (api/routers/users.py)
if request.api_key:
    # Validate API key
    key_info = api_key_service.validate_api_key(request.api_key)
    
    # Get/create external user
    user = user_service.get_or_create_external_user(
        username=request.username,
        api_key_id=key_info["id"]
    )
    
    # Issue token with external role
    token_data = token_service.create_token(
        user_id=user["id"],
        metadata={"role": "external"}
    )
```

## Chat Sharing Feature

### Share Chat Session
External users can share their conversation sessions with administrators for review or assistance.

**User Interaction**:
1. External user creates a chat session by sending a message.
2. User clicks the **Share** button (<icon name="share" />) in the header.
   - **Default State**: Orange outline (stroke) to indicate available action.
   - **Shared State**: Filled orange icon to indicate active sharing.
3. The chat session is now visible to administrators in the "Shared Chats" panel.
4. **Unshare**: Clicking the button again toggles the state off, removing admin access.

**Backend Endpoints**:

*Share*:
```http
POST /api/history/{session_id}/share
Authorization: Bearer <external_user_token>

Response:
{
    "status": "success",
    "message": "Session {session_id} shared with admin"
}
```

*Unshare*:
```http
DELETE /api/history/{session_id}/share
Authorization: Bearer <external_user_token>

Response:
{
    "status": "success",
    "message": "Session {session_id} unshared"
}
```

**Implementation**:
```python
# In api/routers/history.py
@router.post("/{session_id}/share")
async def share_session(session_id: str, user_id: str = Depends(get_current_user)):
    """Allow external users to share sessions with admins"""
    success = await chat_history_service.share_session(session_id, target_role="admin")
    return {"status": "success", ...}

@router.delete("/{session_id}/share")
async def unshare_session(session_id: str, user_id: str = Depends(get_current_user)):
    """Unshare a session"""
    success = await chat_history_service.unshare_session(session_id)
    return {"status": "success", ...}
```

### Admin View of Shared Chats
Administrators can view all shared conversations:

**UI Access**: Main view → Shared Chats (admin only)  
**Backend Endpoint**: `GET /api/admin/shared-chats`

## Security & Isolation

### Tenant Isolation
Each API key represents a separate tenant. External users are scoped to their authenticating API key:

```cypher
// Database relationship
(ExternalUser)-[:AUTHENTICATED_WITH]->(ApiKey)

// Users from different API keys are isolated
// Even with same username
```

### Access Restrictions
External users **cannot** access:
- Admin panels (API Keys, Shared Chats)
- Database management
- Graph visualization
- System settings
- Other users' data

External users **can** access:
- Chat functionality
- Their own chat history
- Session sharing

## Embedding in External Applications

### iframe Embed
```html
<iframe 
    src="https://amber-host/?external=true" 
    width="400" 
    height="600"
    style="border: 1px solid #ccc; border-radius: 8px;"
></iframe>
```

### JavaScript Widget
```html
<div id="amber-chat"></div>
<script src="https://amber-host/embed.js"></script>
<script>
    AmberChat.init({
        container: '#amber-chat',
        apiKey: 'your-api-key',
        username: 'user-123'
    })
</script>
```

## Troubleshooting

### Issue: External user sees full dashboard
**Cause**: Role not correctly set to 'external'  
**Solution**: Verify API key authentication flow, check token role

### Issue: External user cannot access chat
**Cause**: Authentication failed or token expired  
**Solution**: Re-authenticate with valid API key

### Issue: Messages not sending
**Cause**: Network error or invalid session  
**Solution**: Check browser console, verify backend is accessible

## Future Enhancements

- **Customization**: Allow API keys to specify UI theme/branding
- **Rate Limiting**: Per-key rate limits for API usage
- **Analytics**: Track usage metrics per API key/tenant
- **Webhooks**: Notify external apps of chat events
- **Custom Prompts**: Per-tenant system prompts

## Related Documentation

- [API Key Management](./api-key-management.md)
- [Role-Based Access Control](./role-based-access-control.md)
- [Chat Interface](../03-components/chat-interface.md)
