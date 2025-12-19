# API Key Management

## Overview

API Key Management provides secure, database-backed authentication for administrators and external applications. API keys enable programmatic access to Amber's chat functionality without requiring user credentials.

> [!IMPORTANT]
> **Security Improvements in v2.1.0:**
> - **SHA-256 Hashing**: API keys are now hashed before storage (no plaintext keys in database)
> - **Tenant Isolation**: Each user can have only one active API key at a time
> - **No Static Tokens**: The insecure `JOBS_ADMIN_TOKEN` fallback has been removed
> - **Database-Backed Authentication**: All API keys must be generated via `scripts/generate_admin_key.py` or the admin UI

## Access

**Admin Only**: This feature is only accessible to users with the `admin` role.

**Location**: 
- Main UI: Chat Tuning → API Keys (sidebar navigation)
- Direct views: `adminApiKeys` active view
- Initial Setup: `scripts/generate_admin_key.py` for first admin key

## Features

### Create API Key
- Generate new API keys with custom names/descriptions
- Assign role to key: `external` (standard) or `admin`
- One-time display of full key (never shown again)
- Key format: `sk-` prefix with UUID-based secure tokens
- **SHA-256 hashed** before storage in Neo4j
- **Automatic revocation** of previous active key for the same user (tenant isolation)

### List API Keys
- View all active API keys
- Display masked key for security
- Show creation date and assigned role
- Color-coded role badges (external: green, admin: purple)

### Revoke API Key
- Immediately invalidate an API key
- Confirmation dialog to prevent accidental revocation
- External integrations using revoked keys will be denied access

## UI Integration

### Location in Application
The API Keys panel is integrated into the Chat Tuning section for consistency with other configuration panels.

**Navigation Path**: 
1. Open Chat Tuning panel from dock
2. Select "API Keys" from sidebar
3. Management interface displays in main content area

### Styling
- Matches application theme (CSS variables)
- Dark mode support
- Consistent with other Chat Tuning sections
- Uses `--bg-primary`, `--bg-secondary`, `--text-primary`, `--accent` variables

## Backend Implementation

### Service Layer
Location: `api/services/api_key_service.py`

```python
class ApiKeyService:
    def create_api_key(self, name: str, role: str, user_id: str, metadata: dict):
        """Create new API key and store SHA-256 hash in Neo4j
        
        Security improvements (v2.1.0):
        - Keys are hashed with SHA-256 before storage
        - Only one active key per user (tenant isolation)
        - Previous active key is automatically revoked
        """
        # Generate key with sk- prefix
        key = f"sk-{uuid.uuid4()}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Revoke previous active key for this user (tenant isolation)
        db.run("""
            MATCH (u:User {id: $user_id})-[:HAS_API_KEY]->(old:ApiKey {is_active: true})
            SET old.is_active = false
        """, user_id=user_id)
        
        # Store hashed key in Neo4j
        query = """
        CREATE (k:ApiKey {
            id: $key_id,
            key_hash: $key_hash,
            name: $name,
            role: $role,
            created_at: $timestamp,
            is_active: true,
            metadata: $metadata_str
        })
        WITH k
        MATCH (u:User {id: $user_id})
        CREATE (u)-[:HAS_API_KEY]->(k)
        RETURN k
        """
        
        return {"key": key, "id": key_id, ...}  # Full key returned ONCE
    
    def validate_api_key(self, key: str):
        """Validate key by comparing SHA-256 hashes"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        # Query Neo4j for active key with matching hash
        result = db.run("""
            MATCH (k:ApiKey {key_hash: $key_hash, is_active: true})
            RETURN k
        """, key_hash=key_hash)
        return result.single() if result else None
```

### API Endpoints
Location: `api/routers/admin.py`

#### List API Keys
```http
GET /api/admin/api-keys
Authorization: Bearer <admin_token>

Response:
[
    {
        "id": "uuid",
        "name": "PRODUCT NAME",
        "role": "external",
        "created_at": "2024-01-15T10:30:00Z",
        "key_masked": "abc...xyz"
    }
]
```

#### Create API Key
```http
POST /api/admin/api-keys
Authorization: Bearer <admin_token>
Content-Type: application/json

{
    "name": "Mobile App Integration",
    "role": "external",
    "metadata": {"source": "admin_ui"}
}

Response:
{
    "id": "uuid",
    "key": "sk-full-uuid-key-ONLY-SHOWN-ONCE",  # Note: sk- prefix added in v2.1.0
    "name": "Mobile App Integration",
    "role": "external",
    "created_at": "2024-01-15T10:30:00Z"
}

Note: If user already has an active API key, it will be automatically revoked (tenant isolation).
```

#### Revoke API Key
```http
DELETE /api/admin/api-keys/{key_id}
Authorization: Bearer <admin_token>

Response:
{
    "success": true,
    "message": "API key revoked"
}
```

### Database Schema (Neo4j)

```cypher
(:ApiKey {
    id: String,          // UUID
    key_hash: String,    // SHA-256 hash of actual key
    name: String,        // Human-readable description
    role: String,        // 'external' or 'admin'
    created_at: String,  // ISO timestamp
    is_active: Boolean,  // true/false
    metadata: String     // JSON metadata
})
```

**Relationships**:
```cypher
(:User)-[:HAS_API_KEY]->(:ApiKey)
```

**Indexes** (v2.1.0):
```cypher
// Fast lookup by hash for authentication
CREATE INDEX api_key_hash IF NOT EXISTS FOR (k:ApiKey) ON (k.key_hash);

// Enforce one active key per user
CREATE CONSTRAINT api_key_user_unique IF NOT EXISTS 
FOR ()-[r:HAS_API_KEY]->() REQUIRE r IS UNIQUE;
```

## Frontend Implementation

### Component Structure
Location: `frontend/src/components/Admin/AdminApiKeys.tsx`

Key features:
- Form for creating new keys
- Table displaying all keys
- One-time key display with copy-to-clipboard
- Revocation confirmation dialog

### State Management
```typescript
const [keys, setKeys] = useState<ApiKey[]>([])
const [createdKey, setCreatedKey] = useState<ApiKey | null>(null)

// Create key
const handleCreate = async (e: FormEvent) => {
    const result = await api.createApiKey({name, role, metadata})
    setCreatedKey(result)  // Display full key ONCE
    loadKeys()  // Refresh list
}

// Revoke key
const handleRevoke = async (id: string) => {
    if (confirm('Revoke this key?')) {
        await api.revokeApiKey(id)
        loadKeys()
    }
}
```

### Integration with Chat Tuning Panel
Location: `frontend/src/components/ChatTuning/ChatTuningPanel.tsx`

```typescript
// Conditional rendering based on active section
{activeCategory === 'api-keys' ? (
    <AdminApiKeys />
) : (
    // Other tuning parameters
)}
```

## External Application Integration

### Using an API Key

**Step 1: Obtain API Key**
Admin creates key via UI, copies full key value

**Step 2: Authentication**
External app sends identify request with API key:

```typescript
const response = await fetch('http://amber-host/api/users/identify', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        username: 'external-app-user',
        api_key: 'full-uuid-key'
    })
})

const {user_id, token, role} = await response.json()
// role will be 'external'
// Use token for subsequent requests
```

**Step 3: Chat Requests**
```typescript
await fetch('http://amber-host/api/chat/query', {
    method: 'POST',
    headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        message: 'User query here',
        session_id: sessionId
    })
})
```

## Security Best Practices

### Key Storage
- ✅ DO: Store keys in secure environment variables or secrets managers
- ❌ DON'T: Hardcode keys in source code
- ❌ DON'T: Commit keys to version control
- ❌ DON'T: Share keys via insecure channels (email, chat)

### Key Rotation
- Regularly create new keys and revoke old ones
- Track key usage (future enhancement)
- Use descriptive names to identify key purposes

### Monitoring
- Monitor for 403 errors (revoked/invalid keys)
- Track which integrations are using which keys
- Set up alerts for unusual activity patterns
- **v2.1.0**: Orphaned API keys are automatically cleaned up on startup

## Initial Setup (v2.1.0+)

### Generating the First Admin Key

> [!WARNING]
> The static `JOBS_ADMIN_TOKEN` environment variable has been removed for security. You must manually generate the first admin key.

**For Docker deployments:**
```bash
docker compose exec backend python scripts/generate_admin_key.py
```

**For local development:**
```bash
source .venv/bin/activate
python scripts/generate_admin_key.py
```

**Output:**
```
Generated admin API key: sk-1234567890abcdef...
This key has been saved to the database.
Please save this key securely - it will not be shown again.
```

Save this key immediately - it's required to log in to the admin panel. Additional keys can be generated from the UI after initial login.

## Troubleshooting

### Error: 403 Forbidden when accessing /api/admin/api-keys
**Cause**: User token does not have `admin` role  
**Solution**:
1. Logout from application
2. Login via `/admin` with admin password
3. Ensure token has `role: 'admin'`

### Error: Invalid API Key during external auth
**Cause**: Key was revoked or never existed  
**Solution**: 
- Admin: Create new key via UI
- External App: Update configuration with new key

### Issue: Created API key not working
**Cause**: External app not sending key correctly  
**Solution**: Verify request format matches example in "Using an API Key" section

## Related Documentation

- [Role-Based Access Control](./role-based-access-control.md)
- [External User Integration](./external-user-integration.md)
- [Authentication](../06-api-reference/authentication.md)
