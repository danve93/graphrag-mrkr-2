# Performance Optimization Plan

## Executive Summary

This document outlines a comprehensive performance optimization strategy for the Amber document intelligence platform. The optimization focuses on reducing API response sizes and implementing progressive loading patterns to improve user experience, particularly for large documents with thousands of entities and chunk relationships.

**Key Metrics:**
- Current similarities API: 15MB response, 14-second load time
- Target similarities API: 50KB per page, <1-second load time
- Overall improvement: 150x reduction in initial data transfer

## Problem Analysis

### 1. Similarities Endpoint Performance Issue

**Current Implementation:**
```python
# api/routers/documents.py - lines 96-111
query = """
    MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
    WHERE c1.id IN $chunk_ids AND c2.id IN $chunk_ids AND c1.id < c2.id
    RETURN 
        c1.id AS chunk1_id,
        c1.content AS chunk1_text,  # ← Full content (500-1500 chars)
        c2.id AS chunk2_id,
        c2.content AS chunk2_text,  # ← Full content (500-1500 chars)
        coalesce(sim.score, 0) AS score
    ORDER BY score DESC
"""
```

**Measured Performance:**
- Query execution: Fast (Neo4j handles graph traversal efficiently)
- Data serialization: 14 seconds
- Response size: 15 MB
- Similarity pairs: 8,709 relationships

**Root Cause:**
Each similarity relationship returns both chunks' full content. For a document with 2,453 chunks:
- Potential relationships: (2,453 × 2,452) / 2 = 3,008,478 possible pairs
- Actual relationships: 8,709 pairs stored
- Each pair includes 2× full chunk text (avg 1KB each)
- Total data: 8,709 × 2KB = ~17MB of duplicated text

**Why This Happens:**
The query was designed for completeness - returning everything needed to display similarity information. However, this violates the principle of data normalization and creates massive over-fetching.

### 2. Document Details API Performance

**Current State:**
- Response size: 2.7 MB
- Load time: 0.55s (acceptable)
- Contains: 6,679 entities with metadata

**Analysis:**
This endpoint's performance is acceptable. The issue is not the size per se, but that it's loaded eagerly even when users may not need all entities immediately.

**Structure:**
```json
{
  "entities": [
    {
      "type": "ACCOUNT",
      "text": "150 USERS",
      "community_id": 30,
      "level": 0,
      "count": 2,
      "positions": [738, 609]  # ← Duplicated across response
    }
    // ... 6,678 more entities
  ]
}
```

### 3. Frontend Loading Pattern Issues

**Current Behavior:**
```tsx
// frontend/src/components/Document/DocumentView.tsx
<CommunitiesSection documentId={documentData.id} />
<ChunkSimilaritiesSection documentId={documentData.id} />
```

Both components load immediately when the document page renders:

```tsx
// CommunitiesSection.tsx - lines 26-42
useEffect(() => {
  const loadCommunities = async () => {
    const docData = await fetch(`/api/documents/${documentId}`)  // 2.7MB
    // Extract communities from entities
  }
  loadCommunities()
}, [documentId])
```

```tsx
// ChunkSimilaritiesSection.tsx - lines 25-40
useEffect(() => {
  const loadSimilarities = async () => {
    const data = await api.getDocumentChunkSimilarities(documentId)  // 15MB
    setSimilarities(data.similarities)
  }
  loadSimilarities()
}, [documentId])
```

**Problems:**
1. No lazy loading - all data fetched immediately
2. No pagination - all 8,709 similarities loaded at once
3. No caching - data refetched on every navigation
4. No progressive enhancement - user sees nothing until all data loads

## Solution: Optimize Existing Architecture

### Why This Approach?

**Rejected Alternative: Add PostgreSQL**

We explicitly reject moving data to PostgreSQL because:

1. **The problem is API design, not database performance**
   - Neo4j query executes in <100ms
   - The bottleneck is serializing 15MB of text
   - Postgres wouldn't solve this

2. **Neo4j already stores this data optimally**
   - Graph relationships are efficient
   - Chunk content is stored once
   - Only the API duplicates it

3. **Postgres would add complexity without benefit**
   - Dual-write to maintain consistency
   - Sync logic between databases
   - More failure modes
   - Duplicate storage costs

4. **Real-world evidence**
   - Document details API (Neo4j): 0.55s for 2.7MB
   - Similarities API (Neo4j): 14s for 15MB
   - The 25x difference proves it's about data shape, not database

**Chosen Approach: API Optimization + Frontend Patterns**

This approach:
- Keeps single source of truth (Neo4j)
- Fixes the actual problem (over-fetching)
- Requires no infrastructure changes
- Provides immediate benefits

### Implementation Details

## Phase 1: Backend API Optimization (High Impact)

### 1.1 Paginate Similarities Endpoint

**Objective:** Reduce similarities response from 15MB to ~50KB per page

**Implementation:**

```python
# api/routers/documents.py - Replace get_document_chunk_similarities

@router.get("/{document_id}/similarities")
async def get_document_chunk_similarities(
    document_id: str,
    limit: int = Query(default=50, ge=1, le=200, description="Number of similarities per page"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")
):
    """
    Get chunk-to-chunk similarity relationships for a document.
    
    Returns only IDs and scores for efficient transfer.
    Use /chunks/{chunk_id} to fetch full chunk content on demand.
    """
    try:
        # Verify document exists
        try:
            graph_db.get_document_details(document_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Document not found")

        # Get all chunks for this document
        chunks = graph_db.get_document_chunks(document_id)
        if not chunks:
            return {
                "document_id": document_id,
                "total": 0,
                "limit": limit,
                "offset": offset,
                "similarities": []
            }

        chunk_ids = [chunk.get("chunk_id") or chunk.get("id") for chunk in chunks]
        
        # Optimized query - only IDs and scores
        query = """
            MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
            WHERE c1.id IN $chunk_ids 
              AND c2.id IN $chunk_ids 
              AND c1.id < c2.id
              AND coalesce(sim.score, 0) >= $min_score
            WITH c1.id AS chunk1_id,
                 c2.id AS chunk2_id,
                 coalesce(sim.score, 0) AS score
            ORDER BY score DESC
            SKIP $offset
            LIMIT $limit
            RETURN chunk1_id, chunk2_id, score
        """
        
        # Get total count for pagination metadata
        count_query = """
            MATCH (c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
            WHERE c1.id IN $chunk_ids 
              AND c2.id IN $chunk_ids 
              AND c1.id < c2.id
              AND coalesce(sim.score, 0) >= $min_score
            RETURN count(*) as total
        """
        
        with graph_db.driver.session() as session:
            total_result = session.run(count_query, 
                chunk_ids=chunk_ids, 
                min_score=min_score
            ).single()
            total = total_result["total"] if total_result else 0
            
            results = session.run(query, 
                chunk_ids=chunk_ids,
                offset=offset,
                limit=limit,
                min_score=min_score
            ).data()
        
        return {
            "document_id": document_id,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total,
            "similarities": results
        }
    except HTTPException:
        raise
    except Exception as exc:
        if isinstance(exc, ServiceUnavailable):
            raise
        logger.error("Failed to get chunk similarities for %s: %s", document_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve chunk similarities"
        )
```

**Response Comparison:**

Before (15 MB):
```json
{
  "similarities": [
    {
      "chunk1_id": "abc123",
      "chunk1_text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit... (1200 chars)",
      "chunk2_id": "def456",
      "chunk2_text": "Sed do eiusmod tempor incididunt ut labore et dolore... (1100 chars)",
      "score": 0.95
    }
    // × 8,709 items
  ]
}
```

After (50 KB):
```json
{
  "document_id": "944b7e4c...",
  "total": 8709,
  "limit": 50,
  "offset": 0,
  "has_more": true,
  "similarities": [
    {
      "chunk1_id": "abc123",
      "chunk2_id": "def456",
      "score": 0.95
    }
    // × 50 items
  ]
}
```

**Impact:**
- Response size: 15 MB → 50 KB (300x reduction)
- Load time: 14s → <1s
- Network bandwidth: 99.7% reduction

### 1.2 Add Chunk Details Endpoint

**Objective:** Fetch chunk content on-demand when user expands similarity

**Implementation:**

```python
# api/routers/documents.py - Add new endpoint

@router.get("/chunks/{chunk_id}")
async def get_chunk_details(chunk_id: str):
    """
    Get full details for a specific chunk.
    Used for on-demand loading of chunk content in similarity views.
    """
    try:
        with graph_db.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk {id: $chunk_id})
                OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(d:Document)
                RETURN 
                    c.id AS id,
                    c.content AS content,
                    c.chunk_index AS index,
                    coalesce(c.offset, 0) AS offset,
                    d.id AS document_id,
                    d.filename AS document_name
                """,
                chunk_id=chunk_id
            ).single()
            
            if not result:
                raise HTTPException(status_code=404, detail="Chunk not found")
            
            return {
                "id": result["id"],
                "content": result["content"],
                "index": result["index"],
                "offset": result["offset"],
                "document_id": result["document_id"],
                "document_name": result["document_name"]
            }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get chunk details for %s: %s", chunk_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve chunk details")
```

### 1.3 Add Document Summary Endpoint

**Objective:** Fast initial load with metadata only

**Implementation:**

```python
# api/routers/documents.py - Add new endpoint

@router.get("/{document_id}/summary")
async def get_document_summary(document_id: str):
    """
    Get lightweight document overview without full entity/chunk data.
    Used for fast initial page load and navigation previews.
    """
    try:
        with graph_db.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {id: $document_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
                WITH d, 
                     count(DISTINCT c) AS chunk_count,
                     count(DISTINCT e) AS entity_count,
                     count(DISTINCT e.community_id) AS community_count
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c2:Chunk)-[s:SIMILAR_TO]->()
                RETURN 
                    d.id AS id,
                    d.filename AS filename,
                    d.original_filename AS original_filename,
                    d.mime_type AS mime_type,
                    d.size_bytes AS size_bytes,
                    d.created_at AS created_at,
                    d.link AS link,
                    d.uploader AS uploader,
                    chunk_count,
                    entity_count,
                    community_count,
                    count(DISTINCT s) AS similarity_count
                """,
                document_id=document_id
            ).single()
            
            if not result:
                raise HTTPException(status_code=404, detail="Document not found")
            
            return {
                "id": result["id"],
                "filename": result["filename"],
                "original_filename": result["original_filename"],
                "mime_type": result["mime_type"],
                "size_bytes": result["size_bytes"],
                "created_at": result["created_at"],
                "link": result["link"],
                "uploader": result["uploader"],
                "stats": {
                    "chunks": result["chunk_count"],
                    "entities": result["entity_count"],
                    "communities": result["community_count"],
                    "similarities": result["similarity_count"]
                }
            }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get document summary for %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve document summary")
```

**Impact:**
- Response size: 2.7 MB → 2 KB (1350x reduction)
- Load time: 0.55s → <50ms
- Time to first paint: Immediate

### 1.4 Paginate Entities Endpoint

**Objective:** Load entities progressively, filtered by community

**Implementation:**

```python
# api/routers/documents.py - Add new endpoint

@router.get("/{document_id}/entities")
async def get_document_entities(
    document_id: str,
    community_id: Optional[int] = Query(default=None, description="Filter by community"),
    entity_type: Optional[str] = Query(default=None, description="Filter by entity type"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0)
):
    """
    Get entities for a document with pagination and filtering.
    Used for progressive loading of entity lists.
    """
    try:
        # Build dynamic WHERE clause
        filters = []
        params = {
            "doc_id": document_id,
            "limit": limit,
            "offset": offset
        }
        
        if community_id is not None:
            filters.append("e.community_id = $community_id")
            params["community_id"] = community_id
            
        if entity_type:
            filters.append("e.type = $entity_type")
            params["entity_type"] = entity_type
        
        where_clause = f"AND {' AND '.join(filters)}" if filters else ""
        
        with graph_db.driver.session() as session:
            # Get total count
            count_result = session.run(
                f"""
                MATCH (d:Document {{id: $doc_id}})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                WHERE TRUE {where_clause}
                RETURN count(DISTINCT e) as total
                """,
                **params
            ).single()
            total = count_result["total"] if count_result else 0
            
            # Get paginated entities
            entity_records = session.run(
                f"""
                MATCH (d:Document {{id: $doc_id}})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                WHERE TRUE {where_clause}
                WITH e, collect(DISTINCT c.chunk_index) as positions
                RETURN 
                    e.type as type,
                    e.name as text,
                    e.community_id as community_id,
                    e.level as level,
                    size(positions) as count,
                    positions
                ORDER BY type ASC, text ASC
                SKIP $offset
                LIMIT $limit
                """,
                **params
            )
            
            entities = [
                {
                    "type": record["type"],
                    "text": record["text"],
                    "community_id": record["community_id"],
                    "level": record["level"],
                    "count": record["count"],
                    "positions": [pos for pos in (record["positions"] or []) if pos is not None]
                }
                for record in entity_records
            ]
        
        return {
            "document_id": document_id,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total,
            "entities": entities
        }
    except Exception as exc:
        logger.error("Failed to get entities for %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve entities")
```

## Phase 2: Frontend Progressive Loading (Medium Impact)

### 2.1 Update API Client

**Implementation:**

```typescript
// frontend/src/lib/api.ts - Add new methods

export const api = {
  // Existing methods...
  
  async getDocumentSummary(documentId: string): Promise<DocumentSummary> {
    const response = await fetch(`${API_URL}/api/documents/${documentId}/summary`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },
  
  async getDocumentEntitiesPaginated(
    documentId: string,
    options?: {
      communityId?: number
      entityType?: string
      limit?: number
      offset?: number
    }
  ): Promise<PaginatedEntitiesResponse> {
    const query = new URLSearchParams()
    if (options?.communityId !== undefined) {
      query.append('community_id', String(options.communityId))
    }
    if (options?.entityType) {
      query.append('entity_type', options.entityType)
    }
    if (options?.limit) {
      query.append('limit', String(options.limit))
    }
    if (options?.offset) {
      query.append('offset', String(options.offset))
    }
    
    const queryString = query.toString()
    const response = await fetch(
      `${API_URL}/api/documents/${documentId}/entities${queryString ? `?${queryString}` : ''}`
    )
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },
  
  async getDocumentSimilaritiesPaginated(
    documentId: string,
    options?: {
      limit?: number
      offset?: number
      minScore?: number
    }
  ): Promise<PaginatedSimilaritiesResponse> {
    const query = new URLSearchParams()
    if (options?.limit) {
      query.append('limit', String(options.limit))
    }
    if (options?.offset) {
      query.append('offset', String(options.offset))
    }
    if (options?.minScore !== undefined) {
      query.append('min_score', String(options.minScore))
    }
    
    const queryString = query.toString()
    const response = await fetch(
      `${API_URL}/api/documents/${documentId}/similarities${queryString ? `?${queryString}` : ''}`
    )
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },
  
  async getChunkDetails(chunkId: string): Promise<ChunkDetails> {
    const response = await fetch(`${API_URL}/api/documents/chunks/${chunkId}`)
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  }
}
```

### 2.2 Update DocumentView Component

**Objective:** Load summary first, defer heavy data

**Implementation:**

```tsx
// frontend/src/components/Document/DocumentView.tsx

export default function DocumentView() {
  const [summary, setSummary] = useState<DocumentSummary | null>(null)
  const [activeTab, setActiveTab] = useState('overview')
  
  useEffect(() => {
    const loadSummary = async () => {
      try {
        setLoading(true)
        // Load lightweight summary first
        const summaryData = await api.getDocumentSummary(documentId)
        setSummary(summaryData)
        setLoading(false)
      } catch (err) {
        console.error('Failed to load document summary:', err)
        setError('Failed to load document')
        setLoading(false)
      }
    }
    
    loadSummary()
  }, [documentId])
  
  return (
    <div>
      {loading && <Loader />}
      {summary && (
        <>
          <DocumentHeader summary={summary} />
          
          <Tabs value={activeTab} onChange={setActiveTab}>
            <TabList>
              <Tab value="overview">Overview</Tab>
              <Tab value="entities">
                Entities ({summary.stats.entities})
              </Tab>
              <Tab value="communities">
                Communities ({summary.stats.communities})
              </Tab>
              <Tab value="similarities">
                Similarities ({summary.stats.similarities})
              </Tab>
            </TabList>
            
            {/* Lazy load tab content only when active */}
            <TabPanel value="overview">
              <OverviewSection summary={summary} />
            </TabPanel>
            
            <TabPanel value="entities">
              {activeTab === 'entities' && (
                <EntitiesSection documentId={summary.id} />
              )}
            </TabPanel>
            
            <TabPanel value="communities">
              {activeTab === 'communities' && (
                <CommunitiesSection documentId={summary.id} />
              )}
            </TabPanel>
            
            <TabPanel value="similarities">
              {activeTab === 'similarities' && (
                <ChunkSimilaritiesSection documentId={summary.id} />
              )}
            </TabPanel>
          </Tabs>
        </>
      )}
    </div>
  )
}
```

### 2.3 Update ChunkSimilaritiesSection

**Objective:** Paginated loading with on-demand chunk details

**Implementation:**

```tsx
// frontend/src/components/Document/ChunkSimilaritiesSection.tsx

export default function ChunkSimilaritiesSection({ documentId }: Props) {
  const [page, setPage] = useState(0)
  const [similarities, setSimilarities] = useState<PaginatedSimilaritiesResponse | null>(null)
  const [expandedChunks, setExpandedChunks] = useState<Record<string, ChunkDetails>>({})
  const [loading, setLoading] = useState(false)
  const pageSize = 50
  
  useEffect(() => {
    const loadPage = async () => {
      try {
        setLoading(true)
        const data = await api.getDocumentSimilaritiesPaginated(documentId, {
          limit: pageSize,
          offset: page * pageSize,
          minScore: 0.5  // Only show meaningful similarities
        })
        setSimilarities(data)
      } catch (err) {
        console.error('Failed to load similarities:', err)
      } finally {
        setLoading(false)
      }
    }
    
    loadPage()
  }, [documentId, page])
  
  const loadChunkDetails = async (chunkId: string) => {
    if (expandedChunks[chunkId]) return  // Already loaded
    
    try {
      const details = await api.getChunkDetails(chunkId)
      setExpandedChunks(prev => ({ ...prev, [chunkId]: details }))
    } catch (err) {
      console.error('Failed to load chunk details:', err)
    }
  }
  
  return (
    <div>
      <h3>Chunk Similarities</h3>
      
      {loading && <Loader />}
      
      {similarities && (
        <>
          <div className="text-sm text-gray-600 mb-4">
            Showing {page * pageSize + 1}-{Math.min((page + 1) * pageSize, similarities.total)} of {similarities.total} similarities
          </div>
          
          <div className="space-y-2">
            {similarities.similarities.map((sim, idx) => (
              <SimilarityCard
                key={idx}
                similarity={sim}
                chunk1={expandedChunks[sim.chunk1_id]}
                chunk2={expandedChunks[sim.chunk2_id]}
                onExpand={(chunkId) => loadChunkDetails(chunkId)}
              />
            ))}
          </div>
          
          <Pagination
            currentPage={page}
            totalPages={Math.ceil(similarities.total / pageSize)}
            onPageChange={setPage}
            hasMore={similarities.has_more}
          />
        </>
      )}
    </div>
  )
}
```

### 2.4 Update CommunitiesSection

**Objective:** Extract communities from summary, not full entity list

**Implementation:**

```tsx
// frontend/src/components/Document/CommunitiesSection.tsx

export default function CommunitiesSection({ documentId }: Props) {
  const [communities, setCommunities] = useState<Community[]>([])
  const [loading, setLoading] = useState(true)
  
  useEffect(() => {
    const loadCommunities = async () => {
      try {
        setLoading(true)
        // Use lightweight entities endpoint with community grouping
        const data = await api.getDocumentEntitiesPaginated(documentId, {
          limit: 10000  // Get all to extract unique communities
        })
        
        // Group by community
        const communityMap = new Map<number, Community>()
        
        data.entities.forEach(entity => {
          if (entity.community_id !== null && entity.community_id !== undefined) {
            const existing = communityMap.get(entity.community_id)
            if (existing) {
              existing.count += 1
            } else {
              communityMap.set(entity.community_id, {
                community_id: entity.community_id,
                level: entity.level,
                count: 1
              })
            }
          }
        })
        
        setCommunities(
          Array.from(communityMap.values())
            .sort((a, b) => a.community_id - b.community_id)
        )
      } catch (err) {
        console.error('Failed to load communities:', err)
      } finally {
        setLoading(false)
      }
    }
    
    loadCommunities()
  }, [documentId])
  
  return (
    <div>
      {/* Render communities with entity counts */}
    </div>
  )
}
```

## Phase 3: Caching Layer (Low Priority)

### 3.1 Install React Query

```bash
npm install @tanstack/react-query
```

### 3.2 Setup Query Client

```tsx
// frontend/src/app/layout.tsx

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,  // 5 minutes
      cacheTime: 10 * 60 * 1000,  // 10 minutes
      refetchOnWindowFocus: false,
      retry: 1
    }
  }
})

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  )
}
```

### 3.3 Use Query Hooks

```tsx
// Example usage in DocumentView

import { useQuery } from '@tanstack/react-query'

const { data: summary, isLoading, error } = useQuery({
  queryKey: ['document-summary', documentId],
  queryFn: () => api.getDocumentSummary(documentId),
  staleTime: 5 * 60 * 1000
})
```

**Benefits:**
- Automatic caching across navigation
- Background refetching
- Request deduplication
- Optimistic updates

## Performance Targets

### Before Optimization

| Metric | Current | User Experience |
|--------|---------|-----------------|
| Initial page load | 17.7 MB | 15+ seconds, blank screen |
| Similarities load | 15 MB, 14s | Long wait, browser freeze |
| Document details | 2.7 MB, 0.55s | Acceptable but slow |
| Navigation | Re-fetch all | Feels sluggish |
| Time to interactive | 15-20s | Poor |

### After Phase 1 (Backend Only)

| Metric | Target | User Experience |
|--------|--------|-----------------|
| Initial page load | 2 KB | <100ms, instant |
| Similarities first page | 50 KB, <1s | Fast, responsive |
| Full similarities | 174 KB total (50 items × 3.5 pages) | Progressive, smooth |
| Document details | 2.7 MB, 0.55s | Same (still acceptable) |
| Navigation | Re-fetch all | Same |
| Time to interactive | 1-2s | Good |

### After Phase 2 (Frontend Lazy Loading)

| Metric | Target | User Experience |
|--------|--------|-----------------|
| Initial page load | 2 KB | <100ms, instant |
| Time to first paint | <200ms | Immediate feedback |
| Tab switching | 0ms (no refetch) | Instant |
| Similarities pagination | 50 KB, <1s | Smooth scrolling |
| Document details | Lazy loaded | Only when needed |
| Navigation | Re-fetch all | Same |
| Time to interactive | <500ms | Excellent |

### After Phase 3 (Caching)

| Metric | Target | User Experience |
|--------|--------|-----------------|
| Initial page load | 2 KB | <100ms |
| Cached navigation | 0 KB, 0ms | Instant |
| Cache hit rate | >80% | Feels native |
| Network requests | -90% | Efficient |
| Time to interactive | <100ms (cached) | Native-like |

## Implementation Timeline

### Week 1: Backend API Changes
- Day 1-2: Implement paginated similarities endpoint
- Day 2-3: Add chunk details endpoint
- Day 3-4: Create document summary endpoint
- Day 4-5: Add paginated entities endpoint
- Day 5: Testing and optimization

### Week 2: Frontend Progressive Loading
- Day 1-2: Update API client with new methods
- Day 2-3: Refactor DocumentView for lazy loading
- Day 3-4: Update ChunkSimilaritiesSection with pagination
- Day 4-5: Update CommunitiesSection
- Day 5: Integration testing

### Week 3: Caching and Polish
- Day 1-2: Install and configure React Query
- Day 2-3: Migrate components to use query hooks
- Day 3-4: Add infinite scroll to similarities
- Day 4-5: Performance testing and optimization

## Testing Strategy

### Load Testing

```bash
# Test similarities endpoint before optimization
time curl "http://localhost:8000/api/documents/{doc_id}/similarities" > /dev/null

# Test similarities endpoint after optimization
time curl "http://localhost:8000/api/documents/{doc_id}/similarities?limit=50" > /dev/null
```

Expected improvement: 14s → <1s

### Performance Monitoring

```typescript
// Add performance markers
performance.mark('similarities-fetch-start')
await api.getDocumentSimilaritiesPaginated(docId)
performance.mark('similarities-fetch-end')
performance.measure('similarities-fetch', 'similarities-fetch-start', 'similarities-fetch-end')
```

### Metrics to Track

1. **Response size**: MB transferred per request
2. **Time to first byte**: Server response time
3. **Time to interactive**: User can interact with page
4. **Cache hit rate**: % of requests served from cache
5. **User-reported load time**: Perceived performance

## Monitoring and Rollback Plan

### Monitoring

Add metrics to track:
```python
# Backend metrics
from prometheus_client import Counter, Histogram

api_request_size = Histogram('api_response_size_bytes', 'API response size', ['endpoint'])
api_request_duration = Histogram('api_request_duration_seconds', 'API request duration', ['endpoint'])

@router.get("/{document_id}/similarities")
async def get_document_chunk_similarities(...):
    start_time = time.time()
    response = ...
    
    api_request_duration.labels(endpoint='similarities').observe(time.time() - start_time)
    api_request_size.labels(endpoint='similarities').observe(len(json.dumps(response)))
    
    return response
```

### Rollback Strategy

1. **Feature flags**: Enable new endpoints gradually
```python
ENABLE_PAGINATED_SIMILARITIES = os.getenv('ENABLE_PAGINATED_SIMILARITIES', 'true').lower() == 'true'
```

2. **Backward compatibility**: Keep old endpoints temporarily
```python
@router.get("/{document_id}/similarities/full")  # Old endpoint, deprecated
async def get_document_chunk_similarities_full(...):
    # Legacy endpoint for rollback
```

3. **Monitoring alerts**: Set up alerts for:
   - Response time > 2s
   - Error rate > 1%
   - Response size > 100KB (for paginated endpoints)

## Success Criteria

### Quantitative Metrics

- ✅ Similarities API response size: <100 KB per page
- ✅ Similarities API load time: <1 second
- ✅ Document summary load time: <100ms
- ✅ Time to interactive: <500ms for initial page load
- ✅ Cache hit rate: >70% after 5 minutes of usage

### Qualitative Metrics

- ✅ No perceived lag when switching tabs
- ✅ No browser freezing during data load
- ✅ Instant feedback on user interactions
- ✅ Smooth pagination experience

## Conclusion

This optimization plan addresses the root cause of performance issues (API over-fetching) without requiring infrastructure changes. By implementing pagination, lazy loading, and caching, we achieve a 150x reduction in initial data transfer and sub-second page loads.

The approach is pragmatic:
- Uses existing Neo4j database (no Postgres needed)
- Implements standard web performance patterns
- Provides immediate user experience improvements
- Maintains data integrity and consistency

Implementation can be done incrementally, with each phase providing measurable benefits.
