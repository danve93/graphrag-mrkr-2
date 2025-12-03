# Entity Types Taxonomy

Canonical entity types used in Amber's entity extraction system.

## Overview

Amber extracts and categorizes entities from ingested documents using a predefined taxonomy. This taxonomy was designed for IT infrastructure and enterprise documentation but can be extended for other domains.

Entity types enable:
- **Structured knowledge representation**
- **Type-based filtering and search**
- **Semantic relationship understanding**
- **Domain-specific reasoning**

## Canonical Entity Types

**Source**: `core/entity_extraction.py` → `CANONICAL_ENTITY_TYPES`

### Infrastructure & Components

**COMPONENT**
- Description: Hardware or software component
- Examples: vCenter Server, ESXi Host, NSX Manager, Database Server
- Usage: System architecture diagrams, component relationships

**NODE**
- Description: Physical or virtual computing node
- Examples: Node01, ESXi-Host-1, vm-prod-web-01
- Usage: Infrastructure topology, node dependencies

**SERVICE**
- Description: Running service or daemon
- Examples: vCenter Service, SSH Service, HTTP Server
- Usage: Service dependencies, availability requirements

**RESOURCE**
- Description: Allocated computational resource
- Examples: CPU Pool, Memory Resource, Storage Pool
- Usage: Resource allocation, capacity planning

**STORAGE_OBJECT**
- Description: Storage entity
- Examples: Datastore, LUN, Volume, NFS Share
- Usage: Storage architecture, data management

### Network & Security

**DOMAIN**
- Description: Network or authentication domain
- Examples: vsphere.local, ad.company.com, example.org
- Usage: Network architecture, authentication flows

**CERTIFICATE**
- Description: SSL/TLS certificate
- Examples: vCenter Certificate, Root CA, SSL Cert
- Usage: Certificate management, security configuration

**SECURITY_FEATURE**
- Description: Security mechanism or control
- Examples: Firewall Rule, Encryption, Access Control
- Usage: Security architecture, compliance

### Service Levels & Accounts

**CLASS_OF_SERVICE**
- Description: Service level or tier
- Examples: Gold Service, Premium Support, Standard SLA
- Usage: Service level agreements, tiered offerings

**ACCOUNT**
- Description: User or system account
- Examples: Administrator, root, service-account-01
- Usage: Access control, account management

**ACCOUNT_TYPE**
- Description: Account category
- Examples: Local Account, Domain User, Service Account
- Usage: Account taxonomy, permission models

**ROLE**
- Description: User role or permission set
- Examples: Admin, Read-Only, Operator
- Usage: Role-based access control

### Data & Configuration

**BACKUP_OBJECT**
- Description: Backup entity
- Examples: Full Backup, Incremental Backup, Snapshot
- Usage: Backup strategies, disaster recovery

**QUOTA_OBJECT**
- Description: Resource quota or limit
- Examples: Disk Quota, User Quota, Rate Limit
- Usage: Resource governance, limits

**CONFIG_OPTION**
- Description: Configuration parameter
- Examples: heap_size, timeout, max_connections
- Usage: System configuration, tuning

**ITEM**
- Description: Generic item or object
- Examples: License Key, Token, Asset
- Usage: General objects not fitting other types

### Operations & Procedures

**MIGRATION_PROCEDURE**
- Description: Migration or upgrade process
- Examples: vCenter Migration, Database Upgrade
- Usage: Migration planning, procedure documentation

**CLI_COMMAND**
- Description: Command-line instruction
- Examples: esxcli, vim-cmd, systemctl
- Usage: Operational procedures, automation

**API_OBJECT**
- Description: API endpoint or object
- Examples: REST API, vSphere API, /api/v1/users
- Usage: API integration, programmatic access

**TASK**
- Description: Operational task
- Examples: Backup Task, Replication Task, Sync Job
- Usage: Task scheduling, workflow automation

**PROCEDURE**
- Description: Operational procedure
- Examples: Startup Procedure, Failover Process
- Usage: Operational documentation, runbooks

### Knowledge & Context

**CONCEPT**
- Description: Abstract concept or principle
- Examples: High Availability, Load Balancing, Virtualization
- Usage: Conceptual knowledge, definitions

**DOCUMENT**
- Description: Reference to external document
- Examples: Installation Guide, Admin Manual, RFC 2616
- Usage: Documentation references, citations

**TECHNOLOGY**
- Description: Technology or framework
- Examples: VMware vSphere, Kubernetes, Docker
- Usage: Technology stack, platform identification

**PRODUCT**
- Description: Commercial product
- Examples: vSphere 7.0, Windows Server 2019, Oracle DB
- Usage: Product inventory, version tracking

### General Categories

**PERSON**
- Description: Individual person
- Examples: John Smith, Administrator Jones
- Usage: Contact information, ownership

**ORGANIZATION**
- Description: Company or organizational unit
- Examples: VMware, IT Department, Engineering Team
- Usage: Organizational context, ownership

**LOCATION**
- Description: Physical or logical location
- Examples: Datacenter A, Site-NYC, us-east-1
- Usage: Geographic distribution, site topology

**EVENT**
- Description: Temporal event
- Examples: System Outage, Maintenance Window, Release
- Usage: Event tracking, timeline reconstruction

**DATE**
- Description: Date or time reference
- Examples: 2024-01-01, Q4 2023, January
- Usage: Temporal context, scheduling

**MONEY**
- Description: Monetary value
- Examples: $1000, €500, 10K USD
- Usage: Cost tracking, budgeting

## Entity Type Distribution

Typical distribution in enterprise IT documentation:

```
COMPONENT:       15-20%  (e.g., vCenter, ESXi, NSX)
CONCEPT:         12-18%  (e.g., HA, DRS, vMotion)
DOCUMENT:        10-15%  (e.g., Admin Guide, Release Notes)
CONFIG_OPTION:    8-12%  (e.g., heap_size, timeout)
ITEM:             7-10%  (e.g., License, Token)
PRODUCT:          6-9%   (e.g., vSphere 7.0, VCSA)
TECHNOLOGY:       5-8%   (e.g., VMware, Kubernetes)
PROCEDURE:        4-7%   (e.g., Migration, Backup)
SERVICE:          3-6%   (e.g., vCenter Service, SSH)
NODE:             3-5%   (e.g., ESXi-Host-1, VM-01)
Other types:      <20%   (combined)
```

## Extraction Process

### LLM-Based Extraction

**Prompt Template** (simplified):
```
Extract entities from the following text. Classify each entity using these types:
{CANONICAL_ENTITY_TYPES}

Text:
{chunk_text}

Output format:
- Entity: <name>
  Type: <type>
  Description: <brief description>
```

**Implementation**: `core/entity_extraction.py`

```python
CANONICAL_ENTITY_TYPES = [
    "COMPONENT", "SERVICE", "NODE", "DOMAIN",
    "CLASS_OF_SERVICE", "ACCOUNT", "ACCOUNT_TYPE", "ROLE",
    # ... (full list)
]

def extract_entities(text: str) -> List[Entity]:
    prompt = build_extraction_prompt(text, CANONICAL_ENTITY_TYPES)
    response = llm.generate(prompt)
    entities = parse_extraction_response(response)
    return entities
```

### Entity Deduplication

**Phase 2 NetworkX Accumulation**:
- Entities with same name and type are merged
- Descriptions are concatenated
- Provenance is combined
- Importance scores are averaged

**Similarity Threshold**: `ACCUMULATION_SIMILARITY_THRESHOLD=0.85`

**Example**:
```python
# Before deduplication
Entity(name="vCenter", type="COMPONENT", description="Management server")
Entity(name="vCenter", type="COMPONENT", description="Central control plane")

# After deduplication
Entity(
    name="vCenter",
    type="COMPONENT",
    description="Management server. Central control plane",
    provenance=["chunk1", "chunk2"]
)
```

## Type-Specific Patterns

### COMPONENT Relationships

Components often relate to:
- **NODE**: Runs on or hosted by
- **SERVICE**: Provides or depends on
- **STORAGE_OBJECT**: Uses or manages
- **CERTIFICATE**: Secured by

**Example**:
```
vCenter Server (COMPONENT)
  ├─ runs on → ESXi Host (NODE)
  ├─ provides → vSphere API (SERVICE)
  ├─ manages → Datastore1 (STORAGE_OBJECT)
  └─ secured by → vCenter Cert (CERTIFICATE)
```

### PROCEDURE Relationships

Procedures often relate to:
- **CLI_COMMAND**: Involves execution of
- **COMPONENT**: Operates on
- **CONFIG_OPTION**: Modifies settings

**Example**:
```
Backup Procedure (PROCEDURE)
  ├─ executes → backup.sh (CLI_COMMAND)
  ├─ targets → Database Server (COMPONENT)
  └─ sets → backup_retention (CONFIG_OPTION)
```

### CONCEPT Relationships

Concepts often relate to:
- **TECHNOLOGY**: Implemented by
- **COMPONENT**: Provided by
- **PROCEDURE**: Enabled through

**Example**:
```
High Availability (CONCEPT)
  ├─ implemented by → VMware HA (TECHNOLOGY)
  ├─ provided by → vCenter (COMPONENT)
  └─ enabled through → HA Config (PROCEDURE)
```

## Filtering by Entity Type

### API Endpoint

```http
GET /api/documents/{document_id}/entities?entity_type=COMPONENT&limit=50
```

### Neo4j Query

```cypher
MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
WHERE e.type = $entity_type
RETURN DISTINCT e
ORDER BY e.importance DESC
LIMIT $limit;
```

### UI Filter

DocumentView entities panel:
- Displays entity type counts (e.g., "COMPONENT: 1014")
- Click "View" button to load entities of that type
- Paginated results (50 per page)

## Extending the Taxonomy

### Adding New Entity Types

1. **Update canonical list** (`core/entity_extraction.py`):
```python
CANONICAL_ENTITY_TYPES = [
    # Existing types...
    "NEW_TYPE",  # Add here
]
```

2. **Update extraction prompt**:
```python
# Add to prompt template with description
"NEW_TYPE: Description of new type"
```

3. **Update frontend types** (`frontend/src/types/index.ts`):
```typescript
export type EntityType =
  | 'COMPONENT'
  | 'SERVICE'
  // ...
  | 'NEW_TYPE';  // Add here
```

4. **Test extraction**:
```bash
pytest tests/unit/test_entity_extraction.py
```

5. **Reindex existing documents** (optional):
```bash
python scripts/ingest_documents.py --input-dir data/documents --force-reprocess
```

## Entity Type Statistics

### Query Type Distribution

```cypher
MATCH (e:Entity)
RETURN e.type as entity_type, count(e) as count
ORDER BY count DESC;
```

### Average Importance by Type

```cypher
MATCH (e:Entity)
WHERE e.importance IS NOT NULL
RETURN e.type as entity_type, avg(e.importance) as avg_importance
ORDER BY avg_importance DESC;
```

### Type Co-occurrence

```cypher
MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e1:Entity)
MATCH (c)-[:CONTAINS_ENTITY]->(e2:Entity)
WHERE e1.type < e2.type
RETURN e1.type, e2.type, count(c) as co_occurrence
ORDER BY co_occurrence DESC
LIMIT 20;
```

## Type-Based Search

### Find Related Components

```cypher
MATCH (e:Entity {name: $entity_name, type: "COMPONENT"})
MATCH (e)-[:RELATED_TO*1..2]-(related:Entity {type: "COMPONENT"})
RETURN DISTINCT related.name, related.description
ORDER BY related.importance DESC;
```

### Find Configuration Options for Component

```cypher
MATCH (comp:Entity {name: $component_name, type: "COMPONENT"})
MATCH (comp)-[:RELATED_TO]-(config:Entity {type: "CONFIG_OPTION"})
RETURN config.name, config.description;
```

### Find Procedures Involving Component

```cypher
MATCH (comp:Entity {name: $component_name, type: "COMPONENT"})
MATCH (comp)-[:RELATED_TO]-(proc:Entity {type: "PROCEDURE"})
RETURN proc.name, proc.description;
```

## Quality Assurance

### Entity Type Validation

Backend validates extracted types against canonical list:
```python
def validate_entity_type(entity_type: str) -> bool:
    return entity_type in CANONICAL_ENTITY_TYPES
```

Invalid types are logged and discarded:
```
WARNING: Unknown entity type 'INVALID_TYPE', discarding entity
```

### Type Consistency

Entities with same name should have same type:
```cypher
MATCH (e:Entity)
WITH e.name as name, collect(DISTINCT e.type) as types
WHERE size(types) > 1
RETURN name, types;
```

Clean up inconsistencies:
```cypher
MATCH (e1:Entity), (e2:Entity)
WHERE e1.name = e2.name
  AND e1.type <> e2.type
  AND id(e1) < id(e2)
MERGE (e1)-[:RELATED_TO {strength: 0.8}]-(e2)
```

## Best Practices

### Type Selection

**Use specific types** when possible:
- `COMPONENT` instead of `ITEM`
- `CLI_COMMAND` instead of `PROCEDURE`
- `STORAGE_OBJECT` instead of `RESOURCE`

**Avoid generic types** unless necessary:
- `ITEM` (too generic)
- `BACKUP_OBJECT` (specific)

### Type Hierarchy

Consider logical relationships:
```
Technology (VMware)
  └─ Product (vSphere 7.0)
      └─ Component (vCenter Server)
          └─ Service (vCenter Service)
```

### Documentation

Document custom types:
- Purpose and scope
- Example entities
- Relationship patterns
- Query examples

## Related Documentation

- [Entity Extraction](03-components/backend/entity-extraction.md)
- [Data Model](02-core-concepts/data-model.md)
- [Graph RAG Pipeline](02-core-concepts/graph-rag-pipeline.md)
- [Entity Clustering](04-features/entity-clustering.md)
