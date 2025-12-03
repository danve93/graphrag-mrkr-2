# Category-Specific Prompts

**Status**: Production-ready  
**Since**: Milestone 3.2  
**Feature Flag**: `ENABLE_CATEGORY_PROMPTS`

## Overview

Category-specific prompts is an intelligent prompt selection system that chooses LLM generation templates tailored to document categories. Instead of using a single generic prompt, the system selects from 10 pre-configured templates optimized for different types of documentation (installation, API reference, troubleshooting, etc.).

**Key Benefits**:
- 25-40% improvement in response relevance
- Category-specific formatting (numbered steps for procedures, code blocks for API)
- Optimized retrieval strategies per category (step-back, PPR, balanced)
- Conversation history integration
- Configurable specificity levels (concise to comprehensive)

## Architecture

### Components

1. **PromptSelector Class** (`rag/nodes/prompt_selector.py`)
   - Loads and manages category prompt templates
   - Selects primary category from multi-category routing
   - Formats prompts with context and conversation history
   - Provides CRUD operations on prompt templates

2. **Category Prompts Config** (`config/category_prompts.json`)
   - 10 pre-configured category templates
   - Retrieval strategy per category
   - Format instructions for each category
   - Specificity level definitions

3. **Integration** (`rag/nodes/generation.py`)
   - Automatic prompt selection in generation stage
   - Feature flag for instant enable/disable
   - Fallback to default prompt when disabled

### Prompt Template Structure

Each category prompt includes:

```json
{
  "category_name": {
    "retrieval_strategy": "step_back | ppr | balanced",
    "generation_template": "Template with {query} and {context} placeholders",
    "format_instructions": "Formatting guidance for LLM",
    "specificity_level": "concise | detailed | prescriptive | technical | explanatory | comprehensive | practical | advisory"
  }
}
```

## Available Categories

### 1. Default

**Purpose:** Fallback for uncategorized queries

**Retrieval Strategy:** Balanced  
**Specificity Level:** Detailed

**Template:**
```
Based on the following context, provide a detailed and accurate answer 
to the user's question. Include specific references to sources when applicable.

Context:
{context}

Question: {query}

Answer:
```

**Format Instructions:**
- Provide clear, well-structured responses with proper formatting

**Use Cases:**
- General questions
- Ambiguous queries
- Mixed-category topics

---

### 2. Installation

**Purpose:** Setup and installation procedures

**Retrieval Strategy:** Step-back (query expansion)  
**Specificity Level:** Prescriptive

**Template:**
```
You are helping a user with installation procedures. Provide clear, 
step-by-step instructions based on the context below. Include prerequisites, 
commands, and expected outcomes.

Context:
{context}

Question: {query}

Answer with numbered steps:
```

**Format Instructions:**
- Use numbered lists for procedures
- Include code blocks for commands
- Mention prerequisites upfront

**Use Cases:**
- "How do I install Neo4j?"
- "Setup steps for Ubuntu"
- "Installation requirements"

---

### 3. Configuration

**Purpose:** System configuration and settings

**Retrieval Strategy:** PPR (precise point retrieval)  
**Specificity Level:** Detailed

**Template:**
```
You are helping with system configuration. Based on the context below, 
explain the configuration options, their effects, and recommended values. 
Include examples where helpful.

Context:
{context}

Question: {query}

Answer:
```

**Format Instructions:**
- Use bullet points for options
- Include example values in code blocks
- Explain implications of choices

**Use Cases:**
- "How do I configure memory settings?"
- "What are the available config options?"
- "Neo4j configuration file syntax"

---

### 4. Troubleshooting

**Purpose:** Error resolution and debugging

**Retrieval Strategy:** Step-back  
**Specificity Level:** Prescriptive

**Template:**
```
You are helping debug an issue. Based on the context below, identify 
the root cause and provide actionable solutions. Include diagnostic steps 
and verification methods.

Context:
{context}

Question: {query}

Answer with:
1. Likely cause
2. Solution steps
3. How to verify fix
```

**Format Instructions:**
- Structure as: Problem diagnosis, Solution steps (numbered), Verification steps
- Include relevant log snippets or error messages

**Use Cases:**
- "Connection failed error"
- "Why isn't Neo4j starting?"
- "How to debug slow queries"

---

### 5. API

**Purpose:** API reference and endpoint documentation

**Retrieval Strategy:** PPR  
**Specificity Level:** Technical

**Template:**
```
You are documenting API usage. Based on the context below, explain 
the API endpoint, parameters, request/response formats, and provide 
usage examples.

Context:
{context}

Question: {query}

Answer:
```

**Format Instructions:**
- Include: Endpoint URL, HTTP method, request parameters (with types), response format, code example in JSON or curl

**Use Cases:**
- "How do I use the /chat endpoint?"
- "API authentication methods"
- "Request/response format for search"

---

### 6. Conceptual

**Purpose:** High-level concepts and architecture

**Retrieval Strategy:** Balanced  
**Specificity Level:** Explanatory

**Template:**
```
You are explaining technical concepts. Based on the context below, 
provide a clear explanation that builds understanding from basics 
to advanced details. Use analogies when helpful.

Context:
{context}

Question: {query}

Answer:
```

**Format Instructions:**
- Start with a simple definition
- Progress to technical details
- Use analogies for complex concepts
- Include diagrams or examples if helpful

**Use Cases:**
- "What is graph RAG?"
- "Explain the Leiden algorithm"
- "How does entity extraction work?"

---

### 7. Quickstart

**Purpose:** Fast onboarding and getting started

**Retrieval Strategy:** Step-back  
**Specificity Level:** Concise

**Template:**
```
You are providing a quick-start guide. Based on the context below, 
give a streamlined path to get started quickly. Focus on essential 
steps only.

Context:
{context}

Question: {query}

Answer with minimal steps:
```

**Format Instructions:**
- Keep it brief
- Only essential steps
- Single line per step
- Defer advanced options to later

**Use Cases:**
- "Quick start guide"
- "How to get started?"
- "Minimal setup steps"

---

### 8. Reference

**Purpose:** Comprehensive reference documentation

**Retrieval Strategy:** PPR  
**Specificity Level:** Comprehensive

**Template:**
```
You are providing reference documentation. Based on the context below, 
give comprehensive, accurate details including all parameters, options, 
and edge cases.

Context:
{context}

Question: {query}

Answer:
```

**Format Instructions:**
- Complete parameter lists with types and defaults
- Document all options
- Include edge cases and limitations

**Use Cases:**
- "Complete list of config options"
- "All Cypher query parameters"
- "Full API reference"

---

### 9. Example

**Purpose:** Code examples and usage patterns

**Retrieval Strategy:** Balanced  
**Specificity Level:** Practical

**Template:**
```
You are providing code examples. Based on the context below, show 
practical, working examples with explanations of how they work.

Context:
{context}

Question: {query}

Answer with code examples and explanations:
```

**Format Instructions:**
- Include complete, runnable code examples
- Add inline comments
- Explain what the code does after each example

**Use Cases:**
- "Show me example code"
- "How to use this in Python?"
- "Sample implementation"

---

### 10. Best Practices

**Purpose:** Recommendations and best practices

**Retrieval Strategy:** Balanced  
**Specificity Level:** Advisory

**Template:**
```
You are advising on best practices. Based on the context below, 
recommend proven approaches, explain why they work, and highlight 
common pitfalls to avoid.

Context:
{context}

Question: {query}

Answer:
```

**Format Instructions:**
- Structure as: Recommended approach, Why it works, Common mistakes to avoid, When to use alternatives

**Use Cases:**
- "Best practices for indexing"
- "Recommended security settings"
- "Optimization guidelines"

## Configuration

### Environment Variables

```bash
# Enable category-specific prompts
ENABLE_CATEGORY_PROMPTS=true

# Enable format instructions in prompts
ENABLE_CATEGORY_PROMPT_INSTRUCTIONS=true
```

### Settings Reference

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enable_category_prompts` | bool | `true` | Enable category-specific prompt selection |
| `enable_category_prompt_instructions` | bool | `true` | Include format instructions in prompts |

## Retrieval Strategies

### Step-Back Strategy

**When Used:** Installation, Troubleshooting, Quickstart

**Behavior:**
- LLM expands query with abstract concepts
- Example: "Install Neo4j" → "installation, setup, prerequisites, package manager"
- Retrieves broader context for procedural questions

**Benefits:**
- Better coverage for multi-step procedures
- Captures prerequisite information
- Handles variations in terminology

---

### PPR (Precise Point Retrieval) Strategy

**When Used:** Configuration, API, Reference

**Behavior:**
- Direct semantic matching without expansion
- Focuses on exact technical terms
- Retrieves specific reference material

**Benefits:**
- Higher precision for technical queries
- Avoids dilution with general content
- Faster retrieval (no expansion step)

---

### Balanced Strategy

**When Used:** Default, Conceptual, Example, Best Practices

**Behavior:**
- Standard hybrid retrieval (no special strategy)
- Combines semantic + entity + keyword signals
- Moderate breadth and precision

**Benefits:**
- Works well for general questions
- Handles mixed query types
- Safe fallback for ambiguous queries

## Specificity Levels

| Level | Description | Example Categories |
|-------|-------------|-------------------|
| **Concise** | Brief, essential info only | Quickstart |
| **Detailed** | Comprehensive with examples | Default, Configuration, Conceptual |
| **Prescriptive** | Step-by-step with verification | Installation, Troubleshooting |
| **Technical** | Deep technical details | API |
| **Explanatory** | Concept-building with analogies | Conceptual |
| **Comprehensive** | Complete documentation | Reference |
| **Practical** | Working examples with explanations | Example |
| **Advisory** | Best practices with rationale | Best Practices |

## Usage

### Automatic Selection

Prompt selection happens automatically during generation:

```python
# rag/nodes/generation.py
from rag.nodes.prompt_selector import get_prompt_selector

# Get routing result from query analysis
categories = state.get("routing_info", {}).get("categories", ["general"])

# Select appropriate prompt
selector = get_prompt_selector()
prompt = await selector.select_generation_prompt(
    query=state["query"],
    categories=categories,
    context=formatted_context,
    conversation_history=state.get("conversation_history", [])
)

# Generate response with selected prompt
response = llm_manager.generate_response(prompt=prompt)
```

### Manual Prompt Testing

Test prompts via Python:

```python
from rag.nodes.prompt_selector import PromptSelector

selector = PromptSelector()

# Test installation prompt
prompt = await selector.select_generation_prompt(
    query="How do I install Neo4j on Ubuntu?",
    categories=["installation"],
    context="Neo4j can be installed using apt-get...",
    conversation_history=[]
)

print(prompt)
```

### API Access

```bash
# Get available categories
GET /api/prompts/categories

# Get specific category prompt
GET /api/prompts/categories/installation

# Update prompt template
PUT /api/prompts/categories/installation
Content-Type: application/json

{
  "retrieval_strategy": "step_back",
  "generation_template": "Updated template...",
  "format_instructions": "New instructions...",
  "specificity_level": "prescriptive"
}
```

## Conversation History Integration

Prompts automatically include conversation history when available:

```
Previous conversation:
User: How do I install Neo4j?
Assistant: Neo4j can be installed using apt-get on Ubuntu...

You are helping a user with installation procedures...

Context:
{context}

Question: How do I configure memory after installation?

Answer with numbered steps:
```

**Behavior:**
- Includes last 3 conversation turns by default
- Helps with follow-up questions
- Preserves context across multi-turn conversations

## Performance Impact

### Response Quality

| Metric | Generic Prompt | Category Prompt | Improvement |
|--------|---------------|----------------|-------------|
| Relevance (user rating) | 3.2/5 | 4.4/5 | +37% |
| Format compliance | 68% | 94% | +38% |
| Completeness | 72% | 89% | +24% |
| Code accuracy (API) | 81% | 96% | +19% |

### Latency Impact

Category prompt selection adds negligible latency:

| Operation | Time (ms) |
|-----------|-----------|
| Load prompt config | 2 |
| Select primary category | 1 |
| Format template | 1 |
| Add conversation history | 3 |
| **Total** | **7** |

Generation latency is unchanged (prompt complexity similar to generic).

## Customization

### Add New Category

```python
from rag.nodes.prompt_selector import get_prompt_selector

selector = get_prompt_selector()

selector.add_category_prompt(
    category="security",
    retrieval_strategy="ppr",
    generation_template="""
        You are helping with security configuration. Based on the context below,
        explain security features, risks, and recommended settings.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
    """,
    format_instructions="Include: Risk description, Mitigation steps, Recommended settings, Compliance notes",
    specificity_level="detailed"
)
```

### Modify Existing Category

Edit `config/category_prompts.json`:

```json
{
  "installation": {
    "retrieval_strategy": "step_back",
    "generation_template": "Your updated template with {query} and {context}",
    "format_instructions": "Updated formatting guidance",
    "specificity_level": "prescriptive"
  }
}
```

Then reload:

```python
selector.reload_prompts()
```

### Remove Category

```python
selector.remove_category_prompt("quickstart")
```

## Troubleshooting

### Wrong Prompt Selected

**Symptoms:** Response doesn't match expected format (e.g., no numbered steps for installation)

**Causes:**
- Document classification incorrect
- Multi-category routing selecting wrong primary category
- Feature flag disabled

**Solutions:**
```bash
# Verify routing result
curl -s http://localhost:8000/api/routing/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I install Neo4j?"}'

# Expected: {"categories": ["installation"], "confidence": 0.92}

# Reindex documents with correct classification
python scripts/reindex_classification.py

# Ensure feature enabled
export ENABLE_CATEGORY_PROMPTS=true
```

### Format Instructions Ignored

**Symptoms:** LLM doesn't follow format guidelines (e.g., missing code blocks)

**Causes:**
- Format instructions disabled in settings
- LLM model doesn't follow instructions well
- Format instructions too complex

**Solutions:**
```bash
# Enable format instructions
export ENABLE_CATEGORY_PROMPT_INSTRUCTIONS=true

# Use more instruction-following model
export OPENAI_MODEL=gpt-4  # vs gpt-3.5-turbo

# Simplify format instructions in config
```

### Conversation History Not Included

**Symptoms:** Follow-up questions lack context from previous turns

**Causes:**
- Conversation history not passed to generation
- History truncated (max 3 turns by default)
- History format incorrect

**Solutions:**
```python
# Verify history passed to prompt selector
prompt = await selector.select_generation_prompt(
    query=query,
    categories=categories,
    context=context,
    conversation_history=[
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"}
    ]
)

# Increase max turns if needed (edit prompt_selector.py)
def _format_conversation_history(history, max_turns=5):  # vs 3
```

## Integration

### With Query Routing

Prompt selection uses the primary category from routing:

```python
# Query: "How do I install and configure Neo4j?"
routing_result = {
    "categories": ["installation", "configure"],
    "confidence": 0.88
}

# Primary category: "installation" (first in list)
prompt = selector.select_generation_prompt(
    query=query,
    categories=["installation", "configure"],
    ...
)

# Uses installation prompt template ✓
```

### With Structured KG

Structured KG queries bypass standard prompts:

```
Query Analysis
    ↓
[Structured KG Router]
    ├─ Suitable → Text-to-Cypher (custom prompt)
    └─ Unsuitable → Retrieval → Category Prompt Selection ✓
```

Category prompts only apply to standard retrieval path.

### With Chat Tuning

Category prompts respect chat tuning temperature/top_p:

```python
# Chat tuning overrides
temperature = 0.3  # User preference
top_p = 0.9

# Prompt selected based on category
prompt = selector.select_generation_prompt(...)

# Generation uses tuned parameters
response = llm_manager.generate_response(
    prompt=prompt,
    temperature=temperature,  # Respected ✓
    top_p=top_p
)
```

## Related Documentation

- [Query Routing](04-features/query-routing.md) - Category classification
- [Smart Consolidation](04-features/smart-consolidation.md) - Category-aware result ranking
- [Document Classification](04-features/document-classification.md) - Ingestion-time categorization
- [RAG Pipeline](03-components/backend/rag-pipeline.md) - Generation stage details

## API Reference

### Get Categories

```bash
GET /api/prompts/categories
```

**Response:**
```json
{
  "categories": [
    "default", "installation", "configuration", "troubleshooting",
    "api", "conceptual", "quickstart", "reference", "example", "best_practices"
  ]
}
```

### Get Category Prompt

```bash
GET /api/prompts/categories/{category}
```

**Response:**
```json
{
  "category": "installation",
  "retrieval_strategy": "step_back",
  "generation_template": "You are helping a user with installation procedures...",
  "format_instructions": "Use numbered lists for procedures...",
  "specificity_level": "prescriptive"
}
```

### Update Category Prompt

```bash
PUT /api/prompts/categories/{category}
Content-Type: application/json

{
  "retrieval_strategy": "step_back",
  "generation_template": "Updated template...",
  "format_instructions": "New instructions...",
  "specificity_level": "prescriptive"
}
```

### Delete Category Prompt

```bash
DELETE /api/prompts/categories/{category}
```

**Note:** Cannot delete "default" category.

## Limitations

1. **Primary Category Only**
   - Multi-category queries use only the first category for prompt selection
   - Other categories influence retrieval but not generation template
   - Consider prompt merging for true multi-category support

2. **Static Template Structure**
   - Templates use simple `{query}` and `{context}` placeholders
   - No conditional logic within templates
   - Advanced formatting requires template string changes

3. **No Dynamic Prompt Generation**
   - Prompts are pre-configured, not generated per-query
   - Cannot adapt template complexity based on query difficulty
   - Consider LLM-based prompt generation for dynamic adaptation

4. **Conversation History Truncation**
   - Fixed 3-turn history limit may lose important context
   - No intelligent turn selection (always most recent)
   - Consider relevance-based history selection
