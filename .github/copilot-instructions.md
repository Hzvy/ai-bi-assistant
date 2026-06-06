# AI BI Assistant V3 - AI Agent Instructions

## Project Overview

**AI BI Assistant V3** is an enterprise-grade Business Intelligence assistant combining:
- **LangGraph-based multi-modal agent** (data analysis + knowledge retrieval + visualization)
- **Production-grade RAG system** with Milvus vector database and BGE-M3 hybrid search
- **Multi-LLM support** (OpenAI, Qwen, Deepseek, Claude, Ollama)
- **Streamlit UI** with real-time charting via Highcharts

## Architecture & Data Flow

### Core Components

```
main.py → agent.py (LangGraph) → Tools:
├── tools_rag.py          → RAG retrieval (知识库检索)
├── tools_text2sql.py  → NL → SQL conversion
├── tools_execute_sql.py → SQL execution
└── tools_charts.py       → Highcharts JSON generation
```

### RAG Pipeline (5-Stage Enhanced)

```
Query → Relevance Check → Query Transform → Vector Search → Result Merge → Rerank
        ↓                 ↓                 ↓              ↓              ↓
    relevance_checker  query_transformer  hybrid_retriever  result_merger  reranker
```

- **Hybrid Search**: Dense (1024-dim BGE-M3) + Sparse vectors with RRF fusion
- **Adaptive Strategy**: `src/rag/adaptive_rag.py` auto-routes by query type (simple/comparison/technical)
- **Context Enrichment**: `context_enriched.py` retrieves adjacent chunks to solve fragmentation

### Agent Workflow (LangGraph State Machine)

1. **System Prompt Selection**: 4 modes based on `db_available` + `kb_available`
   - `SYSTEM_PROMPT_FULL`: Both DB + KB (完整模式)
   - `SYSTEM_PROMPT_DATA_ONLY`: DB only (数据分析模式)
   - `SYSTEM_PROMPT_KB_ONLY`: KB only (对话模式)
   - `SYSTEM_PROMPT_CHAT_ONLY`: No tools (纯对话)

2. **Mandatory Chart Workflow**: For ANY stats/comparison query:
   ```
   text_to_sql_tool → execute_sql_tool → generate_chart_tool (MUST call!)
   ```
   - **Critical**: Agent MUST output both text analysis + ```json``` chart block
   - Never skip `generate_chart_tool` after getting data

3. **State Management**: `main.py` monitors `db_config_active` + `kb_initialized` changes → auto-reinit agent

## Configuration Pattern

### Multi-Provider Factory Pattern

All external integrations use **Factory Pattern** with provider abstraction:

- **LLM**: `tools/llm_factory.py` → `OpenAIProvider`, `QwenProvider`, `DeepseekProvider`, etc.
- **Embedding**: `tools/embedding_factory.py` → `HuggingFaceEmbeddingProvider`, `OpenAIEmbeddingProvider`, etc.
- **Vector DB**: `src/vector_db/base.py` → `MilvusManager`, `ChromaManager`

Example:
```python
# config.py controls which provider is active
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # Switch: openai/qwen/deepseek
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "milvus")
```

### Configuration Override Priority

```
Environment Variable > .env file > config.py defaults
```

**Critical Config Rules**:
- Changing `CHUNK_SIZE`, `EMBEDDING_MODEL`, or `MILVUS_*` requires clicking **"🔄 重新加载知识库"** in UI
- `session_state` tracks last config state; mismatches trigger agent reinitialization

## Development Workflows

### Running the Application

```bash
# 1. Start Milvus (Windows)
docker-compose -f docker-compose.milvus.yml up -d

# 2. Verify Milvus status
docker-compose -f docker-compose.milvus.yml ps
# Expected: milvus-standalone, milvus-minio, milvus-etcd all "Up"

# 3. Run Streamlit app
streamlit run main.py
```

### Testing & Debugging

**Quick Verification**:
```bash
python tests/quick_verify.py  # Shows: RAG config, optimizations enabled, KB status
```

**Specific Module Tests**:
- `tests/quick_test_milvus.py` - Milvus connection & hybrid search
- `tests/test_rag_enhanced.py` - 5-stage RAG pipeline
- `tests/test_top5_rag_strategies.py` - Adaptive/Hybrid/Context strategies
- `tests/diagnose_kb_flow.py` - KB loading & chunking diagnostics

**Common Debugging Pattern**:
```python
# All major modules use this logging setup
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Change in code or config.LOG_LEVEL
```

### Knowledge Base Management

**Upload Flow** (in `tools/kb_manager.py`):
```
1. upload_file() → Save to data/kb_files/
2. merge_files_to_kb() → Consolidate to knowledge_base.txt
3. initialize_kb() → Clean text → Smart split → Embed → Milvus.insert()
```

**Document Deletion**:
```python
# Deletes from both Milvus AND filesystem
delete_kb_document(source="filename.pdf")
→ Milvus.delete(expr=f"source == '{source}'")
→ os.remove(f"data/kb_files/{source}")
```

**Text Cleaning** (automatically applied):
- Removes: page numbers, URLs, symbol-only lines, repeated punctuation
- Preserves: Chinese/English text, code blocks, structured content
- See `tools/kb_loader.py::clean_text(debug=True)` for detailed logs

## Critical Patterns & Conventions

### 1. Tool Output Format for Charts

**MUST follow this pattern** in `tools_charts.py`:
```python
# ✅ CORRECT: Return JSON wrapped in markdown code block
json_str = json.dumps(config_dict, ensure_ascii=False, indent=2)
return f"```json\n{json_str}\n```"

# ❌ WRONG: Return raw dict or string
return config_dict  # UI won't render!
```

UI parses with `ui/chatbot_ui.py::extract_json_code_blocks()` using regex:
```python
pattern = r'```json([\s\S]*?)```'
```

### 2. Vector DB Abstraction (Base Class Contract)

All vector DBs inherit `src/vector_db/base.py::BaseVectorDB`:
```python
class BaseVectorDB(ABC):
    @abstractmethod
    def initialize_kb(self, documents: List[Document]) -> bool: ...
    @abstractmethod
    def similarity_search(self, query: str, k: int) -> List[Document]: ...
    @abstractmethod
    def delete_documents(self, expr: str) -> int: ...
```

**Milvus-specific**: Uses `AnnSearchRequest` + `RRFRanker` for hybrid search:
```python
# Dense vector search request
dense_req = AnnSearchRequest(
    data=[dense_embedding],
    anns_field="dense_vector",
    param={"metric_type": "IP", "params": {"nprobe": 10}},
    limit=k
)
# Sparse vector search request (similar)
# Then: hybrid_results = collection.hybrid_search([dense_req, sparse_req], rerank=RRFRanker())
```

### 3. Session State Management (Streamlit-Specific)

**Reinit Triggers** (in `main.py`):
```python
# Track last known states
if "last_db_state" not in st.session_state:
    st.session_state.last_db_state = current_db
    need_reinit = True
elif st.session_state.last_db_state != current_db:
    need_reinit = True  # DB config changed → recreate agent

# Similar for kb_initialized, embedding_provider, etc.
```

**Always check session state** before accessing `st.session_state.agent`:
```python
if "agent" not in st.session_state or st.session_state.agent is None:
    initialize_agent()
```

### 4. SQL Injection Prevention

Uses **SQLite parameterized queries** (though text-to-SQL doesn't auto-sanitize):
```python
# In tools_execute_sql.py
cursor.execute(sql_query)  # Direct execution - SQL is LLM-generated
# Validation happens in tools_text2sql.py via LLM prompt engineering
```

**Security Note**: Current implementation trusts LLM output. For production, add SQL parsing validation.

### 5. Error Handling Pattern

All tools return structured responses:
```python
# ✅ Success
{"status": "success", "data": results, "message": "..."}

# ❌ Failure
{"status": "error", "error": str(e), "traceback": traceback.format_exc()}
```

Agent handles gracefully via `try/except` in tool nodes.

## Project-Specific Quirks

### Why Two docker-compose Files?

- `docker-compose.milvus.yml` - Full Milvus (etcd + MinIO + standalone)
- `docker-compose.milvus-core.yml` - Minimal Milvus (faster startup, testing only)

### Chinese File Paths in Windows

Project uses `工业大模型` in path. Be careful with:
```python
# ✅ Use Pathlib for cross-platform compatibility
from pathlib import Path
kb_dir = Path("data/kb_files")

# ❌ Avoid hardcoded separators
kb_dir = "data\\kb_files"  # Breaks on Linux
```

### BGE-M3 Model Location

Embedding model auto-downloads to `~/.cache/huggingface/`:
```bash
# Model: BAAI/bge-m3 (~2.3GB)
# First run: ~5-10min download
# Check: %USERPROFILE%\.cache\huggingface\hub\models--BAAI--bge-m3
```

### Streamlit Caching Strategy

**NOT using** `@st.cache_data` or `@st.cache_resource` for agent/LLM:
- Reason: Config changes should immediately reflect
- Instead: Manual state tracking via `session_state.last_*_state`

## When Modifying This Project

### Adding New RAG Strategy

1. Create `src/rag/your_strategy.py` inheriting pattern from `adaptive_rag.py`
2. Add config switches in `config.py`: `RAG_ENABLE_YOUR_STRATEGY = ...`
3. Integrate in `tools/tools_rag.py::rag_manager.retrieve()`
4. Document in `docs/TOP5_RAG策略集成说明.md`

### Adding New LLM Provider

1. Create class in `tools/llm_factory.py`:
   ```python
   class YourProvider(BaseLLMProvider):
       def get_llm(self, **kwargs): ...
   ```
2. Add to `LLMFactory.providers` dict
3. Update `config.py` with API keys/endpoints
4. Test with `python tests/quick_verify.py`

### Changing Chunking Strategy

**Warning**: Requires full KB rebuild!
```python
# In tools/kb_loader.py
RecursiveCharacterTextSplitter(
    chunk_size=500,      # config.CHUNK_SIZE
    chunk_overlap=50,    # config.CHUNK_OVERLAP
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
)
```

After changing: Users MUST click "🔄 重新加载知识库" to re-embed all documents.

## File & Directory Responsibilities

- `agent.py` - LangGraph state machine, system prompts, tool binding
- `main.py` - Streamlit app entry, session state management, agent initialization
- `config.py` - Single source of truth for all configuration
- `tools/` - LangChain tools (RAG, SQL, charts) + factories (LLM, embedding)
- `src/rag/` - RAG enhancement modules (hybrid, adaptive, rerank, etc.)
- `src/vector_db/` - Vector DB abstraction (Milvus, Chroma)
- `ui/` - Streamlit UI components (chatbot, sidebar, KB manager)
- `data/kb_files/` - User-uploaded documents (persistent)
- `data/knowledge_base.txt` - Merged KB source (auto-generated)
- `sql/` - Sample database schemas (DDL + sample data)

## Testing Checklist Before Committing

- [ ] Run `python tests/quick_verify.py` (no errors)
- [ ] Milvus connection test: `python tests/quick_test_milvus.py`
- [ ] Test chart generation: Upload → Query "各产品类型数量分布" → Verify JSON block
- [ ] Test KB upload: Upload PDF → Check Milvus count → Delete → Verify removal
- [ ] Test config switch: Change `LLM_PROVIDER` → Restart → Verify correct model used

---

**Version**: V3 (Milvus + Enhanced RAG)  
**Last Updated**: 2025-10  
**For questions**: Check `docs/TOP5_RAG策略集成说明.md` and `README.md`
