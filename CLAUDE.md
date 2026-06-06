# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AI BI Assistant** is an enterprise-grade Business Intelligence assistant combining:
- **LangGraph-based agent** (data analysis + knowledge retrieval + visualization)
- **Production-grade RAG system** with Milvus vector database and BGE-M3 hybrid search
- **Multi-LLM support** (OpenAI, Qwen, Deepseek, Claude, Ollama)
- **Streamlit UI** with real-time charting via Highcharts

## Running the Application

```bash
# 1. Start Milvus (Docker)
docker-compose -f docker-compose.milvus.yml up -d

# 2. Verify Milvus is running
docker-compose -f docker-compose.milvus.yml ps
# Expected: milvus-standalone, milvus-minio, milvus-etcd all "Up"

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Streamlit app
streamlit run main.py
```

Access at http://localhost:8501

## Key Commands

```bash
# Quick verification (RAG config, optimizations, KB status)
python tests/quick_verify.py

# Milvus connection & hybrid search test
python tests/quick_test_milvus.py

# 5-stage RAG pipeline test
python tests/test_rag_enhanced.py

# Adaptive/Hybrid/Context RAG strategy test
python tests/test_top5_rag_strategies.py

# KB loading & chunking diagnostics
python tests/diagnose_kb_flow.py
```

## Architecture

### Core Flow

```
main.py → agent.py (LangGraph) → Tools:
├── tools_rag.py          → RAG retrieval (知识库检索)
├── tools_text2sqlite.py  → NL → SQL conversion
├── tools_execute_sqlite.py → SQL execution
└── tools_charts.py       → Highcharts JSON generation
```

### RAG Pipeline (5-Stage Enhanced, Milvus only)

```
Query → Relevance Check → Query Transform → Vector Search → Result Merge → Rerank
        ↓                 ↓                 ↓              ↓              ↓
    relevance_checker  query_transformer  hybrid_retriever  result_merger  reranker
```

- **Hybrid Search**: Dense (1024-dim BGE-M3) + Sparse vectors with RRF fusion
- **Adaptive Strategy**: `src/rag/adaptive_rag.py` auto-routes by query type
- **Context Enrichment**: `src/rag/context_enriched.py` retrieves adjacent chunks

### Agent Workflow (LangGraph State Machine)

1. **System Prompt Selection**: 4 modes based on `db_available` + `kb_available`
   - `SYSTEM_PROMPT_FULL`: Both DB + KB
   - `SYSTEM_PROMPT_DATA_ONLY`: DB only
   - `SYSTEM_PROMPT_KB_ONLY`: KB only
   - `SYSTEM_PROMPT_CHAT_ONLY`: No tools

2. **Mandatory Chart Workflow**: For stats/comparison queries:
   ```
   text_to_sql_tool → execute_sql_tool → generate_chart_tool
   ```
   The agent MUST output both text analysis + a ` ```json ` code block with the chart config.

3. **Session State Management**: `main.py` monitors `db_config_active` + `kb_initialized` changes → auto-reinit agent.

### Key Files

| File | Responsibility |
|------|---------------|
| `agent.py` | LangGraph state machine, system prompts, tool binding |
| `main.py` | Streamlit app entry, session state, agent initialization |
| `config.py` | All configuration (LLM, embedding, vector DB, RAG, text processing) |
| `tools/tools_rag.py` | RAGManager class + `rag_retrieval_tool` |
| `tools/tools_text2sqlite.py` | Natural language to SQL conversion |
| `tools/tools_execute_sqlite.py` | SQL execution against MySQL |
| `tools/tools_charts.py` | Highcharts JSON chart generation |
| `tools/llm_factory.py` | Multi-provider LLM factory (OpenAI/Qwen/Deepseek/Claude/Ollama) |
| `tools/embedding_factory.py` | Multi-provider embedding factory |
| `tools/kb_loader.py` | Document loading and text cleaning |
| `tools/kb_manager.py` | Knowledge base file management (upload/delete) |
| `src/rag/enhanced_rag.py` | EnhancedRAGPipeline (5-stage pipeline) |
| `src/rag/adaptive_rag.py` | Adaptive strategy routing |
| `src/rag/hybrid_retriever.py` | Dense + sparse hybrid search |
| `src/rag/reranker.py` | BGE-Reranker re-ranking |
| `src/rag/context_enriched.py` | Adjacent chunk context enrichment |
| `src/vector_db/milvus_manager.py` | Milvus vector DB manager |
| `ui/chatbot_ui.py` | Chat interface rendering |
| `ui/sidebar.py` | Sidebar config panel |
| `ui/login.py` | Login/authentication |

### Configuration Pattern

All external integrations use **Factory Pattern** with provider abstraction controlled by environment variables:

```python
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "bgem3")
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma")  # chroma or milvus
```

Config priority: Environment Variable > `.env` file > `config.py` defaults.

Changing `CHUNK_SIZE`, `EMBEDDING_MODEL`, or `MILVUS_*` requires clicking **"重新加载知识库"** in UI.

### Knowledge Base Management

**Upload Flow**: `kb_manager.py::upload_file()` → Save to `data/kb_files/` → merge to `knowledge_base.txt` → clean text → smart split → embed → Milvus insert.

**Document Deletion**: Deletes from both Milvus AND filesystem.

**Text Cleaning**: Automatically removes page numbers, URLs, symbol-only lines. See `kb_loader.py::clean_text(debug=True)` for detailed logs.

## Critical Patterns

### 1. Tool Output Format for Charts

Charts MUST be returned as JSON wrapped in a markdown code block:
```python
json_str = json.dumps(config_dict, ensure_ascii=False, indent=2)
return f"```json\n{json_str}\n```"
```

The UI parses with `ui/chatbot_ui.py::extract_json_code_blocks()` using regex `r'```json([\s\S]*?)```'`.

### 2. Vector DB Abstraction

All vector DBs inherit `src/vector_db/base.py::BaseVectorDB`. Milvus uses `AnnSearchRequest` + `RRFRanker` for hybrid search.

### 3. Error Handling

All tools return structured responses:
```python
{"status": "success", "data": results, "message": "..."}
{"status": "error", "error": str(e), "traceback": traceback.format_exc()}
```

## RAG Strategy Modes

Configured via `RAG_STRATEGY_MODE` in `config.py`:
- `"simple"`: Basic vector search (fastest, lowest cost)
- `"hybrid"`: Hybrid search (recommended, balanced)
- `"enhanced"`: Hybrid + context enrichment (high quality)
- `"adaptive"`: Adaptive routing (intelligent, higher cost)
- `"full"`: Full pipeline (highest quality, highest cost)
