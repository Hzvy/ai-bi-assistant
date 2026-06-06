# AI BI Assistant V4

企业级智能 BI 助手：基于 **Streamlit + LangGraph** 的多工具 Agent，融合 **Text-to-SQL**、**RAG 知识库检索** 与 **Highcharts 图表生成**。本版本支持 **分级权限知识库**（Milvus 标量过滤）。

---

## 🆕 V4 相比之前版本的更新

> 这里的“之前版本”指仓库中较早的 V2/V3 实现与旧 README 描述；以下条目均以 v4 代码与配置为准。

- **分级权限知识库（Milvus）**：引入 `kb_level/department` 等元数据，检索时使用 Milvus 标量过滤实现按 `access_level` 隔离；登录与权限来源于 [user.txt](user.txt)，并提供了权限验证脚本 [tests/test_permission_system.py](tests/test_permission_system.py)。
- **交付与启动更清晰**：README 明确使用仓库内置的 Milvus Compose（[docker-compose.milvus.yml](docker-compose.milvus.yml)、[docker-compose.milvus-core.yml](docker-compose.milvus-core.yml)）与默认端口。
- **图表链路更确定**：数据问题走强制工作流（NL→SQL→执行→Highcharts JSON），并要求输出 **```json 代码块** 供 UI 解析渲染（见 `tools/tools_charts.py` 生成逻辑与 UI 的 JSON 提取）。
- **配置项更细 + 可组合优化**：`.env.example` 提供更丰富的 LLM/Embedding/向量库/RAG 开关（如 `RAG_STRATEGY_MODE`、混合检索、上下文增强、重排等），便于在质量/成本/性能间切换。
- **诊断与验证更体系化**：新增/整理了多类快速验证脚本（如 [tests/quick_verify.py](tests/quick_verify.py)、[tests/diagnose_kb_flow.py](tests/diagnose_kb_flow.py)、[tests/quick_test_milvus.py](tests/quick_test_milvus.py)），用于定位配置与入库/检索问题。

### 更新点对照表（参考 V3 说明文档 v2 的表格风格）

| 功能模块 | V2 (基础版) | V3 (企业版) | V4 (当前) |
|---|---|---|---|
| **向量数据库** | ChromaDB（单机） | Milvus（生产级） | Milvus / Chroma（Milvus 模式支持 `kb_level` 标量过滤） |
| **知识库管理** | 简单文档加载 | 完整管理系统（上传/删除/预览） | 完整管理系统 + **分级权限**（上传选择级别，检索按用户级别过滤） |
| **RAG 架构** | 基础检索 | 增强型 RAG（多阶段优化） | 增强型 RAG + 权限过滤（Milvus 模式下在向量库侧过滤） |
| **文档处理** | 直接分割 | 智能清洗 + 自适应分割 | 保持（同 V3，可通过 `.env` 开关控制） |
| **嵌入模型** | 单一模型 | BGE-M3（混合检索） | 保持（BGE-M3 混合检索；可切换多种 Embedding Provider） |
| **登录与权限** | 无 | 无（默认不做用户隔离） | **user.txt 登录** + `access_level/department`（知识库分级可用） |
| **测试/诊断脚本** | 基础 | `quick_verify/diagnose_kb_flow/...` | 在 V3 基础上新增 **权限系统测试**（[tests/test_permission_system.py](tests/test_permission_system.py)） |

## 🚀 快速开始（Windows）

### 1) 创建虚拟环境并安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

（可选）你也可以用 conda，但 README 默认以 venv 为主。

### 2) 配置环境变量

```bash
copy .env.example .env
notepad .env
```

最低需要配置：

- `LLM_PROVIDER` + 对应的 API Key（例如 `QWEN_API_KEY` / `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` 等）
- `EMBEDDING_PROVIDER`（推荐 `bgem3` 或 `huggingface`）
- `VECTOR_DB_TYPE`（`milvus` 或 `chroma`）

完整可选项请以 [.env.example](.env.example) 为准。

### 3)（推荐）启动 Milvus

项目内已提供 compose 文件：

```bash
docker-compose -f docker-compose.milvus.yml up -d
```

也可使用精简版：

```bash
docker-compose -f docker-compose.milvus-core.yml up -d
```

默认端口：Milvus `19530`，健康检查 `9091`，MinIO `9000/9001`。

### 4) 启动应用

```bash
streamlit run main.py
```

打开 http://localhost:8501

### 5) 登录

登录凭据定义在 [user.txt](user.txt)。示例（可能已在你的仓库中预置）：

- 用户名：`Hzvy`
- 密码：`123456`

---

## ✨ 核心能力

- **对话式 BI**：自然语言提问 → 数据分析（SQL）/知识库检索（RAG）/图表生成
- **多模型支持**：OpenAI / Qwen / Deepseek / Anthropic / Ollama / 智谱 等（见 `.env.example`）
- **Highcharts 可视化**：工具生成 Highcharts JSON，UI 自动解析并渲染
- **知识库管理**：上传/删除/状态查看，文本清洗 + 智能分割 + 向量化入库
- **分级权限知识库（Milvus）**：上传时选择知识库级别，检索时按用户级别过滤

---

## 🧭 架构与关键流程

### 代码结构（主链路）

```
main.py → agent.py (LangGraph) → tools/
  ├─ tools_text2sql.py      # NL → SQL
  ├─ tools_execute_sql.py   # 执行 SQL
  ├─ tools_charts.py        # 生成 Highcharts JSON（必须输出 ```json 代码块）
  └─ tools_rag.py           # RAG 检索（Milvus/Chroma）
```

### Agent 模式

Agent 会根据资源可用性自动选择模式：

- **Full**：数据库 + 知识库
- **Data**：仅数据库
- **KB**：仅知识库
- **Chat**：无工具（纯对话）

### 图表生成强制工作流

对“统计/对比/趋势/分布”等数据问题，Agent 会遵循固定顺序：

1. `text_to_sql_tool`
2. `execute_sql_tool`
3. `generate_chart_tool`

并在回答中输出：

- 文字分析
- 以及一个 **```json 代码块**（Highcharts 配置）。UI 会用正则提取该 JSON 并渲染图表。

---

## 📚 知识库与权限（Milvus）

### 文件与目录

- UI 上传文件保存目录：`data/knowledge_base/`
- （可选/历史路径）`data/kb_files/`：部分管理脚本会用到该目录

知识库入库使用 `load_kb_from_files()` 读取目录内的 `.txt/.md/.pdf` 文件，并按当前配置进行清洗、分割与向量化。

### 分级权限说明

登录信息来自 [user.txt](user.txt)，其中包含：

- `access_level`：用户权限级别（1/2/3）
- `department`：部门（可用于元数据）

当 `VECTOR_DB_TYPE=milvus` 时：

- 上传文档会写入 `kb_level` 等元数据
- 检索时使用 Milvus 标量过滤（例如 `kb_level <= user_level`）实现权限隔离

更详细的使用说明见：

- [PERMISSION_SYSTEM_GUIDE.md](PERMISSION_SYSTEM_GUIDE.md)
- [PERMISSION_SYSTEM_IMPLEMENTATION.md](PERMISSION_SYSTEM_IMPLEMENTATION.md)

### 重要：Schema 变更后需要清理集合

如果你从旧集合升级而来（或出现字段不匹配/检索异常），请先清理并重建：

```bash
python tests/clear_milvus.py
```

---

## 🧪 测试与快速诊断

常用脚本在 [tests/](tests/)：

```bash
python tests/quick_verify.py
python tests/quick_test_milvus.py
python tests/test_rag_enhanced.py
python tests/test_top5_rag_strategies.py
python tests/test_permission_system.py
python tests/diagnose_kb_flow.py
```

---

## 🔍 常见问题

### 1) Milvus 连接失败

- 确认容器已启动：`docker-compose -f docker-compose.milvus.yml ps`
- 确认端口监听：`netstat -an | findstr 19530`

### 2) 图表不渲染

- 确认回答中包含 **```json 代码块**（Highcharts 配置）
- 相关逻辑在 UI 的 JSON 提取函数中

### 3) 上传后检索不到知识库内容

- 确认已加载知识库（侧边栏的“重新加载/初始化”相关操作）
- 确认权限级别足够（Level 1 看不到 Level 2/3 文档）
- 如果更改过嵌入模型/分块参数/Milvus 配置，建议清理后重建集合

---

## 🙏 致谢

- LangChain / LangGraph
- Streamlit
- Milvus
- FlagEmbedding（BGE-M3）
# ai-bi-assistant
