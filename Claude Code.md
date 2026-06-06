## 项目详细分析：AI BI Assistant V4

### 一、项目概述

这是一个 **企业级智能 BI 助手** ，基于 LangGraph + LangChain 构建，结合了**数据库查询（Text-to-SQL）**和**RAG（检索增强生成）**两大核心能力。应用通过 Streamlit 提供 Web UI。

 **核心定位** ：让非技术人员通过自然语言与数据库和数据文档交互。

---

### 二、技术栈

| 层次                 | 技术选型                                                             |
| -------------------- | -------------------------------------------------------------------- |
| **Web 框架**   | Streamlit 1.49.0                                                     |
| **Agent 框架** | LangGraph 0.2.38 + LangChain 0.3.17                                  |
| **LLM**        | 多厂商支持：OpenAI / 通义千问 / DeepSeek / Anthropic / 智谱 / Ollama |
| **Embedding**  | HuggingFace (BGE-M3) / OpenAI / Ollama / ZhipuAI 等 7 种             |
| **向量数据库** | Milvus (生产) + ChromaDB (开发/回退)                                 |
| **数据库**     | MySQL (PyMySQL)                                                      |
| **可视化**     | Highcharts (streamlit-highcharts) + Altair                           |
| **文档处理**   | pdfplumber / PyPDF2 + 文本清洗 + 智能分割                            |

---

### 三、架构设计

#### 3.1 整体架构

```
Streamlit UI (main.py)
    ├── Login (ui/login.py)
    ├── Sidebar (ui/sidebar.py)
    │    ├── 数据库配置
    │    ├── 知识库管理
    │    ├── 应用设置
    │    └── 帮助文档
    └── Chatbot (ui/chatbot_ui.py)
         ├── 消息历史
         ├── JSON 代码块提取
         └── Highcharts 渲染

LangGraph Agent (agent.py)
    ├── 完整模式: RAG + Text2SQL + SQL执行 + 图表
    ├── 数据分析模式: Text2SQL + SQL执行 + 图表
    ├── 知识库模式: RAG检索
    └── 纯对话模式: 直接 LLM

Tools (4个 LangChain Tool)
    ├── rag_retrieval_tool     → RAGManager → EnhancedRAGPipeline
    ├── text_to_sql_tool       → LLM → 生成SQL
    ├── execute_sql_tool       → PyMySQL → 执行SQL
    └── generate_chart_tool    → LLM → Highcharts JSON
```

#### 3.2 Agent 状态机

```
START → llm_agent → tools_condition → (有工具调用) → tools → llm_agent (循环)
                                        → (无工具调用) → END
```

使用 `MemorySaver` 保持对话记忆，`thread_id` 隔离会话。

#### 3.3 RAG 增强管道（5阶段）

```
查询 → 相关性检查 → 查询变换 → 混合检索(Dense+Sparse+RRF)
                                              ↓
                                        结果合并 → 重排序 → 返回
```

支持 5 种策略模式：`simple` / `hybrid` / `enhanced` / `adaptive` / `full`

---

### 四、核心模块分析

| 模块                | 文件                                                                                                                        | 职责                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **入口**      | [main.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/main.py)                                       | Streamlit 入口，Session State 管理，Agent 自动重初始化 |
| **Agent**     | [agent.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/agent.py)                                     | LangGraph 状态机构建，4种模式，系统提示词管理          |
| **配置**      | [config.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/config.py)                                   | Config 类，环境变量加载，双层配置（Session + .env）    |
| **LLM 工厂**  | [tools/llm_factory.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/tools/llm_factory.py)             | 6种提供商策略模式，单例 LLMManager                     |
| **Embedding** | [tools/embedding_factory.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/tools/embedding_factory.py) | 7种 Embedding 提供商，BGEM3 支持混合检索               |
| **RAG 核心**  | [tools/tools_rag.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/tools/tools_rag.py)                 | RAGManager，知识库初始化/检索，工具包装                |
| **增强RAG**   | [src/rag/enhanced_rag.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/src/rag/enhanced_rag.py)       | 5阶段管道，5种策略，懒加载优化                         |
| **文档处理**  | [tools/kb_loader.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/tools/kb_loader.py)                 | 多编码读取，文本清洗(6类规则)，智能分割(7种模式)       |
| **Text2SQL**  | [tools/tools_text2sql.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/tools/tools_text2sql.py)       | LLM 生成 MySQL SQL，JSON 解析，日期函数指导            |
| **SQL 执行**  | [tools/tools_execute_sql.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/tools/tools_execute_sql.py) | PyMySQL 连接，类型转换，JSON 序列化                    |
| **图表**      | [tools/tools_charts.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/tools/tools_charts.py)           | Highcharts JSON 生成，3种图表模板                      |
| **聊天UI**    | [ui/chatbot_ui.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/ui/chatbot_ui.py)                     | 消息渲染，JSON 块提取，Highcharts 可视化               |
| **侧边栏**    | [ui/sidebar.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/ui/sidebar.py)                           | Tab导航，数据库配置，知识库管理入口                    |

---

### 五、关键设计亮点

1. **工厂模式 + 策略模式** ：LLM 和 Embedding 均通过工厂可热插拔切换供应商，无需改代码
2. **LangGraph 确定性工作流** ：替代传统 ReAct，用状态机确保 SQL→执行→图表 的强制顺序
3. **双层配置优先级** ：Session State > .env > 默认值，支持运行时动态切换数据库
4. **5阶段增强 RAG** ：从基础向量检索(85%召回)到全流程优化(98%召回)逐级递进
5. **智能文档处理** ：6类文本清洗规则 + 7种自动分割模式检测
6. **权限控制** ：知识库文档标注 `kb_level`（1-3级），检索时按用户权限过滤
7. **优雅降级** ：Milvus 不可用→ChromaDB；增强模式失败→标准检索

---

### 六、配置现状

根据 `.env` 文件：

* **LLM** : 通义千问 qwen-plus (DashScope)
* **Embedding** : BGEM3 (BAAI/bge-m3)
* **向量库** : Milvus (localhost:19530)
* **RAG 策略** : `simple`（默认，可改为 `hybrid`/`full` 提升效果）
* **数据库** : MySQL (root/root@localhost:3306/llm)

---

### 七、待改进点

1. **安全问题** ：[tools_execute_sql.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/tools/tools_execute_sql.py#L80-L85) 直接执行 LLM 生成的 SQL，无 DROP/DELETE 防护
2. **策略未启用** ：`RAG_STRATEGY_MODE=simple` 浪费了混合检索能力，建议改为 `hybrid`
3. **冗余代码** ：`llm_manager.py` 是一个薄的代理层，全部委托给 `llm_factory.py` 的 manager
4. **硬编码 thread_id** ：[chatbot_ui.py](vscode-webview://1p3k8nj72i9kflfv10ij6p5qh0aq25vssl5hid6oatmk47q5vnaq/ui/chatbot_ui.py#L574) 使用 `"conversation_1"`，多用户时会产生交叉
5. **文档命名** ：README 仍写 "V3"，实际版本已到 V4
6. **测试覆盖率** ：tests/ 目录存在但文件较少，核心 RAG 管道缺乏集成测试
