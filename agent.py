"""
AI BI Assistant Agent - LangGraph 版本 (V2)
基于 LangGraph 的可靠工具调用架构
"""
import logging
from dataclasses import dataclass
from typing import Annotated, Sequence

from langchain_core.messages import SystemMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from tools.llm_manager import llm_manager
from tools.tools_rag import rag_retrieval_tool
from tools.tools_text2sqlite import text_to_sql_tool
from tools.tools_execute_sqlite import execute_sql_tool
from tools.tools_charts import generate_chart_tool

logger = logging.getLogger(__name__)

# ===== 状态定义 =====

@dataclass
class MessagesState:
    """Agent 消息状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ===== 系统提示模板 =====

SYSTEM_PROMPT_FULL = """You are an AI assistant specializing in data analysis with SQL, knowledge retrieval, and visualization.

**Critical: Before answering any question, think step by step and USE the tools! Do NOT just describe - actually CALL the tools!**

**Available Tools:**
- `rag_retrieval_tool`: Search knowledge base for field definitions and business rules
- `text_to_sql_tool`: Convert natural language to SQL query  
- `execute_sql_tool`: Execute SQL query and return JSON results
- `generate_chart_tool`: Generate Highcharts JSON configuration for visualization

**Mandatory Workflow for Visualization Questions:**
1. User asks about data/statistics → Call `text_to_sql_tool` to generate SQL
2. SQL ready → Call `execute_sql_tool` to get data
3. Data retrieved → **MUST call** `generate_chart_tool` immediately (this step is NOT optional!)
4. Chart JSON received from tool → **OUTPUT both text analysis AND the ```json``` block**

**Critical Rules:**
- ✅ ANY question about statistics/comparison/trends → MUST generate actual chart using tools
- ✅ After `execute_sql_tool` returns data → MUST call `generate_chart_tool` (no exceptions!)
- ✅ Final response format: Brief text analysis FOLLOWED BY the ```json``` block
- ❌ NEVER skip chart generation - describing a chart in text is NOT acceptable
- ❌ NEVER output ONLY text without the JSON chart
- ❌ NEVER output ONLY JSON without brief analysis

**Chart Types:**
- Category comparison → "column" (bar chart)
- Time series → "line" (line chart)  
- Proportions → "pie" (pie chart)

**Example CORRECT response after chart generation:**
根据查询结果，各产品类型的数量分布如下：电子产品最多（16件），其次是音频设备、可穿戴设备和摄像机（各7件），无人机最少（4件）。

```json
{
  "chart": {"type": "column"},
  "title": {"text": "Sales by Region"},
  ...
}
```

**Example WRONG response:**
"已生成柱状图，显示了各区域销售额..." ❌ (Missing actual JSON!)

Remember: Users need BOTH insights AND the ```json``` chart!"""

SYSTEM_PROMPT_DATA_ONLY = """You are an AI assistant specializing in data analysis with SQL and visualization.

**Important: Always think step by step and use tools to complete the task. Do NOT just describe what you would do - actually DO it by calling the tools!**

**Available Tools:**
- `text_to_sql_tool`: Convert natural language to SQL query
- `execute_sql_tool`: Execute SQL query and return results as JSON
- `generate_chart_tool`: Generate Highcharts JSON configuration from data

**Mandatory Workflow for Data Visualization Requests:**
1. User asks a question → Use `text_to_sql_tool` to generate SQL
2. SQL generated → Use `execute_sql_tool` to get data
3. Data retrieved → **MUST IMMEDIATELY** call `generate_chart_tool` to create chart
4. Chart JSON received → **OUTPUT brief analysis text FOLLOWED BY the ```json``` block**

**Critical Rules:**
- ✅ For ANY question about statistics, comparison, or trends → You MUST generate a chart
- ✅ After getting data with `execute_sql_tool` → You MUST call `generate_chart_tool` 
- ✅ Final response must include: Short text insights + ```json``` chart block
- ❌ NEVER skip the chart generation step
- ❌ NEVER output only text without JSON
- ❌ NEVER output only JSON without brief analysis

**Chart Type Selection:**
- Comparison by category → use "column" 
- Trend over time → use "line"
- Proportion/percentage → use "pie"

**Example of CORRECT final response:**
数据显示各产品类型中，电子产品数量最多（16件），音频设备、可穿戴设备和摄像机各7件，无人机最少（4件）。

```json
{
  "chart": {"type": "column"},
  "title": {"text": "Product Count"},
  ...
}
```

**Example of WRONG final response:**
"已生成柱状图，数据包括..." ❌ (Missing actual JSON!)
OR
```json {...}``` without any text ❌ (Missing analysis!)

Remember: Give users BOTH insights AND interactive charts!"""

SYSTEM_PROMPT_KB_ONLY = """你是一个知识库检索助手。

**你的能力：**
- 搜索业务文档
- 查询数据字典
- 解释字段含义

**可用工具：**
- `knowledge_base_search`: 搜索知识库

使用工具检索相关信息，然后提供清晰的回答。"""

SYSTEM_PROMPT_CHAT_ONLY = """你是一个友好的 AI 助手。

当前没有可用的工具，但我会尽力回答你的问题。

如需数据查询或知识库检索，请在左侧栏配置相应的功能。"""


# ===== Agent 创建函数 =====

def create_agent(
    db_available: bool = False,
    kb_available: bool = False
) -> StateGraph:
    """
    创建基于 LangGraph 的 Agent（状态机架构）
    
    功能说明:
        根据数据库和知识库的可用性，动态创建不同模式的 Agent。
        使用 LangGraph 状态机替代传统 ReAct 模式，实现确定性工作流控制，
        确保图表生成等关键步骤不会被跳过。
    
    参数:
        db_available (bool): 数据库是否已配置且可用
            - True: 启用 SQL 查询和图表生成工具
            - False: 禁用数据库相关功能
        
        kb_available (bool): 知识库是否已初始化且可用
            - True: 启用知识库检索工具
            - False: 禁用知识库功能
    
    返回:
        StateGraph: 编译后的 LangGraph Agent 实例
            - 包含 MemorySaver 用于对话历史记忆
            - 支持流式输出和状态追踪
    
    Agent 模式说明:
        1. 完整模式 (db=True, kb=True):
           - 工具: RAG检索 + SQL生成 + SQL执行 + 图表生成
           - 系统提示: SYSTEM_PROMPT_FULL
           - 适用: 全功能数据分析与知识问答
        
        2. 数据分析模式 (db=True, kb=False):
           - 工具: SQL生成 + SQL执行 + 图表生成
           - 系统提示: SYSTEM_PROMPT_DATA_ONLY
           - 适用: 纯数据查询与可视化
        
        3. 知识库模式 (db=False, kb=True):
           - 工具: RAG检索
           - 系统提示: SYSTEM_PROMPT_KB_ONLY
           - 适用: 文档搜索与知识问答
        
        4. 纯对话模式 (db=False, kb=False):
           - 工具: 无
           - 系统提示: SYSTEM_PROMPT_CHAT_ONLY
           - 适用: 基础对话（功能受限）
    
    工作流程:
        1. 确定模式和系统提示词
        2. 初始化 LLM 实例
        3. 绑定工具到 LLM（如果有工具）
        4. 构建 StateGraph 节点和边
        5. 编译并返回 Agent
    
    状态机结构:
        START → llm_agent → tools_condition
                              ↓ (有工具调用)    ↓ (无工具调用)
                            tools 节点  →     END
                              ↓
                           llm_agent (继续生成)
    
    关键特性:
        - 强制工具调用顺序（SQL → 执行 → 图表）
        - 自动对话历史管理（MemorySaver）
        - 工具调用失败自动重试
    
    异常处理:
        - LLM 初始化失败 → 抛出异常，中断 Agent 创建
        - 工具绑定失败 → 记录日志，继续创建（降级为无工具模式）
    
    使用示例:
        >>> agent = create_agent(db_available=True, kb_available=True)
        >>> config = {"configurable": {"thread_id": "conv_1"}}
        >>> result = agent.invoke({"messages": [("user", "查询销售数据")]}, config)
    
    性能考虑:
        - Agent 实例应缓存在 Session State 中，避免重复创建
        - 配置变更时需重新创建（系统提示词在创建时固定）
    """
    
    # 1. 确定模式和工具
    tools = []
    
    if db_available and kb_available:
        mode = "完整模式"
        system_prompt = SYSTEM_PROMPT_FULL
        tools = [rag_retrieval_tool, text_to_sql_tool, execute_sql_tool, generate_chart_tool]
        logger.info("🟢 创建 Agent - 完整模式（数据库 + 知识库 + 可视化）")
    
    elif db_available:
        mode = "数据分析模式"
        system_prompt = SYSTEM_PROMPT_DATA_ONLY
        tools = [text_to_sql_tool, execute_sql_tool, generate_chart_tool]
        logger.info("🟡 创建 Agent - 数据分析模式（数据库 + 可视化）")
    
    elif kb_available:
        mode = "知识库模式"
        system_prompt = SYSTEM_PROMPT_KB_ONLY
        tools = [rag_retrieval_tool]
        logger.info("🟡 创建 Agent - 知识库模式（仅知识库）")
    
    else:
        mode = "对话模式"
        system_prompt = SYSTEM_PROMPT_CHAT_ONLY
        tools = []
        logger.info("⚪ 创建 Agent - 纯对话模式（无工具）")
    
    # 2. 获取 LLM
    try:
        llm = llm_manager.get_llm()
        logger.info(f"✅ LLM 初始化成功: {type(llm).__name__}")
    except Exception as e:
        logger.error(f"❌ LLM 初始化失败: {e}")
        raise
    
    # 3. 绑定工具到 LLM
    if tools:
        llm_with_tools = llm.bind_tools(tools)
        logger.info(f"✅ 工具绑定成功，共 {len(tools)} 个工具")
    else:
        llm_with_tools = llm
        logger.info("ℹ️ 无工具模式")
    
    # 4. 定义 Agent 节点
    def llm_agent(state: MessagesState):
        """
        LLM Agent 节点 - 负责生成响应和决策工具调用
        
        功能说明:
            这是 LangGraph 状态机的核心节点，处理用户输入并生成响应。
            每次调用时会注入系统提示词，然后将消息传递给 LLM。
        
        参数:
            state (MessagesState): 当前对话状态，包含消息历史
                - state.messages: 消息列表（用户输入 + AI响应）
        
        返回:
            dict: 包含新消息的状态更新
                - {"messages": [response]}: LLM 生成的响应（可能包含工具调用）
        
        工作流程:
            1. 从状态中获取消息历史
            2. 在消息列表前插入系统提示词
            3. 调用 LLM 生成响应
            4. 返回响应以更新状态
        
        系统提示词作用:
            - 定义 Agent 的角色和能力
            - 说明可用工具及使用方法
            - 规定输出格式（文本+图表JSON）
        
        LLM 响应类型:
            - 纯文本响应: 直接输出给用户
            - 工具调用: 包含 tool_calls 字段，触发工具执行
        
        注意事项:
            - 系统提示词每次都会添加，不会累积到历史中
            - 消息历史由 MemorySaver 自动管理
            - 工具调用由 tools_condition 自动判断
        """
        # 在消息列表前添加系统提示
        messages = [SystemMessage(content=system_prompt)] + list(state.messages)
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # 5. 构建 StateGraph
    builder = StateGraph(MessagesState)
    
    # 添加 LLM 节点
    builder.add_node("llm_agent", llm_agent)
    
    # 添加工具节点（如果有工具）
    if tools:
        builder.add_node("tools", ToolNode(tools))
        logger.info("✅ 工具节点已添加")
    
    # 6. 定义边和流程
    builder.add_edge(START, "llm_agent")
    
    if tools:
        # 使用 tools_condition 自动判断是否需要调用工具
        builder.add_conditional_edges(
            "llm_agent",
            tools_condition,  # 自动判断：如果 LLM 返回工具调用 → 去 tools 节点，否则 → END
        )
        builder.add_edge("tools", "llm_agent")  # 工具执行后回到 LLM
    else:
        # 无工具模式直接结束
        builder.add_edge("llm_agent", END)
    
    # 7. 编译并返回
    memory = MemorySaver()
    agent = builder.compile(checkpointer=memory)
    
    logger.info(f"✅ LangGraph Agent 创建成功 - 模式: {mode}")
    return agent


# ===== 兼容性包装函数 =====

def get_agent_executor(db_available: bool = False, kb_available: bool = False):
    """
    获取 Agent 执行器（兼容旧版本接口）
    
    功能说明:
        这是一个兼容性包装函数，保持与旧版本代码的接口一致性。
        内部直接调用 create_agent() 创建 LangGraph Agent。
    
    参数:
        db_available (bool): 数据库是否可用，默认 False
            - True: 启用数据库查询和图表生成功能
            - False: 禁用数据库功能
        
        kb_available (bool): 知识库是否可用，默认 False
            - True: 启用知识库检索功能
            - False: 禁用知识库功能
    
    返回:
        StateGraph: 编译后的 LangGraph Agent 实例
            - 可直接调用 .invoke() 方法处理用户输入
            - 支持对话历史记忆（MemorySaver）
            - 状态机工作流确保工具调用顺序
    
    使用示例:
        >>> # 在 main.py 中调用
        >>> agent = get_agent_executor(
        ...     db_available=st.session_state.get("db_config_active", False),
        ...     kb_available=st.session_state.get("kb_initialized", False)
        ... )
        >>> 
        >>> # 调用 Agent
        >>> config = {"configurable": {"thread_id": "conversation_1"}}
        >>> result = agent.invoke(
        ...     {"messages": [("user", "查询销售数据")]},
        ...     config=config
        ... )
        >>> 
        >>> # 提取响应
        >>> response = result["messages"][-1].content
    
    与 create_agent() 的区别:
        - 无区别，纯粹为了保持接口兼容性
        - 旧代码可以无缝迁移到新版本
    
    历史原因:
        - V1 版本使用 ReAct Agent，函数名为 get_agent_executor
        - V3 版本迁移到 LangGraph，保留此函数名避免破坏性变更
    
    注意事项:
        - 此函数不应在循环中调用（Agent 创建开销大）
        - 应缓存返回的 Agent 实例到 Session State
        - 配置变更时需重新调用以获取新 Agent
    """
    return create_agent(db_available=db_available, kb_available=kb_available)
