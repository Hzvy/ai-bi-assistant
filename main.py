"""
AI BI Assistant - 主程序入口 (V2 - LangGraph)
"""
import streamlit as st
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置页面配置
st.set_page_config(
    page_title="AI BI Assistant V4",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入模块
from config import config
from agent import get_agent_executor
from ui.sidebar import render_sidebar
from ui.chatbot_ui import render_chatbot
from ui.login import render_login_page, check_login_status

def initialize_agent():
    """
    初始化 LangGraph Agent
    
    功能说明:
        根据当前数据库和知识库的可用状态创建 Agent 实例。
        Agent 的模式（完整/数据分析/知识库/纯对话）由资源可用性决定。
    
    参数:
        无（从 st.session_state 读取状态）
    
    返回:
        bool: Agent 初始化是否成功
            - True: 成功创建并存储到 session_state.agent
            - False: 初始化失败（通常是 LLM 连接问题）
    
    工作流程:
        1. 从 Session State 读取数据库和知识库状态
        2. 调用 get_agent_executor() 创建 Agent
        3. 将 Agent 实例存储到 session_state.agent
        4. 根据资源可用性设置工作模式标签
        5. 记录日志并返回成功状态
    
    Agent 模式判断逻辑:
        | 数据库 | 知识库 | 模式         | 可用功能                     |
        |--------|--------|--------------|------------------------------|
        | ✅     | ✅     | 完整模式     | SQL查询 + 图表 + 知识检索    |
        | ✅     | ❌     | 数据分析模式 | SQL查询 + 图表               |
        | ❌     | ✅     | 对话模式     | 知识检索                     |
        | ❌     | ❌     | 纯对话模式   | 基础对话（功能受限）         |
    
    Session State 字段:
        读取:
            - db_config_active: 数据库是否已配置
            - kb_initialized: 知识库是否已初始化
        
        写入:
            - agent: Agent 实例
            - agent_mode: 工作模式标签（用于 UI 显示）
    
    错误处理:
        - LLM 连接失败 → 记录错误日志，显示错误提示，返回 False
        - 工具绑定失败 → 记录警告，Agent 仍会创建（降级为无工具模式）
    
    使用场景:
        - 应用首次启动时
        - 数据库配置变更时
        - 知识库重新加载时
        - Agent 为 None 时
    
    注意事项:
        - 此函数开销较大，应避免频繁调用
        - main.py 中通过配置变更检测机制控制调用时机
        - Agent 实例会缓存到 Session State，跨页面刷新保持
    
    异常示例:
        >>> # LLM API Key 错误
        >>> ❌ Agent 初始化失败: Incorrect API key provided
        >>> 
        >>> # 网络连接问题
        >>> ❌ Agent 初始化失败: Connection timeout
    """
    try:
        # 检查数据库和知识库状态
        db_available = st.session_state.get("db_config_active", False)
        kb_available = st.session_state.get("kb_initialized", False)
        
        logger.info(f"🔄 初始化 Agent - 数据库: {db_available}, 知识库: {kb_available}")
        
        # 创建 Agent
        agent = get_agent_executor(
            db_available=db_available,
            kb_available=kb_available
        )
        
        # 存储到 session_state
        st.session_state.agent = agent
        
        # 设置工作模式
        if db_available and kb_available:
            st.session_state.agent_mode = "完整模式"
        elif db_available:
            st.session_state.agent_mode = "数据分析模式"
        elif kb_available:
            st.session_state.agent_mode = "对话模式"
        else:
            st.session_state.agent_mode = "纯对话模式"
        
        logger.info(f"✅ Agent 初始化成功 - 模式: {st.session_state.agent_mode}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Agent 初始化失败: {e}", exc_info=True)
        st.error(f"❌ Agent 初始化失败: {str(e)}")
        return False

def main():
    """
    主函数 - 应用入口点
    
    功能说明:
        Streamlit 应用的主流程控制器，负责：
        1. Session State 初始化
        2. UI 组件渲染（侧边栏、聊天界面）
        3. 配置变更检测
        4. Agent 自动重新初始化
    
    执行流程:
        ┌─────────────────────────────────────┐
        │ 1. 初始化 Session State             │
        │    - agent, agent_mode              │
        │    - db_config_active, kb_initialized│
        ├─────────────────────────────────────┤
        │ 2. 渲染侧边栏                       │
        │    （用户可能修改配置）             │
        ├─────────────────────────────────────┤
        │ 3. 检测配置变更                     │
        │    - 数据库状态变化？               │
        │    - 知识库状态变化？               │
        ├─────────────────────────────────────┤
        │ 4. 重新初始化 Agent（如需要）       │
        │    - need_reinit=True → 调用 initialize_agent() │
        ├─────────────────────────────────────┤
        │ 5. 渲染聊天界面                     │
        │    - 使用最新的 Agent 实例          │
        └─────────────────────────────────────┘
    
    配置变更检测机制:
        使用 "last_*_state" 字段追踪上一次的配置状态，
        通过对比当前状态判断是否需要重新初始化 Agent。
        
        检测逻辑:
        ```python
        current_db = st.session_state.get("db_config_active", False)
        if last_db_state != current_db:
            need_reinit = True  # 触发重新初始化
        ```
    
    为什么需要重新初始化:
        - LangGraph Agent 的系统提示词在创建时固定
        - 不同模式使用不同的系统提示词
        - 配置变更后必须重新创建 Agent 才能切换模式
    
    Session State 字段说明:
        核心字段:
            - agent (StateGraph): Agent 实例
            - agent_mode (str): 当前工作模式标签
        
        状态追踪:
            - db_config_active (bool): 数据库是否激活
            - kb_initialized (bool): 知识库是否就绪
            - last_db_state (bool): 上次数据库状态
            - last_kb_state (bool): 上次知识库状态
    
    性能优化:
        - Agent 实例缓存在 Session State，避免重复创建
        - 仅在配置变更时重新初始化，减少不必要开销
        - 侧边栏优先渲染，确保配置及时生效
    
    日志输出示例:
        ```
        🔄 数据库状态变化: False → True
        🔄 重新初始化 Agent...
        ✅ Agent 初始化成功 - 模式: 数据分析模式
        ```
    
    使用场景:
        - 应用启动: streamlit run main.py
        - 页面刷新: 用户刷新浏览器
        - 配置修改: 用户在侧边栏修改数据库/知识库配置
    
    注意事项:
        - 此函数每次页面交互都会执行（Streamlit 特性）
        - 状态管理依赖 Session State，浏览器标签关闭后清除
        - 多用户并发时 Session State 自动隔离
    """
    
    # ===== 登录检查 =====
    if not check_login_status():
        render_login_page()
        return
    
    # 初始化 session_state
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "agent_mode" not in st.session_state:
        st.session_state.agent_mode = "纯对话模式"
    
    if "db_config_active" not in st.session_state:
        st.session_state.db_config_active = False
    
    if "kb_initialized" not in st.session_state:
        st.session_state.kb_initialized = False
    
    # 渲染侧边栏
    render_sidebar()
    
    # 检查是否需要重新初始化 Agent
    need_reinit = False
    
    # 检查数据库配置变化
    current_db = st.session_state.get("db_config_active", False)
    if "last_db_state" not in st.session_state:
        st.session_state.last_db_state = current_db
        need_reinit = True
    elif st.session_state.last_db_state != current_db:
        logger.info(f"🔄 数据库状态变化: {st.session_state.last_db_state} → {current_db}")
        st.session_state.last_db_state = current_db
        need_reinit = True
    
    # 检查知识库配置变化
    current_kb = st.session_state.get("kb_initialized", False)
    if "last_kb_state" not in st.session_state:
        st.session_state.last_kb_state = current_kb
        need_reinit = True
    elif st.session_state.last_kb_state != current_kb:
        logger.info(f"🔄 知识库状态变化: {st.session_state.last_kb_state} → {current_kb}")
        st.session_state.last_kb_state = current_kb
        need_reinit = True
    
    # 重新初始化 Agent（如果需要）
    if need_reinit or st.session_state.agent is None:
        logger.info("🔄 重新初始化 Agent...")
        initialize_agent()
    
    # 渲染聊天界面
    render_chatbot()

if __name__ == "__main__":
    main()
