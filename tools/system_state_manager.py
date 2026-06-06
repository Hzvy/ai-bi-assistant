"""
系统状态管理器 - 集中管理应用全局状态
确保知识库、数据库、Agent 等状态实时同步
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any
import os


class SystemStateManager:
    """系统状态管理器 - 单例模式"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化状态管理器"""
        self._initialize_session_state()
    
    @staticmethod
    def initialize_session_state():
        """公共初始化方法 - 用于外部调用"""
        SystemStateManager._initialize_session_state()
    
    @staticmethod
    def _initialize_session_state():
        """初始化 Streamlit 会话状态"""
        
        # 初始化模式和配置
        if "agent_mode" not in st.session_state:
            st.session_state.agent_mode = "纯对话模式"
        
        if "agent" not in st.session_state:
            st.session_state.agent = None
        
        # 数据库配置状态
        if "db_config_active" not in st.session_state:
            st.session_state.db_config_active = False
        
        if "db_host" not in st.session_state:
            st.session_state.db_host = os.getenv("DB_HOST", "")
        
        if "db_port" not in st.session_state:
            st.session_state.db_port = int(os.getenv("DB_PORT", "3306"))
        
        if "db_user" not in st.session_state:
            st.session_state.db_user = os.getenv("DB_USER", "")
        
        if "db_password" not in st.session_state:
            st.session_state.db_password = os.getenv("DB_PASSWORD", "")
        
        if "db_name" not in st.session_state:
            st.session_state.db_name = os.getenv("DB_NAME", "")
        
        # 字符集配置
        if "db_charset" not in st.session_state:
            st.session_state.db_charset = os.getenv("DB_CHARSET", "utf8mb4")
        
        # 知识库配置状态
        if "kb_initialized" not in st.session_state:
            st.session_state.kb_initialized = False
        
        if "kb_doc_count" not in st.session_state:
            st.session_state.kb_doc_count = 0
        
        if "kb_last_update" not in st.session_state:
            st.session_state.kb_last_update = "未初始化"
        
        # LLM 和 Embedding 配置
        if "llm_provider" not in st.session_state:
            st.session_state.llm_provider = os.getenv("LLM_PROVIDER", "openai")
        
        if "embedding_provider" not in st.session_state:
            st.session_state.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
        
        if "embedding_model" not in st.session_state:
            embedding_provider = st.session_state.embedding_provider
            if embedding_provider == "huggingface":
                model = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "moka-ai/m3e-base")
            elif embedding_provider == "ollama":
                model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
            else:
                model = "未设置"
            st.session_state.embedding_model = model
        
        # 消息历史
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # 系统初始化标记
        if "system_initialized" not in st.session_state:
            st.session_state.system_initialized = False
    
    @staticmethod
    def update_db_status(host: str, port: int, user: str, password: str, 
                        db_name: str, active: bool, charset: str = "utf8mb4"):
        """
        更新数据库状态
        
        Args:
            host: 主机地址
            port: 端口
            user: 用户名
            password: 密码
            db_name: 数据库名
            active: 是否激活
            charset: 字符集
        """
        st.session_state.db_host = host
        st.session_state.db_port = port
        st.session_state.db_user = user
        st.session_state.db_password = password
        st.session_state.db_name = db_name
        st.session_state.db_charset = charset
        st.session_state.db_config_active = active
        
        # 触发系统更新
        SystemStateManager.trigger_system_update("database")
    
    @staticmethod
    def on_database_connected(host: str, port: int, user: str, password: str, 
                             database: str = "", charset: str = "utf8mb4"):
        """
        数据库连接成功回调 - 与 sidebar_v2.py 兼容
        
        Args:
            host: 主机地址
            port: 端口
            user: 用户名
            password: 密码
            database: 数据库名
            charset: 字符集
            
        Returns:
            (success, message) 元组
        """
        try:
            SystemStateManager.update_db_status(host, port, user, password, database, True, charset)
            return True, f"✅ 数据库已连接: {host}:{port}/{database} (charset: {charset})"
        except Exception as e:
            return False, f"❌ 数据库连接失败: {str(e)}"
    
    @staticmethod
    def update_kb_status(initialized: bool, doc_count: int = 0):
        """
        更新知识库状态
        
        Args:
            initialized: 是否初始化
            doc_count: 文档数量
        """
        st.session_state.kb_initialized = initialized
        st.session_state.kb_doc_count = doc_count
        st.session_state.kb_last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 触发系统更新
        SystemStateManager.trigger_system_update("knowledge_base")
    
    @staticmethod
    def on_knowledge_base_updated(num_docs: int):
        """
        知识库更新成功回调 - 与 kb_manager.py 兼容
        
        Args:
            num_docs: 加载的文档数量
            
        Returns:
            (success, message) 元组
        """
        if num_docs > 0:
            SystemStateManager.update_kb_status(True, num_docs)
            return True, f"✅ 知识库已更新！加载了 {num_docs} 篇文档"
        else:
            return False, "⚠️ 没有成功加载任何文档"
    
    @staticmethod
    def update_agent_mode(mode: str):
        """更新 Agent 工作模式"""
        st.session_state.agent_mode = mode
    
    @staticmethod
    def update_agent(agent):
        """更新 Agent 实例"""
        st.session_state.agent = agent
    
    @staticmethod
    def get_system_status() -> Dict[str, Any]:
        """
        获取完整的系统状态
        
        Returns:
            包含所有系统状态的字典
        """
        return {
            "db_active": st.session_state.get("db_config_active", False),
            "db_host": st.session_state.get("db_host", ""),
            "db_port": st.session_state.get("db_port", 3306),
            "db_charset": st.session_state.get("db_charset", "utf8mb4"),
            "kb_initialized": st.session_state.get("kb_initialized", False),
            "kb_doc_count": st.session_state.get("kb_doc_count", 0),
            "kb_last_update": st.session_state.get("kb_last_update", "未初始化"),
            "agent_mode": st.session_state.get("agent_mode", "纯对话模式"),
            "agent_available": st.session_state.get("agent") is not None,
            "llm_provider": st.session_state.get("llm_provider", "openai"),
            "embedding_provider": st.session_state.get("embedding_provider", "huggingface"),
            "embedding_model": st.session_state.get("embedding_model", "未设置"),
        }
    
    @staticmethod
    def trigger_system_update(reason: str = ""):
        """
        触发系统状态更新并重新初始化 Agent
        
        Args:
            reason: 更新原因（用于日志）
        """
        from agent import get_agent_executor
        
        try:
            # 获取当前状态
            db_active = st.session_state.get("db_config_active", False)
            kb_available = st.session_state.get("kb_initialized", False)
            
            # 重新初始化 Agent（根据资源可用性自动选择模式）
            result = get_agent_executor(db_available=db_active)
            
            if isinstance(result, tuple):
                agent, mode = result
                st.session_state.agent = agent
                st.session_state.agent_mode = mode
            else:
                st.session_state.agent = result
            
            print(f"✅ 系统状态已更新 - 原因: {reason}")
            print(f"   数据库: {'✅ 已连接' if db_active else '⚠️ 未连接'}")
            print(f"   知识库: {'✅ 已初始化' if kb_available else '⚠️ 未初始化'}")
            print(f"   Agent 模式: {st.session_state.agent_mode}")
            
            return True, f"✅ 系统已更新 (模式: {st.session_state.agent_mode})"
        
        except Exception as e:
            error_msg = f"❌ Agent 更新失败: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    @staticmethod
    def update_system_state(trigger: str = "manual"):
        """
        更新系统状态 - trigger_system_update 的别名（向后兼容）
        
        Args:
            trigger: 触发更新的原因
            
        Returns:
            (success, message) 元组
        """
        return SystemStateManager.trigger_system_update(trigger)
    
    @staticmethod
    def print_status():
        """打印系统状态（用于调试）"""
        status = SystemStateManager.get_system_status()
        
        print("\n" + "="*50)
        print("📊 系统状态快照")
        print("="*50)
        
        print("\n🗄️ 数据库:")
        if status["db_active"]:
            print(f"   ✅ 已连接: {status['db_host']}:{status['db_port']}")
            print(f"   🔤 字符集: {status['db_charset']}")
        else:
            print("   ⚠️ 未连接")
        
        print("\n📚 知识库:")
        if status["kb_initialized"]:
            print(f"   ✅ 已初始化: {status['kb_doc_count']} 篇文档")
            print(f"   📅 更新时间: {status['kb_last_update']}")
        else:
            print("   ⚠️ 未初始化")
        
        print("\n🤖 Agent:")
        print(f"   工作模式: {status['agent_mode']}")
        print(f"   状态: {'✅ 可用' if status['agent_available'] else '❌ 不可用'}")
        
        print("\n🔧 配置:")
        print(f"   LLM: {status['llm_provider']}")
        print(f"   Embedding: {status['embedding_provider']}")
        print(f"   Embedding 模型: {status['embedding_model']}")
        
        print("\n" + "="*50 + "\n")


# 创建全局状态管理器实例
system_state_manager = SystemStateManager()
