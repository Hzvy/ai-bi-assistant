"""
LLM 管理器 - 统一的 LLM 调用接口

功能说明:
    提供全局单例 LLM 管理器，封装 LLMFactory 的复杂性。
    所有模块通过此管理器获取 LLM 实例，实现提供者切换和配置管理。

设计模式:
    - 单例模式（Singleton）：全局唯一实例
    - 委托模式（Delegation）：委托给 llm_factory.LLMManager
    - 外观模式（Facade）：简化 LLMFactory 接口

架构层次:
    tools_rag.py → llm_manager (本模块) → llm_factory → OpenAIProvider/QwenProvider/...

使用示例:
    >>> from tools.llm_manager import llm_manager
    >>> llm = llm_manager.get_llm()
    >>> response = llm.invoke("你好")
    >>> print(response.content)
"""
from tools.llm_factory import llm_manager as factory_llm_manager
from config import config
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class LLMManager:
    """
    LLM 管理器单例类
    
    功能说明:
        全局 LLM 访问点，委托给 llm_factory 的实际实现。
        提供简洁的 API 供其他模块调用。
    
    设计理由:
        - 解耦业务代码与 LLMFactory 实现细节
        - 统一日志记录和错误处理
        - 支持运行时提供者切换
        - 便于单元测试 mock
    
    单例实现:
        使用 __new__ 方法确保全局唯一实例
        >>> llm_mgr1 = LLMManager()
        >>> llm_mgr2 = LLMManager()
        >>> assert llm_mgr1 is llm_mgr2  # 同一实例
    
    委托关系:
        所有方法调用转发给 llm_factory.llm_manager
        本类不持有 LLM 实例，仅作为代理层
    
    线程安全:
        当前实现非线程安全，适用于单线程 Streamlit 应用
        多线程环境需添加锁机制
    """
    
    _instance = None
    
    def __new__(cls):
        """
        单例模式实现（__new__ 方法）
        
        功能说明:
            确保 LLMManager 全局仅有一个实例。
            首次创建时初始化，后续返回已有实例。
        
        返回:
            LLMManager: 全局唯一的管理器实例
        
        工作原理:
            1. 检查 cls._instance 是否为 None
            2. None → 调用父类 __new__ 创建实例
            3. 非 None → 直接返回已有实例
            4. 后续所有 LLMManager() 调用返回同一对象
        
        使用示例:
            >>> mgr1 = LLMManager()
            >>> mgr2 = LLMManager()
            >>> print(mgr1 is mgr2)
            True
        
        注意事项:
            - __new__ 在 __init__ 之前调用
            - 本类无 __init__ 方法，使用默认初始化
            - 实例属性存储在 _instance 中
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_llm(self):
        """
        获取当前配置的 LLM 实例
        
        功能说明:
            返回根据配置创建的 LLM 对象（ChatOpenAI, Tongyi, Deepseek 等）。
            会根据环境变量或 session_state 动态选择提供者。
        
        返回:
            BaseChatModel: LangChain 兼容的 LLM 实例
                - ChatOpenAI (OpenAI GPT)
                - Tongyi (阿里千问)
                - ChatZhipuAI (智谱 GLM)
                - Ollama (本地模型)
                - 等
        
        配置来源:
            优先级: Session State > 环境变量 > config.py
            - st.session_state.llm_provider (UI 选择)
            - os.getenv("LLM_PROVIDER")
            - config.LLM_PROVIDER
        
        委托调用:
            实际调用 llm_factory.llm_manager.get_llm()
            → LLMFactory.get_llm()
            → 对应 Provider.get_llm()
        
        使用示例:
            >>> llm = llm_manager.get_llm()
            >>> print(type(llm).__name__)
            ChatOpenAI
            
            >>> response = llm.invoke("写一个 Python 函数")
            >>> print(response.content)
            def hello(): ...
        
        注意事项:
            - 每次调用可能返回不同实例（取决于工厂实现）
            - LLM 实例包含 API 密钥等敏感信息，勿打印
            - 调用失败会抛出异常（API key 未配置等）
        
        异常:
            - ValueError: 提供者未配置或 API 密钥缺失
            - ImportError: 依赖库未安装（如 langchain-openai）
        """
        return factory_llm_manager.get_llm()
    
    def invoke(self, prompt):
        """
        便捷方法：直接调用 LLM
        
        功能说明:
            简化调用流程，无需先获取 LLM 实例。
            内部自动获取 LLM 并执行 invoke()。
        
        参数:
            prompt (str): 提示词文本
                示例: "解释什么是向量数据库"
        
        返回:
            AIMessage: LangChain 消息对象
                - content (str): LLM 生成的文本
                - response_metadata (dict): 响应元数据
        
        使用示例:
            >>> response = llm_manager.invoke("什么是 RAG?")
            >>> print(response.content)
            RAG (Retrieval-Augmented Generation) 是一种...
        
        等价调用:
            ```python
            # 方式 1: 使用 invoke()
            response = llm_manager.invoke(prompt)
            
            # 方式 2: 手动获取 LLM
            llm = llm_manager.get_llm()
            response = llm.invoke(prompt)
            ```
        
        注意事项:
            - 每次调用都会重新获取 LLM（可能产生新实例）
            - 频繁调用建议先缓存 LLM 实例
            - 不支持流式响应，需要可用 get_llm().stream()
        """
        llm = self.get_llm()
        return llm.invoke(prompt)
    
    def switch_provider(self, provider_name: str):
        """
        切换 LLM 提供者
        
        功能说明:
            运行时动态切换 LLM 后端（如从 OpenAI 切换到 Qwen）。
            主要用于 UI 交互切换。
        
        参数:
            provider_name (str): 提供者名称
                支持值:
                - "openai": OpenAI GPT-3.5/GPT-4
                - "qwen": 阿里千问
                - "deepseek": Deepseek Chat
                - "zhipuai": 智谱 GLM
                - "claude": Anthropic Claude
                - "ollama": 本地 Ollama 模型
        
        返回:
            bool: 切换是否成功
                - True: 成功切换
                - False: 切换失败（提供者不存在或配置错误）
        
        工作流程:
            1. 验证 provider_name 有效性
            2. 调用 factory_llm_manager.switch_provider()
            3. 更新内部状态
            4. 记录日志
            5. 返回切换结果
        
        使用示例:
            >>> # 从 OpenAI 切换到千问
            >>> success = llm_manager.switch_provider("qwen")
            >>> if success:
            ...     print(f"当前提供者: {llm_manager.get_current_provider()}")
            当前提供者: qwen
            
            >>> # UI 集成
            >>> provider = st.selectbox("选择模型", ["openai", "qwen"])
            >>> llm_manager.switch_provider(provider)
        
        注意事项:
            - 切换后下次 get_llm() 会返回新提供者实例
            - 不影响已创建的 LLM 实例
            - 需要新提供者的 API 密钥已配置
            - 切换失败会保持原提供者
        
        异常:
            切换失败不会抛出异常，返回 False
        """
        return factory_llm_manager.switch_provider(provider_name)
    
    def get_current_provider(self):
        """
        获取当前使用的 LLM 提供者名称
        
        功能说明:
            查询当前激活的 LLM 后端提供者。
            用于 UI 显示、日志记录、调试诊断。
        
        返回:
            str: 提供者名称
                示例: "openai", "qwen", "deepseek"
        
        使用示例:
            >>> provider = llm_manager.get_current_provider()
            >>> print(f"当前使用: {provider}")
            当前使用: openai
            
            >>> # UI 状态显示
            >>> st.info(f"🤖 当前模型: {llm_manager.get_current_provider()}")
        
        调试用途:
            ```python
            logger.info(f"LLM Provider: {llm_manager.get_current_provider()}")
            logger.info(f"LLM Instance: {type(llm_manager.get_llm()).__name__}")
            ```
        """
        return factory_llm_manager.get_current_provider()

# 全局 LLM 管理器实例
llm_manager = LLMManager()
logger.info(f"✅ LLM 管理器初始化成功，使用提供者: {llm_manager.get_current_provider()}")
