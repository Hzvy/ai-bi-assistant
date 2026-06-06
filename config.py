"""
项目配置管理 - 支持多模型和动态配置
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """应用配置"""
    
    # App
    APP_NAME = "AI BI Assistant V1"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False") == "True"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # ===== LLM 配置 =====
    # 模型提供者: openai, ollama, anthropic, qwen, deepseek
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    
    # OpenAI 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # 阿里通义千问配置
    QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
    QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-turbo")
    
    # Anthropic Claude 配置
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
    
    # Deepseek 配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    # Ollama 本地模型配置
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
    
    # 兼容旧配置
    LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    LLM_BASE_URL = os.getenv(
        "LLM_BASE_URL",
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    
    # ===== Embedding 配置 =====
    # embedding 提供者: openai, huggingface, ollama, zhipuai, baichuan, deepseek, bgem3
    # 推荐配置:
    #   - 使用 Milvus 混合检索: EMBEDDING_PROVIDER=bgem3 ⭐（最佳效果）
    #   - 使用 ChromaDB: EMBEDDING_PROVIDER=huggingface（兼容性好）
    #   - 需要 API 服务: EMBEDDING_PROVIDER=openai（稳定可靠）
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "bgem3")
    
    # OpenAI Embedding
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # HuggingFace Embedding
    HUGGINGFACE_EMBEDDING_MODEL = os.getenv(
        "HUGGINGFACE_EMBEDDING_MODEL",
        "BAAI/bge-m3"  # 改为 BGE-M3 (1024维，支持混合检索)
    )
    HUGGINGFACE_DEVICE = os.getenv("HUGGINGFACE_DEVICE", "cpu")
    
    # Ollama Embedding
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    
    # 智谱 GLM Embedding
    ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")
    ZHIPUAI_EMBEDDING_MODEL = os.getenv("ZHIPUAI_EMBEDDING_MODEL", "embedding-2")
    
    # 百川 Embedding (兼容旧配置)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    BAICHUAN_EMBEDDING_KEY = os.getenv("BAICHUAN_EMBEDDING_KEY", "")
    
    # Deepseek Embedding
    DEEPSEEK_EMBEDDING_BASE_URL = os.getenv("DEEPSEEK_EMBEDDING_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_EMBEDDING_MODEL = os.getenv("DEEPSEEK_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Database - 可选配置（允许为空）
    DB_HOST = os.getenv("DB_HOST", "")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_USER = os.getenv("DB_USER", "")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "")
    DB_CHARSET = os.getenv("DB_CHARSET", "utf8mb4")
    
    # ChromaDB
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    
    # ===== 向量数据库配置 =====
    # 向量数据库类型: chroma, milvus
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma")
    
    # Milvus 配置
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "ai_bi_kb")
    
    # BGEM3 嵌入模型配置（用于 Milvus 混合检索）
    BGEM3_EMBEDDING_MODEL = os.getenv("BGEM3_EMBEDDING_MODEL", "BAAI/bge-m3")
    BGEM3_DENSE_DIM = int(os.getenv("BGEM3_DENSE_DIM", "1024"))
    BGEM3_USE_FP16 = os.getenv("BGEM3_USE_FP16", "True") == "True"
    
    # ===== RAG 优化配置 =====
    # 启用相关性检查
    RAG_ENABLE_RELEVANCE_CHECK = os.getenv("RAG_ENABLE_RELEVANCE_CHECK", "True") == "True"
    
    # 启用查询转换
    RAG_ENABLE_QUERY_TRANSFORM = os.getenv("RAG_ENABLE_QUERY_TRANSFORM", "True") == "True"
    RAG_QUERY_VARIANTS = int(os.getenv("RAG_QUERY_VARIANTS", "3"))
    
    # 启用 HyDE（假设文档扩展）
    RAG_ENABLE_HYDE = os.getenv("RAG_ENABLE_HYDE", "False") == "True"
    
    # 启用重排序
    RAG_ENABLE_RERANK = os.getenv("RAG_ENABLE_RERANK", "True") == "True"
    
    # 检索参数
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
    RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))
    
    # ===== RAG 高级策略配置（TOP 5 推荐方案）=====
    # 1. 融合检索 (Hybrid RAG) - 向量 + BM25
    RAG_ENABLE_HYBRID = os.getenv("RAG_ENABLE_HYBRID", "True") == "True"
    RAG_HYBRID_SEMANTIC_WEIGHT = float(os.getenv("RAG_HYBRID_SEMANTIC_WEIGHT", "0.6"))  # 语义搜索权重
    RAG_HYBRID_USE_RRF = os.getenv("RAG_HYBRID_USE_RRF", "True") == "True"  # 使用倒数排名融合
    
    # 2. 上下文增强 (Context Enriched RAG) - 检索相邻块
    RAG_ENABLE_CONTEXT_ENRICHED = os.getenv("RAG_ENABLE_CONTEXT_ENRICHED", "True") == "True"
    RAG_CONTEXT_RADIUS = int(os.getenv("RAG_CONTEXT_RADIUS", "1"))  # 上下文半径（前后各N个块）
    RAG_CONTEXT_MERGE = os.getenv("RAG_CONTEXT_MERGE", "True") == "True"  # 是否合并上下文到文本
    
    # 3. 自适应 RAG (Adaptive RAG) - 智能策略选择
    RAG_ENABLE_ADAPTIVE = os.getenv("RAG_ENABLE_ADAPTIVE", "False") == "True"
    # 注意: 自适应RAG会根据查询类型自动选择策略，可能会增加成本
    
    # 4. 策略组合模式
    RAG_STRATEGY_MODE = os.getenv("RAG_STRATEGY_MODE", "simple")
    # 可选值:
    #   - "simple": 基础向量搜索（最快，成本最低）⭐ 默认
    #   - "hybrid": 融合检索（推荐，平衡性能和成本）
    #   - "enhanced": 融合 + 上下文增强（高质量）
    #   - "adaptive": 自适应策略（智能选择，可能成本较高）
    #   - "full": 全流程优化（最高质量，成本最高）
    
    # ===== 文本处理配置 =====
    # 是否启用文本清洗（去除页码、URL等噪音）
    TEXT_CLEANING_ENABLED = os.getenv("TEXT_CLEANING_ENABLED", "True").lower() == "true"
    
    # 是否启用智能分割（根据文档结构自动分割）
    SMART_SPLIT_ENABLED = os.getenv("SMART_SPLIT_ENABLED", "True").lower() == "true"
    
    # 自定义清洗模式（逗号分隔的固定字符串）
    CUSTOM_CLEAN_PATTERNS = os.getenv("CUSTOM_CLEAN_PATTERNS", "").split(",") if os.getenv("CUSTOM_CLEAN_PATTERNS") else []
    
    @classmethod
    def get_db_config(cls, from_session=True):
        """
        获取数据库配置字典
        
        功能说明:
            支持双层配置优先级，优先使用运行时配置（Session State），
            其次使用环境变量配置。这样设计允许用户在不重启应用的情况下
            动态修改数据库连接。
        
        参数:
            from_session (bool): 是否优先从 Streamlit Session State 获取配置
                               - True: 优先使用侧边栏动态配置（默认）
                               - False: 仅使用环境变量/.env文件配置
        
        返回:
            dict: 数据库配置字典，包含以下键:
                - host (str): 数据库主机地址
                - port (int): 数据库端口
                - user (str): 数据库用户名
                - password (str): 数据库密码
                - database (str): 数据库名称
                - charset (str): 字符集编码
        
        使用示例:
            >>> config = Config.get_db_config()
            >>> connection = pymysql.connect(**config)
        
        配置优先级:
            1. Session State（侧边栏配置）- 最高优先级
            2. 环境变量（.env 文件）
            3. 默认值（类属性）
        """
        import streamlit as st
        
        # 优先使用 session 中的配置（用户通过侧栏设置）
        if from_session and "db_config_active" in st.session_state:
            if st.session_state.get("db_config_active", False):
                return {
                    "host": st.session_state.get("db_host", ""),
                    "port": st.session_state.get("db_port", 3306),
                    "user": st.session_state.get("db_user", ""),
                    "password": st.session_state.get("db_password", ""),
                    "database": st.session_state.get("db_name", ""),
                    "charset": st.session_state.get("db_charset", "utf8mb4")
                }
        
        # 否则使用环境变量或默认值
        return {
            "host": cls.DB_HOST,
            "port": cls.DB_PORT,
            "user": cls.DB_USER,
            "password": cls.DB_PASSWORD,
            "database": cls.DB_NAME,
            "charset": cls.DB_CHARSET
        }
    
    @classmethod
    def has_db_config(cls, from_session=True):
        """
        检查是否有有效的数据库配置
        
        功能说明:
            验证数据库配置是否完整可用。用于 Agent 初始化时判断
            是否启用数据库查询功能，避免在无配置时调用数据库工具导致错误。
        
        参数:
            from_session (bool): 是否检查 Session State 配置
                               - True: 优先检查侧边栏配置状态（默认）
                               - False: 仅检查环境变量配置
        
        返回:
            bool: 数据库配置是否有效
                - True: 配置完整，可以连接数据库
                - False: 配置缺失，不应启用数据库功能
        
        验证逻辑:
            - Session模式: 检查 db_config_active 标志
            - 环境变量模式: 检查 host 和 user 是否非空
        
        使用示例:
            >>> if Config.has_db_config():
            >>>     agent = create_agent(db_available=True)
            >>> else:
            >>>     agent = create_agent(db_available=False)
        """
        import streamlit as st
        
        if from_session and "db_config_active" in st.session_state:
            return st.session_state.get("db_config_active", False)
        
        return bool(cls.DB_HOST and cls.DB_USER)

config = Config()
