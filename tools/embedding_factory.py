"""
Embedding 管理器 - 支持多种 embedding 模型

功能说明:
    提供统一的嵌入模型接口，支持多家厂商的向量化服务。
    使用工厂模式和提供者模式，实现灵活的模型切换。

支持的模型提供者:
    1. OpenAI: text-embedding-3-small/ada-002
       - 优点: 效果好，API 稳定
       - 缺点: 需付费，有 API 调用限制
       - 向量维度: 1536 (ada-002), 1536 (3-small)
    
    2. HuggingFace: m3e-base, bge-base-zh-v1.5
       - 优点: 免费，本地运行，中文优化
       - 缺点: 首次下载模型较慢
       - 向量维度: 768 (m3e), 768 (bge)
       - 推荐: moka-ai/m3e-base（中文效果好）
    
    3. 智谱 GLM: embedding-2/embedding-3
       - 优点: 国内服务，中文友好
       - 缺点: embedding-3 需付费
       - 向量维度: 1024
    
    4. Deepseek: text-embedding-3-small
       - 优点: 兼容 OpenAI API
       - 缺点: 需配置 API key
       - 向量维度: 1536
    
    5. Ollama: nomic-embed-text
       - 优点: 完全本地，隐私保护
       - 缺点: 需要本地部署 Ollama
       - 向量维度: 768

设计模式:
    - 工厂模式（Factory）：EmbeddingFactory 创建实例
    - 策略模式（Strategy）：BaseEmbeddingProvider 抽象基类
    - 单例模式（Singleton）：全局唯一 embedding 实例

架构:
    tools_rag.py → EmbeddingFactory.create_embedding() 
    → 对应 Provider → LangChain Embeddings

使用示例:
    >>> from tools.embedding_factory import EmbeddingFactory
    >>> embeddings = EmbeddingFactory.create_embedding()
    >>> vector = embeddings.embed_query("测试文本")
    >>> print(len(vector))  # 向量维度
    768
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from config import config
import os


class BaseEmbeddingProvider(ABC):
    """
    Embedding 提供者抽象基类
    
    功能说明:
        定义所有 Embedding 提供者必须实现的接口。
        使用抽象基类确保接口一致性。
    
    设计模式:
        策略模式（Strategy Pattern）
        - 抽象策略: BaseEmbeddingProvider
        - 具体策略: OpenAIEmbeddingProvider, HuggingFaceEmbeddingProvider...
        - 上下文: EmbeddingFactory
    
    子类职责:
        1. 实现 get_embedding() 方法
        2. 读取配置（环境变量或 config.py）
        3. 创建并返回 LangChain Embeddings 实例
    
    使用示例:
        >>> class CustomProvider(BaseEmbeddingProvider):
        ...     def get_embedding(self, **kwargs):
        ...         return CustomEmbeddings(...)
    """
    
    @abstractmethod
    def get_embedding(self, **kwargs):
        """
        获取 embedding 实例（抽象方法）
        
        功能说明:
            子类必须实现此方法，返回对应的 LangChain Embeddings 对象。
        
        参数:
            **kwargs: 可选的额外配置参数
        
        返回:
            Embeddings: LangChain 兼容的 Embeddings 实例
                - 必须实现 embed_query(text) 方法
                - 必须实现 embed_documents(texts) 方法
        
        抛出:
            NotImplementedError: 子类未实现此方法
        """
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI Embedding 提供者"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", config.LLM_API_KEY)
        self.base_url = os.getenv("OPENAI_BASE_URL", config.LLM_BASE_URL)
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    def get_embedding(self, **kwargs):
        """获取 OpenAI Embedding"""
        return OpenAIEmbeddings(
            api_key=self.api_key,
            model=self.model,
        )


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """
    HuggingFace Embedding 提供者
    
    功能说明:
        使用 HuggingFace 模型在本地生成向量嵌入。
        支持中文优化模型（m3e-base, bge-base-zh）。
    
    推荐模型:
        1. moka-ai/m3e-base (默认)
           - 向量维度: 768
           - 优势: 中文效果好，模型小（~400MB）
           - 适合: 中文为主的知识库
        
        2. BAAI/bge-base-zh-v1.5
           - 向量维度: 768
           - 优势: 中文语义理解强
           - 适合: 技术文档检索
        
        3. BAAI/bge-m3
           - 向量维度: 1024（Dense）+ Sparse
           - 优势: 混合检索，召回率高
           - 适合: 生产环境（需 Milvus）
    
    配置参数:
        - HUGGINGFACE_EMBEDDING_MODEL: 模型名称
        - HUGGINGFACE_DEVICE: 运行设备（cpu/cuda/mps）
    
    模型下载:
        首次运行会自动从 HuggingFace Hub 下载模型
        存储位置: ~/.cache/huggingface/hub/
        下载时间: 约 5-10 分钟（取决于网速）
    
    设备选择:
        - cpu: 兼容性最好，速度较慢
        - cuda: NVIDIA GPU 加速（需 CUDA）
        - mps: Apple Silicon GPU 加速（M1/M2）
    
    性能:
        - CPU 嵌入速度: ~100 文档/秒
        - GPU 嵌入速度: ~500 文档/秒
        - 内存占用: 约 1-2GB
    
    使用示例:
        >>> provider = HuggingFaceEmbeddingProvider()
        >>> embeddings = provider.get_embedding()
        >>> vector = embeddings.embed_query("测试文本")
        >>> print(len(vector))
        768
    """
    
    def __init__(self):
        """
        初始化 HuggingFace 提供者配置
        
        读取配置:
            - HUGGINGFACE_EMBEDDING_MODEL: 模型路径
            - HUGGINGFACE_DEVICE: 计算设备
        """
        self.model = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "moka-ai/m3e-base")
        self.device = os.getenv("HUGGINGFACE_DEVICE", "cpu")
    
    def get_embedding(self, **kwargs):
        """
        获取 HuggingFace Embedding 实例
        
        返回:
            HuggingFaceEmbeddings: 配置好的嵌入模型
        
        配置说明:
            - model_name: HuggingFace 模型 ID
            - model_kwargs: {"device": "cpu/cuda/mps"}
            - encode_kwargs:
              - normalize_embeddings: True（归一化向量）
              - batch_size: 32（批处理大小）
        
        归一化说明:
            normalize_embeddings=True 会将向量归一化到单位长度
            好处: 可以直接使用点积计算余弦相似度
            公式: cosine_sim = dot(v1, v2) when ||v1|| = ||v2|| = 1
        """
        return HuggingFaceEmbeddings(
            model_name=self.model,
            model_kwargs={"device": self.device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            }
        )


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Ollama 本地 Embedding 提供者
    
    功能说明:
        使用 Ollama 在本地运行嵌入模型，完全离线，隐私保护。
        需要先在本地安装并启动 Ollama 服务。
    
    推荐模型:
        - nomic-embed-text (默认)
          - 向量维度: 768
          - 优势: 轻量级，速度快
          - 安装: ollama pull nomic-embed-text
        
        - mxbai-embed-large
          - 向量维度: 1024
          - 优势: 效果更好
          - 安装: ollama pull mxbai-embed-large
    
    安装步骤:
        1. 安装 Ollama: https://ollama.ai
        2. 拉取模型: ollama pull nomic-embed-text
        3. 启动服务: ollama serve（默认端口 11434）
        4. 配置环境变量或使用默认值
    
    配置参数:
        - OLLAMA_EMBEDDING_MODEL: 模型名称
        - OLLAMA_BASE_URL: Ollama 服务地址
    
    优势:
        ✅ 完全本地化，数据不出本地
        ✅ 无需 API 密钥和网络连接
        ✅ 无调用次数限制
        ✅ 响应速度快
    
    劣势:
        ❌ 需要本地安装 Ollama
        ❌ 模型效果可能不如商业服务
        ❌ 占用本地资源（CPU/内存）
    
    使用示例:
        >>> # 确保 Ollama 服务运行中
        >>> provider = OllamaEmbeddingProvider()
        >>> embeddings = provider.get_embedding()
        >>> vector = embeddings.embed_query("本地嵌入测试")
    """
    
    def __init__(self):
        self.model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    def get_embedding(self, **kwargs):
        """
        获取 Ollama Embedding 实例
        
        返回:
            OllamaEmbeddings: 配置好的 Ollama 嵌入实例
        
        连接检查:
            如果 Ollama 服务未运行，会在调用时报错
            建议: 使用前先检查 http://localhost:11434 是否可访问
        """
        return OllamaEmbeddings(
            model=self.model,
            base_url=self.base_url,
        )


class ZhipuEmbeddingProvider(BaseEmbeddingProvider):
    """智谱 GLM Embedding 提供者 - 支持 embedding-3"""
    
    def __init__(self):
        try:
            from langchain_community.embeddings import ZhipuAIEmbeddings
            self.ZhipuAIEmbeddings = ZhipuAIEmbeddings
        except ImportError:
            raise ImportError("请先安装 zhipuai: pip install zhipuai")
        
        self.api_key = os.getenv("ZHIPUAI_API_KEY")
        # 支持的模型: embedding-2 (免费), embedding-3 (付费但更好)
        self.model = os.getenv("ZHIPUAI_EMBEDDING_MODEL", "embedding-3")
        self.api_base = "https://open.bigmodel.cn/api/paas/v4"  # 智谱 API 端点
    
    def get_embedding(self, **kwargs):
        """获取智谱 Embedding"""
        try:
            # 尝试使用完整配置
            embeddings = self.ZhipuAIEmbeddings(
                api_key=self.api_key,
                model=self.model,
            )
            print(f"✅ 智谱 Embedding 已初始化: {self.model}")
            return embeddings
        except Exception as e:
            print(f"⚠️ 智谱 Embedding 配置错误: {str(e)}")
            print(f"   API Key: {self.api_key[:20]}..." if self.api_key else "   API Key: 未设置")
            print(f"   Model: {self.model}")
            print(f"   API Base: {self.api_base}")
            raise


class BaichuanEmbeddingProvider(BaseEmbeddingProvider):
    """百川 Embedding 提供者"""
    
    def __init__(self):
        try:
            from langchain_community.embeddings import BaichuanTextEmbeddings
            self.BaichuanTextEmbeddings = BaichuanTextEmbeddings
        except ImportError:
            raise ImportError("请先安装 baichuan-sdk: pip install baichuan-sdk")
        
        self.api_key = os.getenv("BAICHUAN_EMBEDDING_KEY")
    
    def get_embedding(self, **kwargs):
        """获取百川 Embedding"""
        return self.BaichuanTextEmbeddings(
            api_key=self.api_key,
        )


class DeepseekEmbeddingProvider(BaseEmbeddingProvider):
    """Deepseek Embedding 提供者"""
    
    def __init__(self):
        try:
            from langchain_openai import OpenAIEmbeddings
            self.OpenAIEmbeddings = OpenAIEmbeddings
        except ImportError:
            raise ImportError("请先安装 openai: pip install openai langchain-openai")
        
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_EMBEDDING_BASE_URL", "https://api.deepseek.com/v1")
        self.model = os.getenv("DEEPSEEK_EMBEDDING_MODEL", "text-embedding-3-small")
    
    def get_embedding(self, **kwargs):
        """获取 Deepseek Embedding"""
        return self.OpenAIEmbeddings(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
        )


class BGEM3EmbeddingProvider(BaseEmbeddingProvider):
    """
    BGEM3 Embedding 提供者（推荐用于 Milvus 混合检索）⭐
    
    功能说明:
        使用 BAAI/bge-m3 模型生成 Dense + Sparse 双向量。
        专为 Milvus 混合检索优化，支持 RRF 融合算法。
    
    核心优势:
        ✅ 混合向量: Dense (1024维) + Sparse (动态维度)
        ✅ 召回率高: 相比单向量提升 10-15%
        ✅ Milvus 原生支持: 并行检索 + RRF 融合
        ✅ 中英文优化: 专为中文语义优化
    
    技术规格:
        - Dense 向量维度: 1024
        - Sparse 向量: 动态维度（仅存储非零值）
        - 模型大小: ~2.3GB
        - 下载位置: ~/.cache/huggingface/
        - 首次加载: 约 5-10 分钟
    
    配置参数:
        - BGEM3_EMBEDDING_MODEL: 模型路径（默认 "BAAI/bge-m3"）
        - BGEM3_USE_FP16: 是否使用 FP16 加速（默认 True）
        - BGEM3_DENSE_DIM: Dense 向量维度（默认 1024）
    
    性能数据:
        - 编码速度: ~200 文档/秒（GPU）
        - 内存占用: ~2.3GB（模型）+ 1GB（运行时）
        - 检索延迟: <100ms（百万级数据 + Milvus）
        - 召回率: 95%+（混合检索）
    
    适用场景:
        ✅ 使用 Milvus 向量数据库
        ✅ 需要高召回率（>90%）
        ✅ 中英文混合文档
        ✅ 生产环境部署
    
    不适用场景:
        ❌ 使用 ChromaDB/Faiss（不支持 Sparse）
        ❌ 内存受限环境（<4GB）
        ❌ 纯英文文档（OpenAI embedding 更好）
    
    与 HuggingFace m3e-base 对比:
        | 特性              | BGEM3           | m3e-base        |
        |-------------------|-----------------|-----------------|
        | Dense 维度        | 1024            | 768             |
        | Sparse 支持       | ✅ 是           | ❌ 否           |
        | 模型大小          | 2.3GB           | 400MB           |
        | 召回率            | 95%+            | 85-90%          |
        | Milvus 混合检索   | ✅ 原生支持     | ❌ 不支持       |
        | 适合场景          | 生产环境        | 开发/测试       |
    
    使用示例:
        >>> provider = BGEM3EmbeddingProvider()
        >>> embeddings = provider.get_embedding()
        >>> # 注意：返回的是 BGEM3FlagModel，不是 LangChain Embeddings
        >>> result = embeddings.encode(
        ...     ["测试文本"],
        ...     return_dense=True,
        ...     return_sparse=True
        ... )
        >>> print(f"Dense 维度: {len(result['dense_vecs'][0])}")
        Dense 维度: 1024
        >>> print(f"Sparse 非零元素: {len(result['lexical_weights'][0])}")
        Sparse 非零元素: 18
    
    注意事项:
        ⚠️ 返回的是 BGEM3FlagModel，而非 LangChain Embeddings
        ⚠️ API 与其他提供者不同（使用 encode() 而非 embed_query()）
        ⚠️ 需要配合 MilvusManager 使用，不能单独用于 ChromaDB
        ⚠️ 首次运行需下载模型（~2.3GB）
    
    兼容性说明:
        由于 BGEM3 API 特殊性，此提供者返回的不是标准 LangChain Embeddings。
        主要用于 MilvusManager 内部调用，不建议直接在其他场景使用。
    """
    
    def __init__(self):
        """
        初始化 BGEM3 提供者配置
        
        读取配置:
            - BGEM3_EMBEDDING_MODEL: 模型路径
            - BGEM3_USE_FP16: FP16 加速开关
        """
        self.model_path = os.getenv("BGEM3_EMBEDDING_MODEL", "BAAI/bge-m3")
        self.use_fp16 = os.getenv("BGEM3_USE_FP16", "True") == "True"
    
    def get_embedding(self, **kwargs):
        """
        获取 BGEM3 Embedding 实例
        
        返回:
            BGEM3FlagModel: BGE-M3 模型实例
        
        注意:
            返回类型不是 LangChain Embeddings！
            使用方法:
            - embeddings.encode(texts, return_dense=True, return_sparse=True)
            - 不支持 embed_query() 或 embed_documents()
        
        配置说明:
            - use_fp16=True: 使用 FP16 精度
              - 优点: 速度提升 ~2x，显存减半
              - 缺点: 精度损失 < 0.1%（可忽略）
        
        异常:
            - ImportError: FlagEmbedding 未安装
            - RuntimeError: 模型加载失败
        """
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "请先安装 FlagEmbedding: pip install FlagEmbedding\n"
                "或者: pip install -U FlagEmbedding"
            )
        
        print(f"📦 加载 BGEM3 模型: {self.model_path}")
        print(f"   FP16 加速: {'启用' if self.use_fp16 else '禁用'}")
        
        return BGEM3FlagModel(
            self.model_path,
            use_fp16=self.use_fp16,
            **kwargs
        )


class EmbeddingFactory:
    """
    Embedding 工厂类（核心入口）
    
    功能说明:
        统一的嵌入模型创建接口，管理所有提供者。
        支持注册自定义提供者，实现提供者热插拔。
    
    设计模式:
        工厂模式（Factory Pattern）
        - 简单工厂: create_embedding() 根据配置创建实例
        - 注册机制: register_provider() 支持扩展
    
    提供者注册表:
        _providers 字典存储所有可用提供者
        key: 提供者名称（字符串）
        value: 提供者类（BaseEmbeddingProvider 子类）
    
    使用场景:
        1. **RAG 系统初始化**
           tools_rag.py 通过此工厂获取嵌入模型
        
        2. **知识库构建**
           kb_loader.py 使用此工厂向量化文档
        
        3. **动态切换提供者**
           UI 选择不同嵌入模型
    
    扩展示例:
        >>> # 注册自定义提供者
        >>> class MyProvider(BaseEmbeddingProvider):
        ...     def get_embedding(self):
        ...         return MyCustomEmbeddings()
        >>> EmbeddingFactory.register_provider("custom", MyProvider)
        >>> embeddings = EmbeddingFactory.create_embedding("custom")
    """
    
    _providers = {
        "openai": OpenAIEmbeddingProvider,
        "huggingface": HuggingFaceEmbeddingProvider,
        "ollama": OllamaEmbeddingProvider,
        "zhipuai": ZhipuEmbeddingProvider,
        "baichuan": BaichuanEmbeddingProvider,
        "deepseek": DeepseekEmbeddingProvider,
        "bgem3": BGEM3EmbeddingProvider,  # ⭐ Milvus 混合检索推荐
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class):
        """
        注册新的 embedding 提供者
        
        功能说明:
            动态添加自定义嵌入模型提供者到工厂。
            实现提供者插件化扩展。
        
        参数:
            name (str): 提供者唯一标识符
                示例: "custom", "aliyun", "tencent"
            provider_class (Type[BaseEmbeddingProvider]): 提供者类
                必须继承 BaseEmbeddingProvider
        
        使用示例:
            >>> class AliyunProvider(BaseEmbeddingProvider):
            ...     def get_embedding(self):
            ...         return AliyunEmbeddings(...)
            >>> EmbeddingFactory.register_provider("aliyun", AliyunProvider)
        
        注意事项:
            - name 重复会覆盖已有提供者
            - provider_class 必须可实例化
            - 不验证类是否继承 BaseEmbeddingProvider
        """
        cls._providers[name] = provider_class
    
    @classmethod
    def get_provider(cls, provider_name: str) -> BaseEmbeddingProvider:
        """
        获取 embedding 提供者实例
        
        功能说明:
            根据提供者名称创建对应的提供者对象。
            验证提供者是否已注册。
        
        参数:
            provider_name (str): 提供者名称
                支持值: "openai", "huggingface", "ollama", "zhipuai", etc.
        
        返回:
            BaseEmbeddingProvider: 提供者实例
        
        异常:
            ValueError: 提供者名称不存在
                错误信息包含所有支持的提供者列表
        
        使用示例:
            >>> provider = EmbeddingFactory.get_provider("huggingface")
            >>> embeddings = provider.get_embedding()
        """
        if provider_name not in cls._providers:
            raise ValueError(
                f"不支持的 embedding 提供者: {provider_name}\n"
                f"支持的提供者: {list(cls._providers.keys())}"
            )
        return cls._providers[provider_name]()
    
    @classmethod
    def create_embedding(cls, provider_name: Optional[str] = None, **kwargs):
        """
        创建 Embedding 实例（主要入口方法）
        
        功能说明:
            一站式创建嵌入模型，自动读取配置并返回实例。
            所有模块通过此方法获取 Embeddings。
        
        参数:
            provider_name (str, optional): 提供者名称
                - 指定: 使用指定的提供者
                - None: 读取 EMBEDDING_PROVIDER 环境变量
                  默认: "huggingface"
            **kwargs: 传递给提供者的额外参数
        
        返回:
            Embeddings: LangChain 兼容的嵌入模型实例
        
        配置优先级:
            1. provider_name 参数
            2. EMBEDDING_PROVIDER 环境变量
            3. "huggingface" 默认值
        
        使用示例:
            >>> # 使用默认提供者（HuggingFace）
            >>> embeddings = EmbeddingFactory.create_embedding()
            
            >>> # 指定提供者
            >>> embeddings = EmbeddingFactory.create_embedding("openai")
            
            >>> # 传递额外参数
            >>> embeddings = EmbeddingFactory.create_embedding(
            ...     "huggingface",
            ...     device="cuda"
            ... )
        
        调用链:
            create_embedding() → get_provider() → Provider.__init__()
            → Provider.get_embedding() → LangChain Embeddings
        
        异常:
            - ValueError: 提供者不存在
            - ImportError: 依赖库未安装
            - RuntimeError: API 密钥未配置
        """
        # 如果没有指定提供者，使用配置中的默认提供者
        if provider_name is None:
            provider_name = os.getenv("EMBEDDING_PROVIDER", "bgem3")
        
        provider = cls.get_provider(provider_name)
        return provider.get_embedding(**kwargs)


class EmbeddingManager:
    """
    Embedding 管理器（单例模式）
    
    功能说明:
        全局嵌入模型管理器，缓存 Embedding 实例。
        避免重复创建嵌入模型，提升性能。
    
    设计模式:
        单例模式（Singleton）
        - 全局唯一实例
        - 缓存 Embedding 对象
        - 提供者变更时自动重新初始化
    
    缓存策略:
        - 首次调用: 创建并缓存 Embedding
        - 提供者未变: 返回缓存实例
        - 提供者变更: 重新创建并更新缓存
    
    使用场景:
        适合长期运行的应用（如 Web 服务）
        不适合频繁切换提供者的场景
    
    性能优化:
        - HuggingFace 模型加载较慢（~3-5秒）
        - 缓存后调用 <1ms
        - 内存占用: 约 1-2GB（取决于模型）
    
    使用示例:
        >>> manager = EmbeddingManager()
        >>> embeddings = manager.get_embedding()
        >>> # 后续调用返回缓存实例
        >>> embeddings2 = manager.get_embedding()
        >>> assert embeddings is embeddings2
    """
    
    _instance = None
    _embedding = None
    _current_provider = None
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化管理器"""
        self._initialize()
    
    def _initialize(self):
        """
        初始化 Embedding 实例
        
        功能说明:
            检查提供者是否变更，按需重新创建 Embedding。
        
        逻辑:
            - 当前提供者 != 缓存提供者 → 重新创建
            - 提供者相同 → 跳过，使用缓存
        """
        provider_name = os.getenv("EMBEDDING_PROVIDER", "huggingface")
        if self._current_provider != provider_name:
            self._embedding = EmbeddingFactory.create_embedding(provider_name)
            self._current_provider = provider_name
    
    def get_embedding(self, **kwargs):
        """
        获取 Embedding 实例
        
        功能说明:
            返回缓存的 Embedding，必要时重新初始化。
        
        返回:
            Embeddings: 嵌入模型实例
        
        使用示例:
            >>> embedding_mgr = EmbeddingManager()
            >>> emb = embedding_mgr.get_embedding()
            >>> vector = emb.embed_query("测试")
        """
        self._initialize()
        return self._embedding
    
    def switch_provider(self, provider_name: str, **kwargs):
        """切换 Embedding 提供者"""
        self._embedding = EmbeddingFactory.create_embedding(provider_name, **kwargs)
        self._current_provider = provider_name
    
    def get_available_providers(self):
        """获取所有可用的提供者"""
        return list(EmbeddingFactory._providers.keys())
    
    def get_current_provider(self):
        """获取当前提供者"""
        return self._current_provider


# 全局 Embedding 管理器实例
embedding_manager = EmbeddingManager()
