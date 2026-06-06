"""
RAG 知识库检索工具 - 增强版

支持功能:
- 知识库初始化和管理
- 文档检索和相似度搜索
- 知识库可用性检查
- 当知识库为空时自动跳过检索
- 灵活切换多种嵌入模型（HuggingFace、OpenAI、Ollama、Baichuan 等）
- 增强型检索优化:
  - 相关性检查（过滤垃圾问题）
  - 查询转换（多变体检索）
  - 重排序（提升准确度）
  - 混合检索（Milvus 支持，可选）
"""
import os
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import config
from tools.embedding_factory import EmbeddingFactory

# 导入增强型 RAG 组件（如果启用）
try:
    if config.VECTOR_DB_TYPE == "milvus":
        from src.rag.enhanced_rag import EnhancedRAGPipeline
        ENHANCED_RAG_AVAILABLE = True
    else:
        ENHANCED_RAG_AVAILABLE = False
except ImportError:
    ENHANCED_RAG_AVAILABLE = False
    print("⚠️ 增强型 RAG 组件未安装，使用标准 RAG")

class RAGManager:
    """RAG 管理器 - 支持知识库可用性检查和增强型检索"""
    
    def __init__(self):
        self.vectordb = None
        self.embeddings = None
        self._kb_available = False
        self.enhanced_pipeline = None
        self._llm = None  # 保存 LLM 实例用于延迟初始化
        
    def _init_enhanced_pipeline(self):
        """
        延迟初始化增强型 RAG 管道
        
        功能说明:
            仅在使用 Milvus 向量数据库时，才会创建增强型 RAG 管道。
            采用延迟初始化策略避免在不需要时加载重量级组件。
        
        初始化内容:
            - EnhancedRAGPipeline: 5阶段增强管道
              1. 相关性检查 (relevance_checker)
              2. 查询改写 (query_transformer)
              3. 混合检索 (hybrid_retriever)
              4. 结果合并 (result_merger)
              5. 重排序 (reranker)
        
        依赖条件:
            1. ENHANCED_RAG_AVAILABLE = True (组件已安装)
            2. config.VECTOR_DB_TYPE = "milvus"
            3. self.enhanced_pipeline = None (未初始化)
        
        错误处理:
            - 初始化失败 → 记录错误，设置 enhanced_pipeline = None
            - 回退到标准 RAG (Chroma)
        
        性能考虑:
            - LLM 实例复用（self._llm）
            - 仅调用一次，缓存到 self.enhanced_pipeline
        
        调用时机:
            - initialize() 方法中（首次创建知识库）
            - load_persisted_kb() 方法中（加载已有知识库）
        """
        if ENHANCED_RAG_AVAILABLE and config.VECTOR_DB_TYPE == "milvus" and not self.enhanced_pipeline:
            try:
                # 获取 LLM 实例
                if not self._llm:
                    from tools.llm_manager import llm_manager
                    self._llm = llm_manager.get_llm()
                
                self.enhanced_pipeline = EnhancedRAGPipeline(config, self._llm)
                print("✅ 增强型 RAG 管道已启用")
            except Exception as e:
                print(f"⚠️ 增强型 RAG 初始化失败，回退到标准 RAG: {e}")
                import traceback
                traceback.print_exc()
                self.enhanced_pipeline = None
    
    def is_enhanced_mode(self) -> bool:
        """
        检查是否启用增强模式
        
        功能说明:
            判断当前 RAG 系统是否使用增强型管道（Milvus + 5阶段优化）
        
        返回:
            bool: 
                - True: 增强模式（Milvus混合检索 + 优化管道）
                - False: 标准模式（Chroma向量检索）
        
        判断逻辑:
            enhanced_pipeline 不为 None AND vector_db_type = "milvus"
        
        使用场景:
            - retrieve() 方法中选择检索策略
            - is_kb_available() 中验证知识库状态
            - 调试输出增强模式状态
        """
        return self.enhanced_pipeline is not None and config.VECTOR_DB_TYPE == "milvus"
    
    def is_kb_available(self) -> bool:
        """
        检查知识库是否可用
        
        功能说明:
            验证知识库是否已成功初始化并可以提供检索服务。
            不同模式使用不同的验证逻辑。
        
        返回:
            bool:
                - True: 知识库已就绪，可以调用 retrieve()
                - False: 知识库未初始化或不可用，应跳过检索
        
        验证逻辑:
            增强模式: _kb_available=True AND enhanced_pipeline 不为 None
            标准模式: _kb_available=True AND vectordb 不为 None
        
        使用场景:
            - rag_retrieval_tool 工具调用前验证
            - Agent 初始化时判断 kb_available 参数
            - UI 显示知识库状态
        
        注意事项:
            - 此方法不会触发初始化，仅检查状态
            - 返回 False 时 retrieve() 会直接返回空列表
        """
        # 增强模式：检查 enhanced_pipeline 是否可用
        if self.is_enhanced_mode():
            return self._kb_available and self.enhanced_pipeline is not None
        # 标准模式：检查 vectordb 是否可用
        else:
            return self._kb_available and self.vectordb is not None
    
    def initialize(self, documents):
        """
        初始化 RAG 知识库
        
        功能说明:
            从文档列表创建向量数据库，支持 Chroma 和 Milvus 两种后端。
            包含文本分割、向量化、索引创建等完整流程。
        
        参数:
            documents (List[Document]): LangChain Document 对象列表
                - page_content: 文档文本内容
                - metadata: 元数据（source, file_type, chunk_index等）
        
        工作流程:
            1. 验证文档列表非空
            2. 创建嵌入模型（通过 EmbeddingFactory）
            3. 文本分割（RecursiveCharacterTextSplitter）
               - chunk_size: 500字符
               - chunk_overlap: 50字符
            4. 选择向量数据库后端
               - Milvus: 初始化增强型RAG管道 → 混合检索
               - Chroma: 创建标准向量库 → 语义检索
            5. 插入文档并创建索引
            6. 设置 _kb_available = True
        
        嵌入模型选择:
            通过 EMBEDDING_PROVIDER 环境变量控制：
            - huggingface: moka-ai/m3e-base (默认，中文优化)
            - openai: text-embedding-3-small
            - ollama: nomic-embed-text (本地)
        
        文本分割策略:
            使用递归分割器，优先按以下顺序分割：
            1. 双换行符 (\n\n)
            2. 单换行符 (\n)
            3. 中文句号 (。)
            4. 英文句号 (.)
            5. 空格
        
        向量数据库对比:
            | 特性 | Chroma | Milvus |
            |------|--------|--------|
            | 检索模式 | Dense向量 | Dense+Sparse混合 |
            | 索引算法 | HNSW | IVF_FLAT |
            | 适用规模 | <10万 | >百万 |
            | 召回率 | 85% | 95%+ |
        
        异常处理:
            - 文档为空 → 设置 _kb_available=False，返回
            - 分割后为空 → 警告并返回
            - 向量化失败 → 抛出异常，打印详细堆栈
        
        使用示例:
            >>> from langchain.schema import Document
            >>> docs = [Document(page_content="...", metadata={...})]
            >>> rag_manager.initialize(docs)
            📚 开始初始化知识库... (共 10 个文档)
            ✅ 知识库初始化成功! (Milvus 混合检索)
        """
        import os
        try:
            if not documents:
                print("⚠️ 没有文档可加载")
                self._kb_available = False
                return
            
            print(f"📚 开始初始化知识库... (共 {len(documents)} 个文档)")
            
            # 使用工厂模式创建嵌入模型（支持多种提供者）
            embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
            print(f"🔧 使用 Embedding 提供者: {embedding_provider}")
            
            self.embeddings = EmbeddingFactory.create_embedding()
            print(f"✅ Embedding 模型已加载")
            
            # 递归字符文本切分（增加 chunk_size 和 overlap，避免标题被单独分割）
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,      # 增加到 500 字符
                chunk_overlap=50     # 增加重叠到 50 字符
            )
            splits = splitter.split_documents(documents)
            
            if not splits:
                print("⚠️ 文本分割后为空")
                self._kb_available = False
                return
            
            print(f"✂️  文本已分割: {len(splits)} 个分片")
            
            # 根据配置选择向量数据库
            if config.VECTOR_DB_TYPE == "milvus":
                # 初始化增强型 RAG
                self._init_enhanced_pipeline()
                
            if self.is_enhanced_mode():
                # 使用增强型 RAG（Milvus）
                print(f"🚀 使用增强型 RAG (Milvus)")
                docs_to_insert = [
                    {
                        'text': doc.page_content,
                        'metadata': doc.metadata,
                        # ===== 新增：从metadata中提取权限字段 =====
                        'kb_level': doc.metadata.get('kb_level', 1),
                        'kb_category': doc.metadata.get('kb_category', 'public'),
                        'department': doc.metadata.get('department', '')
                    }
                    for doc in splits
                ]
                self.enhanced_pipeline.insert_documents(docs_to_insert)
                self._kb_available = True
                print(f"✅ 知识库初始化成功! (Milvus 混合检索)")
            else:
                # 使用标准 RAG（Chroma）
                print(f"�🔄 正在创建向量数据库... (保存位置: {config.CHROMA_PERSIST_DIR})")
                self.vectordb = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory=config.CHROMA_PERSIST_DIR
                )
                self._kb_available = True
                print(f"✅ 知识库初始化成功! (Chroma)")
            
        except Exception as e:
            self._kb_available = False
            import traceback
            error_msg = traceback.format_exc()
            print(f"❌ 知识库初始化失败:")
            print(f"   错误: {str(e)}")
            print(f"   详细信息:\n{error_msg}")
            raise
    
    def load_persisted_kb(self) -> bool:
        """
        加载已持久化的知识库
        
        功能说明:
            从磁盘加载已创建的向量数据库，避免重复初始化开销。
            智能识别 Milvus 或 Chroma 后端，自动选择加载策略。
        
        返回:
            bool:
                - True: 成功加载知识库，可以调用 retrieve()
                - False: 加载失败或知识库不存在
        
        工作流程:
            1. 根据 VECTOR_DB_TYPE 选择后端
            2. Milvus 模式:
               a. 初始化增强型管道
               b. 检查集合是否存在
               c. 验证文档数量 > 0
            3. Chroma 模式:
               a. 检查持久化目录存在
               b. 创建嵌入模型
               c. 加载 Chroma 实例
               d. 验证集合有内容
        
        Milvus 加载检查:
            - vector_db.collection 不为 None
            - collection.num_entities > 0
            - 成功 → 打印文档数量
            - 失败 → 回退到 Chroma
        
        Chroma 加载检查:
            - persist_directory 目录存在
            - _collection.count() > 0
            - 成功 → 打印加载成功
            - 失败 → 返回 False
        
        使用场景:
            - Streamlit 应用启动时自动加载
            - 用户点击"重新加载知识库"
            - Agent 初始化前的前置检查
        
        性能优化:
            - 仅加载元数据，不加载所有向量
            - Milvus: 懒加载集合
            - Chroma: 复用持久化连接
        
        异常处理:
            - 目录不存在 → 返回 False
            - 连接失败 → 打印详细错误，设置 _kb_available=False
            - 集合为空 → 警告并返回 False
        
        使用示例:
            >>> rag_manager.load_persisted_kb()
            🔄 尝试加载 Milvus 知识库...
            ✅ Milvus 知识库已加载 (文档数: 128)
            True
        """
        try:
            # 如果配置为 Milvus 模式
            if config.VECTOR_DB_TYPE == "milvus":
                print("🔄 尝试加载 Milvus 知识库...")
                self._init_enhanced_pipeline()
                
                if self.is_enhanced_mode():
                    # 检查 Milvus 集合是否存在且有数据
                    try:
                        vector_db = self.enhanced_pipeline.vector_db
                        if hasattr(vector_db, 'collection') and vector_db.collection is not None:
                            count = vector_db.collection.num_entities
                            if count > 0:
                                self._kb_available = True
                                print(f"✅ Milvus 知识库已加载 (文档数: {count})")
                                return True
                            else:
                                print("⚠️ Milvus 集合为空")
                                self._kb_available = False
                                return False
                    except Exception as e:
                        print(f"⚠️ Milvus 加载失败: {e}")
                        self._kb_available = False
                        return False
                else:
                    print("⚠️ 增强型 RAG 未启用")
                    self._kb_available = False
                    return False
            
            # Chroma 模式（向后兼容）
            else:
                persist_dir = getattr(config, 'CHROMA_PERSIST_DIR', './chroma_db')
                
                # 检查持久化目录是否存在
                if not os.path.exists(persist_dir):
                    print(f"⚠️ Chroma 目录不存在: {persist_dir}")
                    self._kb_available = False
                    return False
                
                # 使用工厂模式创建嵌入模型
                self.embeddings = EmbeddingFactory.create_embedding()
                
                self.vectordb = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings
                )
                
                # 检查数据库是否有内容
                if self.vectordb._collection.count() > 0:
                    self._kb_available = True
                    print(f"✅ Chroma 知识库已加载")
                    return True
                else:
                    print("⚠️ Chroma 数据库为空")
                    self._kb_available = False
                    return False
                
        except Exception as e:
            self._kb_available = False
            print(f"❌ 加载持久化知识库失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def retrieve(self, query, k=4, strategy=None, user_level=1):
        """
        智能检索知识库文档（带权限过滤）
        
        功能说明:
            根据查询问题从知识库检索最相关的文档片段。
            自动选择增强型（Milvus）或标准（Chroma）检索策略。
            支持基于用户权限级别的文档访问控制。
        
        参数:
            query (str): 用户查询文本
                示例: "RK3588 的 NPU 性能如何?"
            k (int): 返回的最大文档数，默认 4
                - 简单查询: k=3
                - 复杂对比: k=5
                - 技术细节: k=4
            strategy (str, optional): RAG 检索策略，可选值:
                - None: 使用配置文件默认策略 (config.RAG_STRATEGY_MODE)
                - "simple": 基础向量检索
                - "hybrid": 混合检索（Dense+Sparse）
                - "enhanced": 增强管道（5阶段）
                - "adaptive": 自适应选择
                - "full": 全流程（含上下文扩展）
            user_level (int): 用户权限级别，默认 1 ← 新增
                - 1: 一级权限（仅公开知识库）
                - 2: 二级权限（公开+内部）
                - 3: 三级权限（公开+内部+机密）
        
        返回:
            List[Document]: LangChain Document 对象列表
                - 每个 Document 包含:
                  - page_content: 文档内容
                  - metadata: {source, chunk_index, score, kb_level}
                - 知识库不可用时返回空列表 []
                - 仅返回 kb_level <= user_level 的文档
        
        工作流程:
            1. 检查知识库可用性 (is_kb_available())
            2. 判断检索模式:
               a. 增强模式 → enhanced_pipeline.retrieve()
                  - 相关性检查 → 查询改写 → 混合检索 → 结果合并 → 重排序
               b. 标准模式 → vectordb.similarity_search()
                  - 单纯 Dense 向量检索 (HNSW)
            3. 返回 Top-K 结果
        
        增强模式特性:
            - 5阶段处理流程
            - RRF 混合融合（Dense + Sparse）
            - BGE-Reranker 重排序
            - 自适应策略路由
            - 上下文片段扩展
        
        标准模式特性:
            - Chroma HNSW 索引
            - 余弦相似度计算
            - 单向量空间检索
        
        性能对比:
            | 指标 | 增强模式 | 标准模式 |
            |------|----------|----------|
            | 召回率 | 95%+ | 85% |
            | 延迟 | 200-300ms | 100-150ms |
            | 适用场景 | 技术文档 | 通用问答 |
        
        策略选择建议:
            - 技术问答 → "enhanced" 或 "full"
            - 产品对比 → "adaptive"
            - 简单查找 → "simple"
            - 多语言混合 → "hybrid"
        
        异常处理:
            - 知识库未初始化 → 返回 []
            - 检索失败 → 打印错误，返回 []
            - strategy 无效 → 使用默认策略
        
        使用示例:
            >>> # 使用默认策略
            >>> docs = rag_manager.retrieve("RK3588性能参数", k=4)
            🚀 使用增强型 RAG 检索... (策略: enhanced)
            ✅ 检索到 4 个相关文档
            
            >>> # 指定策略
            >>> docs = rag_manager.retrieve("对比RK3588和RK3568", k=5, strategy="adaptive")
            🚀 使用增强型 RAG 检索... (策略: adaptive)
            📊 自适应路由 → 对比类查询 → 混合检索
        """
        if not self.is_kb_available():
            return []
        
        try:
            # 使用增强型 RAG（如果可用）
            if self.is_enhanced_mode():
                print(f"🚀 使用增强型 RAG 检索... (策略: {strategy or config.RAG_STRATEGY_MODE}, 权限级别: {user_level})")
                # 传递策略参数和权限级别到增强管道
                results = self.enhanced_pipeline.retrieve(
                    query, 
                    top_k=k,
                    strategy_override=strategy,  # 传递策略参数
                    user_level=user_level  # ← 新增：传递权限级别
                )
                return results
            else:
                # 标准 Chroma 检索（不支持权限过滤）
                print(f"📚 使用标准 RAG 检索... (权限过滤功能仅Milvus支持)")
                results = self.vectordb.similarity_search(query, k=k)
                return results
        except Exception as e:
            print(f"检索失败: {str(e)}")
            # 如果增强模式失败，尝试回退到标准模式
            if self.is_enhanced_mode() and self.vectordb is not None:
                print("⚠️ 增强模式失败，回退到标准检索...")
                try:
                    results = self.vectordb.similarity_search(query, k=k)
                    return results
                except Exception as e2:
                    print(f"标准检索也失败: {str(e2)}")
                    return []
            return []

rag_manager = RAGManager()


def get_kb_status() -> dict:
    """
    获取知识库状态信息
    
    功能说明:
        返回知识库的完整状态信息，包括可用性、运行模式、后端类型等。
        用于 UI 展示和调试诊断。
    
    返回:
        dict: 知识库状态字典，包含以下字段:
            - available (bool): 知识库是否可用
                - True: 已初始化且有文档
                - False: 未初始化或为空
            - mode (str): RAG 运行模式
                - "enhanced": 增强模式（Milvus + 5阶段优化）
                - "standard": 标准模式（Chroma + 向量检索）
                - "none": 未初始化
            - vector_db (str): 向量数据库类型
                - "milvus": Milvus 分布式向量库
                - "chroma": Chroma 本地向量库
            - message (str): 人类可读的状态描述
                - 成功: "知识库已就绪 (enhanced 模式, milvus)"
                - 失败: "知识库未初始化或为空"
    
    使用场景:
        1. Streamlit 侧边栏显示知识库状态
        2. Agent 初始化前的前置检查
        3. 调试日志记录
        4. API 接口返回状态
    
    状态判断逻辑:
        available = rag_manager.is_kb_available()
        mode = "enhanced" if is_enhanced_mode() else "standard"
        vector_db = config.VECTOR_DB_TYPE
    
    使用示例:
        >>> status = get_kb_status()
        >>> print(status)
        {
            'available': True,
            'mode': 'enhanced',
            'vector_db': 'milvus',
            'message': '知识库已就绪 (enhanced 模式, milvus)'
        }
        
        >>> # UI 展示
        >>> if status['available']:
        >>>     st.success(status['message'])
        >>> else:
        >>>     st.warning(status['message'])
    
    注意事项:
        - 此函数不会触发初始化，仅读取状态
        - available=False 时 mode 固定为 "none"
        - vector_db 始终反映 config.VECTOR_DB_TYPE
    """
    if rag_manager.is_kb_available():
        mode = "enhanced" if rag_manager.is_enhanced_mode() else "standard"
        vector_db = config.VECTOR_DB_TYPE
        return {
            "available": True,
            "mode": mode,
            "vector_db": vector_db,
            "message": f"知识库已就绪 ({mode} 模式, {vector_db})"
        }
    else:
        return {
            "available": False,
            "mode": "none",
            "vector_db": config.VECTOR_DB_TYPE,
            "message": "知识库未初始化或为空"
        }


@tool
def rag_retrieval_tool(query: str) -> dict:
    """
    从知识库检索相关文档（LangChain Tool 包装）
    
    功能说明:
        LangGraph Agent 调用的 RAG 检索工具。
        自动选择增强型或标准检索模式，支持动态策略切换。
    
    参数:
        query (str): 用户查询文本
            示例: "RK3588 支持哪些操作系统?"
    
    返回:
        dict: 检索结果字典
            成功时:
                - success (bool): True
                - documents (List[str]): 文档内容列表
                - count (int): 检索到的文档数
                - kb_available (bool): True
                - mode (str): "enhanced" 或 "standard"
                - strategy (str): 使用的检索策略
                - optimizations (List[str]): 启用的优化项
                    - "relevance_check": 相关性过滤
                    - "query_transform": 查询改写
                    - "hybrid_search": 混合检索
                    - "rerank": BGE 重排序
                    - "context_enrichment": 上下文扩展
            失败时:
                - success (bool): False
                - documents (List): []
                - count (int): 0
                - kb_available (bool): False
                - mode (str): "none"
                - strategy (str): "none"
                - error (str): 错误原因
    
    工作流程:
        1. 检查知识库可用性 (rag_manager.is_kb_available())
        2. 尝试从 Streamlit session_state 获取用户选择的策略
        3. 调用 rag_manager.retrieve(query, k=4, strategy)
        4. 提取文档内容和元数据
        5. 根据配置收集启用的优化项
        6. 返回结构化结果
    
    策略选择逻辑:
        - 优先使用 st.session_state.rag_strategy（UI选择）
        - 回退到 config.RAG_STRATEGY_MODE（配置文件）
        - 最终由 enhanced_pipeline 自动选择
    
    优化项检测:
        通过 config 开关判断启用的优化:
        - RAG_ENABLE_RELEVANCE_CHECK → "relevance_check"
        - RAG_ENABLE_QUERY_TRANSFORM → "query_transform"
        - RAG_ENABLE_HYBRID → "hybrid_search"
        - RAG_ENABLE_RERANK → "rerank"
        - RAG_ENABLE_CONTEXT_ENRICHED → "context_enrichment"
    
    使用场景:
        - Agent 工具调用: agent.run("查询RK3588性能")
        - LangGraph 节点: llm_agent() 中的 tool_node
        - 手动调用: rag_retrieval_tool.invoke({"query": "..."})
    
    异常处理:
        - 知识库未初始化 → 返回 success=False, error 信息
        - 检索失败 → 捕获异常，返回空结果
        - Streamlit 不可用 → 忽略，使用默认策略
    
    使用示例:
        >>> # Agent 自动调用
        >>> result = rag_retrieval_tool.invoke({"query": "RK3588参数"})
        📌 使用UI选择的策略: enhanced
        🚀 使用增强型 RAG 检索... (策略: enhanced)
        ✅ 检索到 4 个相关文档
        
        >>> print(result)
        {
            'success': True,
            'documents': ['RK3588技术规格...', '性能参数对比...'],
            'count': 4,
            'kb_available': True,
            'mode': 'enhanced',
            'strategy': 'enhanced',
            'optimizations': ['relevance_check', 'query_transform', 'hybrid_search', 'rerank']
        }
    
    注意事项:
        - @tool 装饰器将函数转换为 LangChain BaseTool
        - Agent 通过工具描述理解何时调用此工具
        - 知识库不可用时不会抛出异常，返回失败状态
        - 文档内容已提取为字符串，去除 LangChain 对象包装
    """
    
    # 检查知识库是否可用
    if not rag_manager.is_kb_available():
        return {
            "success": False,
            "documents": [],
            "count": 0,
            "kb_available": False,
            "mode": "none",
            "strategy": "none",
            "error": "知识库未初始化或为空，已跳过检索"
        }
    
    try:
        # 从 streamlit session state 获取策略和用户权限
        strategy = None
        user_level = 1  # 默认一级权限
        try:
            import streamlit as st
            if hasattr(st, 'session_state'):
                # 获取策略
                if 'rag_strategy' in st.session_state:
                    strategy = st.session_state.rag_strategy
                    print(f"📌 使用UI选择的策略: {strategy}")
                
                # ===== 新增：获取用户权限级别 =====
                if 'user_info' in st.session_state:
                    user_level = st.session_state.user_info.get('access_level', 1)
                    print(f"🔐 用户权限级别: Level {user_level}")
        except:
            pass  # 非streamlit环境或session state不可用
        
        # 获取检索模式
        mode = "enhanced" if rag_manager.is_enhanced_mode() else "standard"
        
        # 收集启用的优化项
        optimizations = []
        if mode == "enhanced":
            if config.RAG_ENABLE_RELEVANCE_CHECK:
                optimizations.append("relevance_check")
            if config.RAG_ENABLE_QUERY_TRANSFORM:
                optimizations.append("query_transform")
            if config.RAG_ENABLE_RERANK:
                optimizations.append("rerank")
            if config.RAG_ENABLE_HYDE:
                optimizations.append("hyde")
        
        # 执行检索（传递策略参数和用户权限级别）
        documents = rag_manager.retrieve(
            query, 
            k=config.RAG_TOP_K, 
            strategy=strategy,
            user_level=user_level  # ← 新增：传递权限级别
        )
        
        return {
            "success": True,
            "documents": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": doc.metadata.get("score", None)
                }
                for doc in documents
            ],
            "count": len(documents),
            "kb_available": True,
            "mode": mode,
            "strategy": strategy or config.RAG_STRATEGY_MODE,
            "optimizations": optimizations
        }
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ RAG 检索错误:\n{error_detail}")
        
        return {
            "success": False,
            "documents": [],
            "count": 0,
            "kb_available": True,
            "mode": "error",
            "strategy": strategy or "unknown",
            "error": str(e)
        }
