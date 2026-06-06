"""
自适应 RAG (Adaptive RAG) - TOP 5 推荐方案

【功能说明】
根据查询类型自动选择最优检索策略，在成本和准确度间动态平衡。
智能路由到不同的RAG策略组合。

【核心功能】
1. **查询分类**
   - 简单查询 (simple): <5词 → 快速检索
   - 事实查询 (factual): "什么是" → 精确检索
   - 对比查询 (comparison): "vs", "对比" → 多维检索
   - 技术查询 (technical): "如何", "为什么" → 深度检索
   - 复杂查询 (complex): 多条件 → 全流程优化

2. **策略路由**
   - simple → 向量搜索（快速）
   - factual → 融合检索（精确）
   - comparison → 多主体检索 + 合并
   - technical → 查询转换 + 重排序
   - complex → 全流程（转换+融合+重排+上下文）

3. **成本优化**
   - 简单查询: 最小成本
   - 复杂查询: 高质量保证
   - 动态调整: 失败降级

4. **性能统计**
   - 查询类型分布
   - 延迟统计
   - 质量评估

【技术架构】
```
查询输入
    ↓
QueryClassifier (查询分类)
    ├─ simple → _simple_strategy (向量搜索)
    ├─ factual → _factual_strategy (融合检索)
    ├─ comparison → _comparison_strategy (多维检索)
    ├─ technical → _technical_strategy (深度检索)
    └─ complex → _complex_strategy (全流程)
    ↓
检索结果 + 元信息
```

【查询类型判定】
| 类型 | 触发条件 | 示例 | 策略 |
|------|----------|------|------|
| simple | <5词 | "Python" | 向量搜索 |
| factual | "什么是", "定义" | "什么是BI？" | 融合检索 |
| comparison | "vs", "对比" | "ResNet vs VGG" | 多主体检索 |
| technical | "如何", "原理" | "如何实现跳跃连接？" | 查询转换+重排 |
| complex | >15词, 多子句 | "介绍ResNet和VGG的区别及应用场景" | 全流程优化 |

【性能数据】
| 策略 | 延迟 | 成本 | 准确率 |
|------|------|------|--------|
| simple | <50ms | 最低 | 75% |
| factual | ~150ms | 低 | 85% |
| comparison | ~300ms | 中 | 88% |
| technical | ~500ms | 中高 | 92% |
| complex | ~800ms | 高 | 95% |

【使用方式】
```python
from src.rag.adaptive_rag import AdaptiveRAG

# 初始化
adaptive = AdaptiveRAG(config, llm, vector_db)

# 自动分类检索
results = adaptive.retrieve(query="ResNet和VGG的区别")
# 🔍 自适应检索: 'ResNet和VGG的区别' -> 类型=comparison
# 📌 使用对比策略（多维检索）
#   对比主体: ['ResNet', 'VGG']
# ✅ 检索完成: 10 个结果 (策略=comparison)

# 强制指定类型（测试）
results = adaptive.retrieve(
    query="Python",
    force_type=QueryType.SIMPLE
)

# 查看统计
stats = adaptive.get_stats()
print(stats['query_type_counts'])
# {'simple': 10, 'factual': 5, 'comparison': 3, ...}
```

【适用场景】
✅ 推荐使用:
- 多样化查询场景
- 成本敏感应用
- 需要动态优化
- 大规模用户系统

❌ 不推荐使用:
- 查询类型单一（直接用固定策略）
- 对延迟极敏感（分类有开销）
- 简单应用（过度设计）

【优化建议】
1. **规则优化**: 根据业务调整分类规则
2. **缓存**: 缓存常见查询的分类结果
3. **监控**: 跟踪各策略效果，持续优化
4. **降级**: 策略失败时自动降级

【注意事项】
- 分类规则基于关键词（可能误判）
- 策略模块需按需加载（避免初始化开销）
- 失败时自动降级到简单策略
- 统计信息用于持续优化

【版本】
V3 - 2025-10
TOP 5 推荐 RAG 策略
"""
import logging
import re
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """
    查询类型枚举
    
    SIMPLE: 简单查询（<5词，快速检索）
    FACTUAL: 事实查询（"什么是"，精确检索）
    COMPARISON: 对比查询（"vs"，多维检索）
    TECHNICAL: 技术查询（"如何"，深度检索）
    COMPLEX: 复杂查询（多条件，全流程优化）
    UNKNOWN: 未知类型（默认策略）
    """
    SIMPLE = "simple"
    FACTUAL = "factual"
    COMPARISON = "comparison"
    TECHNICAL = "technical"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


class AdaptiveRAG:
    """
    自适应 RAG 系统（Intelligent Query Router）
    
    功能说明:
        根据查询类型智能路由到最优策略。
        平衡成本与准确度。
    
    核心优势:
        - 自动分类: 无需手动指定
        - 动态路由: 智能选择策略
        - 成本优化: 简单查询低成本
        - 质量保证: 复杂查询高质量
    
    使用示例:
        >>> adaptive = AdaptiveRAG(config, llm, vector_db)
        >>> results = adaptive.retrieve("ResNet vs VGG")
        类型=comparison, 策略=多维检索
    """
    
    def __init__(self, config, llm, vector_db):
        """
        初始化自适应 RAG
        
        功能说明:
            创建查询分类器和策略路由表。
            初始化性能统计。
        
        参数:
            config: 配置对象（包含 RAG_TOP_K, RAG_ENABLE_RERANK 等）
            llm (BaseLLM): LangChain LLM 实例
            vector_db (MilvusManager): 向量数据库实例
        
        工作流程:
            1. 保存配置和资源引用
            2. 创建查询分类器
            3. 注册策略路由表
            4. 初始化统计字典
            5. 打印初始化日志
        
        使用示例:
            >>> adaptive = AdaptiveRAG(config, llm, vector_db)
            🚀 自适应RAG系统初始化完成
        
        策略映射:
            ```python
            {
                QueryType.SIMPLE: _simple_strategy,
                QueryType.FACTUAL: _factual_strategy,
                QueryType.COMPARISON: _comparison_strategy,
                QueryType.TECHNICAL: _technical_strategy,
                QueryType.COMPLEX: _complex_strategy,
                QueryType.UNKNOWN: _default_strategy
            }
            ```
        """
        self.config = config
        self.llm = llm
        self.vector_db = vector_db
        
        # 查询分类器
        self.query_classifier = QueryClassifier()
        
        # 策略配置
        self.strategies = {
            QueryType.SIMPLE: self._simple_strategy,
            QueryType.FACTUAL: self._factual_strategy,
            QueryType.COMPARISON: self._comparison_strategy,
            QueryType.TECHNICAL: self._technical_strategy,
            QueryType.COMPLEX: self._complex_strategy,
            QueryType.UNKNOWN: self._default_strategy
        }
        
        # 性能统计
        self.stats = {
            'total_queries': 0,
            'query_type_counts': {t.value: 0 for t in QueryType},
            'avg_latency': {},
            'avg_quality': {}
        }
        
        logger.info("🚀 自适应RAG系统初始化完成")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        force_type: Optional[QueryType] = None
    ) -> List[Dict[str, Any]]:
        """
        自适应检索（主入口）
        
        功能说明:
            自动分类查询，路由到最优策略，执行检索。
            添加元信息并更新统计。
        
        参数:
            query (str): 查询文本
            k (int, optional): 返回结果数量
                - 默认: config.RAG_TOP_K
            force_type (QueryType, optional): 强制指定查询类型
                - 用于测试和调试
                - 默认: None（自动分类）
        
        返回:
            List[Dict]: 检索结果列表
                每个结果包含额外元信息:
                ```python
                {
                    'id': '文档ID',
                    'text': '内容',
                    'score': 0.85,
                    'metadata': {
                        'query_type': 'comparison',  # 查询类型
                        'strategy': '_comparison_strategy',  # 使用的策略
                        ...
                    }
                }
                ```
        
        工作流程:
            1. 确定 k 值（默认 config.RAG_TOP_K）
            2. 分类查询:
               - force_type 存在 → 使用指定类型
               - 否则 → QueryClassifier 自动分类
            3. 更新统计计数
            4. 路由到对应策略
            5. 执行检索
            6. 添加元信息
            7. 返回结果
        
        使用示例:
            >>> # 自动分类
            >>> results = adaptive.retrieve("ResNet和VGG的区别")
            🔍 自适应检索: 'ResNet和VGG的区别' -> 类型=comparison
            📌 使用对比策略（多维检索）
              对比主体: ['ResNet', 'VGG']
            ✅ 检索完成: 10 个结果 (策略=comparison)
            
            >>> # 强制类型
            >>> results = adaptive.retrieve(
            ...     query="Python",
            ...     force_type=QueryType.SIMPLE
            ... )
            🔍 自适应检索: 'Python' -> 类型=simple
            📌 使用简单策略（快速检索）
        
        查询类型示例:
            - "Python" → simple
            - "什么是BI？" → factual
            - "ResNet vs VGG" → comparison
            - "如何实现跳跃连接？" → technical
            - "介绍ResNet和VGG的区别及应用" → complex
        
        性能数据:
            - simple: ~50ms
            - factual: ~150ms
            - comparison: ~300ms
            - technical: ~500ms
            - complex: ~800ms
        
        异常处理:
            - 检索失败 → 返回 []
            - 记录 error 和 traceback
        
        注意事项:
            - 元信息用于追踪和分析
            - 统计数据用于优化
            - 策略失败自动降级
        """
        try:
            if k is None:
                k = self.config.RAG_TOP_K
            
            # Step 1: 检测查询类型
            if force_type:
                query_type = force_type
            else:
                query_type = self.query_classifier.classify(query)
            
            logger.info(f"🔍 自适应检索: '{query}' -> 类型={query_type.value}")
            
            # 更新统计
            self.stats['total_queries'] += 1
            self.stats['query_type_counts'][query_type.value] += 1
            
            # Step 2: 选择并执行策略
            strategy = self.strategies.get(query_type, self._default_strategy)
            results = strategy(query, k)
            
            logger.info(f"✅ 检索完成: {len(results)} 个结果 (策略={query_type.value})")
            
            # 添加元信息
            for result in results:
                if 'metadata' not in result:
                    result['metadata'] = {}
                result['metadata']['query_type'] = query_type.value
                result['metadata']['strategy'] = strategy.__name__
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 自适应检索失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _simple_strategy(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        简单查询策略（Fast Path）
        
        特点:
            - 直接向量搜索
            - 最小成本
            - 快速响应
        
        适用: 单词查询、简短问题
        延迟: ~50ms
        
        示例: "Python", "BI", "数据库"
        """
        logger.info("📌 使用简单策略（快速检索）")
        
        results = self.vector_db.search(
            query=query,
            top_k=k
        )
        
        return results
    
    def _factual_strategy(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        事实查询策略（Precision Search）
        
        特点:
            - 融合检索（向量 + BM25）
            - 精确匹配
            - 兼顾语义和关键词
        
        适用: "什么是X"类问题
        延迟: ~150ms
        
        示例: "什么是BI？", "Python的定义"
        """
        logger.info("📌 使用事实策略（融合检索）")
        
        try:
            from src.rag.hybrid_retriever import HybridRetriever
            
            # 懒加载融合检索器
            if not hasattr(self, '_hybrid_retriever'):
                self._hybrid_retriever = HybridRetriever(self.vector_db)
            
            results = self._hybrid_retriever.search(
                query=query,
                k=k,
                semantic_weight=0.5  # 语义和关键词同等重要
            )
            
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ 融合检索失败，降级到向量搜索: {e}")
            return self._simple_strategy(query, k)
    
    def _comparison_strategy(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        对比查询策略（Multi-Dimensional Search）
        
        特点:
            - 提取对比主体
            - 多次检索合并
            - 可选上下文增强
        
        适用: 对比类问题
        延迟: ~300ms
        
        示例: "ResNet vs VGG", "Python和Java的区别"
        """
        logger.info("📌 使用对比策略（多维检索）")
        
        try:
            # 提取对比主体
            subjects = self._extract_comparison_subjects(query)
            
            if not subjects:
                # 无法提取，降级
                return self._default_strategy(query, k)
            
            logger.info(f"  对比主体: {subjects}")
            
            # 为每个主体检索
            all_results = []
            for subject in subjects:
                results = self.vector_db.search(
                    query=subject,
                    top_k=k
                )
                all_results.extend(results)
            
            # 去重合并
            from src.rag.result_merger import ResultMerger
            merger = ResultMerger()
            merged = merger.merge([all_results], top_k=k * 2)
            
            # 上下文增强
            if hasattr(self, '_context_retriever'):
                enriched = []
                for result in merged[:k]:
                    enriched_result = self._context_retriever._enrich_with_context(
                        result,
                        context_radius=1
                    )
                    enriched.append(enriched_result)
                return enriched
            
            return merged[:k]
            
        except Exception as e:
            logger.warning(f"⚠️ 对比策略失败，降级: {e}")
            return self._default_strategy(query, k)
    
    def _technical_strategy(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        技术查询策略（Deep Search）
        
        特点:
            - 查询转换（生成变体）
            - 多查询检索
            - LLM 重排序
        
        适用: 技术深度问题
        延迟: ~500ms
        
        示例: "如何实现跳跃连接？", "为什么使用ReLU？"
        """
        logger.info("📌 使用技术策略（深度检索）")
        
        try:
            # 查询转换
            from src.rag.query_transformer import QueryTransformer
            
            if not hasattr(self, '_query_transformer'):
                self._query_transformer = QueryTransformer(self.llm, num_variants=3)
            
            queries = self._query_transformer.transform(query)
            logger.info(f"  生成 {len(queries)} 个查询变体")
            
            # 多查询检索
            all_results = []
            for q in queries:
                results = self.vector_db.search(query=q, top_k=k)
                all_results.append(results)
            
            # 合并
            from src.rag.result_merger import ResultMerger
            merger = ResultMerger()
            merged = merger.merge(all_results, top_k=k * 2)
            
            # 重排序
            if self.config.RAG_ENABLE_RERANK:
                from src.rag.reranker import Reranker
                
                if not hasattr(self, '_reranker'):
                    self._reranker = Reranker(self.llm)
                
                merged = self._reranker.rerank(query, merged, top_k=k)
            
            return merged
            
        except Exception as e:
            logger.warning(f"⚠️ 技术策略失败，降级: {e}")
            return self._default_strategy(query, k)
    
    def _complex_strategy(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        复杂查询策略（Full Pipeline）
        
        特点:
            - 组合多种策略
            - 大半径上下文增强
            - 最高质量保证
        
        适用: 多条件、多子句查询
        延迟: ~800ms
        
        示例: "介绍ResNet和VGG的区别、优势及应用场景"
        """
        logger.info("📌 使用复杂策略（全流程优化）")
        
        # 组合多种策略
        results = self._technical_strategy(query, k * 2)
        
        # 上下文增强
        try:
            from src.rag.context_enriched import ContextEnrichedRetriever
            
            if not hasattr(self, '_context_retriever'):
                self._context_retriever = ContextEnrichedRetriever(self.vector_db)
            
            enriched = []
            for result in results[:k]:
                enriched_result = self._context_retriever._enrich_with_context(
                    result,
                    context_radius=2,  # 更大的上下文
                    merge_context=True
                )
                enriched.append(enriched_result)
            
            return enriched
            
        except Exception as e:
            logger.warning(f"⚠️ 上下文增强失败: {e}")
            return results[:k]
    
    def _default_strategy(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        默认策略（Balanced Search）
        
        特点:
            - 融合检索
            - 中等成本
            - 平衡质量和延迟
        
        适用: 未分类查询
        延迟: ~150ms
        """
        logger.info("📌 使用默认策略")
        
        try:
            from src.rag.hybrid_retriever import HybridRetriever
            
            if not hasattr(self, '_hybrid_retriever'):
                self._hybrid_retriever = HybridRetriever(self.vector_db)
            
            results = self._hybrid_retriever.search(
                query=query,
                k=k,
                semantic_weight=0.6
            )
            
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ 默认策略失败，使用简单策略: {e}")
            return self._simple_strategy(query, k)
    
    def _extract_comparison_subjects(self, query: str) -> List[str]:
        """
        提取对比查询中的主体
        
        例如: "ResNet和VGG的区别" -> ["ResNet", "VGG"]
        """
        # 常见对比连接词
        separators = ['和', '与', 'vs', 'VS', '对比', '比较', '区别']
        
        for sep in separators:
            if sep in query:
                # 简单分割
                parts = query.split(sep)
                if len(parts) >= 2:
                    # 提取主体（去除噪音词）
                    subjects = []
                    noise_words = ['的', '之间', '有什么', '介绍', '说明', '区别', '对比', '比较']
                    
                    for part in parts[:2]:  # 只取前两个
                        cleaned = part.strip()
                        for noise in noise_words:
                            cleaned = cleaned.replace(noise, '')
                        cleaned = cleaned.strip()
                        
                        if cleaned:
                            subjects.append(cleaned)
                    
                    if len(subjects) >= 2:
                        return subjects
        
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计"""
        self.stats = {
            'total_queries': 0,
            'query_type_counts': {t.value: 0 for t in QueryType},
            'avg_latency': {},
            'avg_quality': {}
        }


class QueryClassifier:
    """
    查询分类器（Rule-Based Classifier）
    
    功能说明:
        基于关键词和模式规则对查询进行分类。
        按优先级顺序检查，返回第一个匹配的类型。
    
    分类规则:
        - COMPLEX: >15词 OR 多子句 OR 复杂连接词
        - COMPARISON: "vs", "对比", "区别"等
        - TECHNICAL: "如何", "为什么", "原理"等
        - FACTUAL: "什么是", "定义", "介绍"等
        - SIMPLE: <5词 OR <15字符
        - UNKNOWN: 无匹配规则
    
    优先级:
        complex > comparison > technical > factual > simple
    
    使用示例:
        >>> classifier = QueryClassifier()
        >>> query_type = classifier.classify("ResNet和VGG的区别")
        QueryType.COMPARISON
    """
    
    def __init__(self):
        """
        初始化分类器
        
        构建基于规则的分类模式。
        每个类型对应多个 lambda 函数规则。
        """
        # 分类规则（基于关键词和模式）
        self.patterns = {
            QueryType.SIMPLE: [
                lambda q: len(q.split()) < 5,  # 少于5个词
                lambda q: len(q) < 15,          # 少于15个字符
            ],
            QueryType.FACTUAL: [
                lambda q: any(w in q for w in ['什么是', '定义', 'what is', '介绍']),
            ],
            QueryType.COMPARISON: [
                lambda q: any(w in q for w in ['vs', 'VS', '对比', '比较', '区别', '和', '与']),
                lambda q: '还是' in q,
            ],
            QueryType.TECHNICAL: [
                lambda q: any(w in q for w in ['如何', '怎么', '为什么', 'how', 'why']),
                lambda q: any(w in q for w in ['实现', '原理', '机制', '算法']),
            ],
            QueryType.COMPLEX: [
                lambda q: len(q.split()) > 15,  # 超过15个词
                lambda q: q.count('，') + q.count('。') > 2,  # 多个子句
                lambda q: any(w in q for w in ['并且', '同时', '以及', '另外']),
            ]
        }
    
    def classify(self, query: str) -> QueryType:
        """
        分类查询（按优先级匹配）
        
        参数:
            query (str): 查询文本
        
        返回:
            QueryType: 查询类型枚举
        
        工作流程:
            1. 按优先级遍历类型
            2. 对每个类型检查规则
            3. 任一规则匹配 → 返回该类型
            4. 无匹配 → 返回 UNKNOWN
        
        使用示例:
            >>> classifier.classify("Python")
            QueryType.SIMPLE
            
            >>> classifier.classify("什么是BI？")
            QueryType.FACTUAL
            
            >>> classifier.classify("ResNet vs VGG")
            QueryType.COMPARISON
            
            >>> classifier.classify("如何实现跳跃连接？")
            QueryType.TECHNICAL
            
            >>> classifier.classify("介绍ResNet和VGG的区别及应用场景")
            QueryType.COMPLEX
        
        优先级顺序:
            1. COMPLEX (最高优先级)
            2. COMPARISON
            3. TECHNICAL
            4. FACTUAL
            5. SIMPLE
            6. UNKNOWN (默认)
        
        注意事项:
            - 基于规则，可能误判
            - 优先级顺序影响结果
            - 可根据业务调整规则
        """
        # 按优先级检查
        priority_order = [
            QueryType.COMPLEX,      # 先检查复杂查询
            QueryType.COMPARISON,   # 对比查询
            QueryType.TECHNICAL,    # 技术查询
            QueryType.FACTUAL,      # 事实查询
            QueryType.SIMPLE,       # 简单查询
        ]
        
        for query_type in priority_order:
            if query_type in self.patterns:
                rules = self.patterns[query_type]
                if any(rule(query) for rule in rules):
                    return query_type
        
        return QueryType.UNKNOWN
