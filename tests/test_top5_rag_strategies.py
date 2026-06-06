"""
测试 TOP 5 推荐RAG方案

测试内容：
1. 融合检索 (Hybrid RAG)
2. 上下文增强 (Context Enriched RAG)
3. 自适应RAG (Adaptive RAG)
4. 策略对比（simple vs hybrid vs enhanced vs full）

运行方式：
    python test_top5_rag_strategies.py
"""
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from tools.llm_factory import create_llm
from src.vector_db.milvus_manager import MilvusManager
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.context_enriched import ContextEnrichedRetriever, SmartContextRetriever
from src.rag.adaptive_rag import AdaptiveRAG, QueryType
from src.rag.enhanced_rag import EnhancedRAGPipeline

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_hybrid_retriever():
    """测试融合检索器"""
    print("\n" + "="*80)
    print("📊 测试1: 融合检索 (Hybrid RAG)")
    print("="*80)
    
    try:
        # 初始化
        config = Config()
        milvus_config = {
            'host': config.MILVUS_HOST,
            'port': config.MILVUS_PORT,
            'collection_name': config.MILVUS_COLLECTION_NAME,
            'embedding_model': config.BGEM3_EMBEDDING_MODEL,
            'dense_dim': config.BGEM3_DENSE_DIM
        }
        
        vector_db = MilvusManager(milvus_config)
        hybrid = HybridRetriever(vector_db)
        
        # 测试查询
        test_queries = [
            "iPhone 15",  # 关键词精确匹配
            "苹果手机新品",  # 语义理解
            "黄瑞楠的项目经验"  # 混合场景
        ]
        
        for query in test_queries:
            print(f"\n🔍 查询: {query}")
            print("-" * 60)
            
            # RRF融合
            results_rrf = hybrid.search(
                query=query,
                k=5,
                semantic_weight=0.6,
                use_rrf=True
            )
            
            print(f"📌 RRF融合结果 ({len(results_rrf)} 个):")
            for i, r in enumerate(results_rrf[:3], 1):
                sources = r.get('fusion_sources', [])
                print(f"  {i}. [{', '.join(sources)}] 分数={r['score']:.4f}")
                print(f"     {r['text'][:80]}...")
            
            # 加权融合
            results_weighted = hybrid.search(
                query=query,
                k=5,
                semantic_weight=0.6,
                use_rrf=False
            )
            
            print(f"\n📌 加权融合结果 ({len(results_weighted)} 个):")
            for i, r in enumerate(results_weighted[:3], 1):
                sources = r.get('fusion_sources', [])
                print(f"  {i}. [{', '.join(sources)}] 分数={r['score']:.4f}")
                print(f"     {r['text'][:80]}...")
        
        print("\n✅ 融合检索测试完成")
        
    except Exception as e:
        print(f"\n❌ 融合检索测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_context_enriched():
    """测试上下文增强"""
    print("\n" + "="*80)
    print("📊 测试2: 上下文增强 (Context Enriched RAG)")
    print("="*80)
    
    try:
        # 初始化
        config = Config()
        milvus_config = {
            'host': config.MILVUS_HOST,
            'port': config.MILVUS_PORT,
            'collection_name': config.MILVUS_COLLECTION_NAME,
            'embedding_model': config.BGEM3_EMBEDDING_MODEL,
            'dense_dim': config.BGEM3_DENSE_DIM
        }
        
        vector_db = MilvusManager(milvus_config)
        context_retriever = ContextEnrichedRetriever(vector_db)
        
        # 测试查询
        query = "ResNet的跳跃连接"
        
        print(f"\n🔍 查询: {query}")
        print("-" * 60)
        
        # 不同半径测试
        for radius in [0, 1, 2]:
            print(f"\n📌 上下文半径 = {radius}:")
            
            results = context_retriever.search_with_context(
                query=query,
                k=3,
                context_radius=radius,
                merge_context=True
            )
            
            if results:
                result = results[0]
                
                print(f"  主块: {result['text'][:100]}...")
                
                if 'context_chunks' in result:
                    print(f"  上下文块数: {len(result['context_chunks'])}")
                    for chunk in result['context_chunks']:
                        pos = chunk['position']
                        emoji = '📌' if pos == 'main' else '⬆️' if pos == 'before' else '⬇️'
                        print(f"    {emoji} {pos}: {chunk['text'][:60]}...")
        
        print("\n✅ 上下文增强测试完成")
        
    except Exception as e:
        print(f"\n❌ 上下文增强测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_adaptive_rag():
    """测试自适应RAG"""
    print("\n" + "="*80)
    print("📊 测试3: 自适应RAG (Adaptive RAG)")
    print("="*80)
    
    try:
        # 初始化
        config = Config()
        llm = create_llm(config)
        
        milvus_config = {
            'host': config.MILVUS_HOST,
            'port': config.MILVUS_PORT,
            'collection_name': config.MILVUS_COLLECTION_NAME,
            'embedding_model': config.BGEM3_EMBEDDING_MODEL,
            'dense_dim': config.BGEM3_DENSE_DIM
        }
        
        vector_db = MilvusManager(milvus_config)
        adaptive = AdaptiveRAG(config, llm, vector_db)
        
        # 测试不同类型的查询
        test_cases = [
            ("BI", QueryType.SIMPLE),
            ("什么是商业智能？", QueryType.FACTUAL),
            ("ResNet和VGG的区别", QueryType.COMPARISON),
            ("如何实现RAG系统？", QueryType.TECHNICAL),
            ("介绍一下深度学习在计算机视觉中的应用，包括目标检测、语义分割等", QueryType.COMPLEX)
        ]
        
        for query, expected_type in test_cases:
            print(f"\n🔍 查询: {query}")
            print(f"   预期类型: {expected_type.value}")
            print("-" * 60)
            
            # 自适应检索
            results = adaptive.retrieve(query, k=3)
            
            if results:
                actual_type = results[0].get('metadata', {}).get('query_type', 'unknown')
                strategy = results[0].get('metadata', {}).get('strategy', 'unknown')
                
                print(f"   实际类型: {actual_type}")
                print(f"   使用策略: {strategy}")
                print(f"   返回结果: {len(results)} 个")
                
                match = "✅" if actual_type == expected_type.value else "⚠️"
                print(f"   类型匹配: {match}")
        
        # 显示统计
        stats = adaptive.get_stats()
        print(f"\n📊 统计信息:")
        print(f"   总查询数: {stats['total_queries']}")
        print(f"   类型分布:")
        for query_type, count in stats['query_type_counts'].items():
            if count > 0:
                print(f"     {query_type}: {count}")
        
        print("\n✅ 自适应RAG测试完成")
        
    except Exception as e:
        print(f"\n❌ 自适应RAG测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_strategy_comparison():
    """测试策略对比"""
    print("\n" + "="*80)
    print("📊 测试4: 策略对比 (simple vs hybrid vs enhanced vs full)")
    print("="*80)
    
    try:
        # 初始化
        config = Config()
        llm = create_llm(config)
        
        # 测试查询
        query = "介绍黄瑞楠的工作经历"
        
        print(f"\n🔍 查询: {query}")
        print("="*80)
        
        strategies = ['simple', 'hybrid', 'enhanced', 'full']
        
        for strategy in strategies:
            print(f"\n📌 策略: {strategy}")
            print("-" * 60)
            
            # 临时修改配置
            original_mode = config.RAG_STRATEGY_MODE
            config.RAG_STRATEGY_MODE = strategy
            
            # 创建管道
            pipeline = EnhancedRAGPipeline(config, llm)
            
            # 执行检索
            import time
            start = time.time()
            results = pipeline.retrieve(query, top_k=3)
            elapsed = time.time() - start
            
            # 恢复配置
            config.RAG_STRATEGY_MODE = original_mode
            
            print(f"   耗时: {elapsed:.2f}秒")
            print(f"   结果数: {len(results)}")
            
            if results:
                for i, doc in enumerate(results, 1):
                    score = doc.metadata.get('score', 0.0)
                    text = doc.page_content[:100]
                    print(f"   {i}. 分数={score:.4f} | {text}...")
        
        print("\n✅ 策略对比测试完成")
        
    except Exception as e:
        print(f"\n❌ 策略对比测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🚀 TOP 5 推荐RAG方案 - 完整测试")
    print("="*80)
    
    # 测试1: 融合检索
    test_hybrid_retriever()
    
    # 测试2: 上下文增强
    test_context_enriched()
    
    # 测试3: 自适应RAG
    test_adaptive_rag()
    
    # 测试4: 策略对比
    test_strategy_comparison()
    
    print("\n" + "="*80)
    print("✅ 所有测试完成")
    print("="*80)


if __name__ == "__main__":
    main()
