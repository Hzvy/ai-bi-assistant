"""
RAG 增强模块测试

测试内容：
1. Milvus 连接测试
2. BGEM3 嵌入模型测试
3. 文档插入测试
4. 混合检索测试
5. 完整 RAG 管道测试
"""
import logging
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from tools.llm_manager import LLMManager
from src.rag.enhanced_rag import EnhancedRAGPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_milvus_connection():
    """测试 Milvus 连接"""
    logger.info("=" * 60)
    logger.info("测试 1: Milvus 连接")
    logger.info("=" * 60)
    
    try:
        from src.vector_db.milvus_manager import MilvusManager
        
        milvus_config = {
            'host': config.MILVUS_HOST,
            'port': config.MILVUS_PORT,
            'collection_name': 'test_collection',
            'embedding_model': config.BGEM3_EMBEDDING_MODEL,
            'dense_dim': config.BGEM3_DENSE_DIM
        }
        
        manager = MilvusManager(milvus_config)
        logger.info("✅ Milvus 连接成功")
        
        manager.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Milvus 连接失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_document_insertion():
    """测试文档插入"""
    logger.info("=" * 60)
    logger.info("测试 2: 文档插入")
    logger.info("=" * 60)
    
    try:
        from src.vector_db.milvus_manager import MilvusManager
        
        milvus_config = {
            'host': config.MILVUS_HOST,
            'port': config.MILVUS_PORT,
            'collection_name': 'test_collection',
            'embedding_model': config.BGEM3_EMBEDDING_MODEL,
            'dense_dim': config.BGEM3_DENSE_DIM
        }
        
        manager = MilvusManager(milvus_config)
        manager.create_collection()
        
        # 准备测试文档
        test_docs = [
            {
                'text': 'BI（商业智能）是一种数据分析技术，帮助企业做出更好的决策。',
                'metadata': {'source': 'test', 'type': 'definition'}
            },
            {
                'text': 'SQL是结构化查询语言，用于管理和查询关系数据库。',
                'metadata': {'source': 'test', 'type': 'definition'}
            },
            {
                'text': '数据可视化通过图表展示数据，包括柱状图、折线图、饼图等。',
                'metadata': {'source': 'test', 'type': 'concept'}
            }
        ]
        
        # 插入文档
        ids = manager.insert(documents=test_docs)
        logger.info(f"✅ 成功插入 {len(ids)} 个文档")
        
        # 获取统计信息
        stats = manager.get_collection_stats()
        logger.info(f"📊 集合统计: {stats}")
        
        manager.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ 文档插入失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_hybrid_search():
    """测试混合检索"""
    logger.info("=" * 60)
    logger.info("测试 3: 混合检索")
    logger.info("=" * 60)
    
    try:
        from src.vector_db.milvus_manager import MilvusManager
        
        milvus_config = {
            'host': config.MILVUS_HOST,
            'port': config.MILVUS_PORT,
            'collection_name': 'test_collection',
            'embedding_model': config.BGEM3_EMBEDDING_MODEL,
            'dense_dim': config.BGEM3_DENSE_DIM
        }
        
        manager = MilvusManager(milvus_config)
        
        # 执行检索
        query = "什么是商业智能？"
        results = manager.search(query=query, top_k=3)
        
        logger.info(f"✅ 检索到 {len(results)} 个结果")
        for i, result in enumerate(results, 1):
            logger.info(f"\n结果 {i}:")
            logger.info(f"  文本: {result['text']}")
            logger.info(f"  分数: {result['score']:.4f}")
            logger.info(f"  元数据: {result['metadata']}")
        
        manager.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ 混合检索失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_enhanced_rag_pipeline():
    """测试完整 RAG 管道"""
    logger.info("=" * 60)
    logger.info("测试 4: 完整 RAG 管道")
    logger.info("=" * 60)
    
    try:
        # 初始化 LLM（使用全局实例）
        from tools.llm_manager import llm_manager
        llm = llm_manager.get_llm()
        
        # 初始化 RAG 管道
        pipeline = EnhancedRAGPipeline(config, llm)
        
        # 测试查询
        queries = [
            "什么是BI？",
            "如何使用SQL？",
            "数据可视化有哪些图表类型？"
        ]
        
        for query in queries:
            logger.info(f"\n查询: {query}")
            results = pipeline.retrieve(query, top_k=2)
            
            if results:
                logger.info(f"✅ 检索到 {len(results)} 个结果")
                for i, result in enumerate(results, 1):
                    logger.info(f"  {i}. {result['text'][:50]}... (分数: {result.get('score', 0):.4f})")
            else:
                logger.info("⚠️ 未找到结果")
        
        pipeline.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG 管道测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """主测试函数"""
    logger.info("🚀 开始 RAG 增强模块测试\n")
    
    tests = [
        ("Milvus 连接", test_milvus_connection),
        ("文档插入", test_document_insertion),
        ("混合检索", test_hybrid_search),
        ("完整 RAG 管道", test_enhanced_rag_pipeline)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"测试 '{name}' 异常: {e}")
            results.append((name, False))
        
        print("\n")
    
    # 打印测试总结
    logger.info("=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"{name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    logger.info(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过!")
    else:
        logger.warning("⚠️ 部分测试失败，请检查日志")


if __name__ == "__main__":
    main()
