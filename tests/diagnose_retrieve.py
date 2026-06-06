"""
诊断知识库检索问题
"""
import os
import sys

# 设置环境变量
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-3fcbf76cbcd648e3a4bfa79a65f19a88")

from config import config
from tools.tools_rag import rag_manager

print("\n" + "="*80)
print("🔍 知识库检索诊断")
print("="*80)

# 1. 检查知识库状态
print("\n📊 知识库状态:")
print(f"   可用: {rag_manager.is_kb_available()}")
print(f"   增强模式: {rag_manager.is_enhanced_mode()}")
print(f"   向量库类型: {config.VECTOR_DB_TYPE}")

# 2. 检查 Milvus 连接和文档数
if config.VECTOR_DB_TYPE.lower() == "milvus":
    try:
        from pymilvus import connections, Collection
        
        connections.connect(
            alias="default",
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT
        )
        
        collection = Collection(config.MILVUS_COLLECTION_NAME)
        collection.load()
        
        num_entities = collection.num_entities
        print(f"\n📦 Milvus 集合状态:")
        print(f"   集合名: {config.MILVUS_COLLECTION_NAME}")
        print(f"   文档数: {num_entities}")
        
        # 查询前3个文档
        if num_entities > 0:
            results = collection.query(
                expr="",
                output_fields=["text", "metadata"],
                limit=3
            )
            print(f"\n📄 前3个文档示例:")
            for i, item in enumerate(results, 1):
                text = item.get("text", "")[:100]
                metadata = item.get("metadata", {})
                source = metadata.get("source", "未知")
                print(f"   {i}. 来源: {source}")
                print(f"      内容: {text}...")
    except Exception as e:
        print(f"\n❌ Milvus 检查失败: {str(e)}")
        import traceback
        traceback.print_exc()

# 3. 测试相关性检查
print("\n" + "="*80)
print("🔍 测试相关性检查")
print("="*80)

test_query = "黄瑞楠"

if rag_manager.is_enhanced_mode():
    try:
        from src.rag.relevance_checker import RelevanceChecker
        from tools.llm_manager import LLMManager
        
        # 获取 LLM
        llm_manager = LLMManager()
        llm = llm_manager.get_llm()
        
        # 创建相关性检查器
        checker = RelevanceChecker(llm)
        
        print(f"\n测试问题: '{test_query}'")
        is_relevant = checker.is_relevant(test_query)
        print(f"相关性判断: {'✅ 相关' if is_relevant else '❌ 不相关'}")
        
        if not is_relevant:
            print("\n⚠️ 问题：相关性检查拒绝了该查询！")
            print("   原因：LLM 认为这个问题与知识库无关")
            print("   解决：需要调整相关性检查的提示词")
    except Exception as e:
        print(f"\n❌ 相关性检查测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print("\n💡 未启用增强模式，跳过相关性检查测试")

# 4. 测试直接检索（绕过相关性检查）
print("\n" + "="*80)
print("🔍 测试直接检索（绕过相关性检查）")
print("="*80)

if config.VECTOR_DB_TYPE.lower() == "milvus" and rag_manager.milvus_manager:
    try:
        print(f"\n测试问题: '{test_query}'")
        
        # 直接使用 Milvus 搜索
        results = rag_manager.milvus_manager.search(test_query, top_k=3)
        
        print(f"\n检索结果: {len(results)} 个")
        for i, doc in enumerate(results, 1):
            if isinstance(doc, dict):
                text = doc.get("text", "")[:100]
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "未知")
                score = doc.get("score", 0)
            else:
                text = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)[:100]
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                source = metadata.get("source", "未知")
                score = metadata.get("score", 0)
            
            print(f"\n   结果 {i}:")
            print(f"   来源: {source}")
            print(f"   相似度: {score:.4f}")
            print(f"   内容: {text}...")
    except Exception as e:
        print(f"\n❌ 直接检索失败: {str(e)}")
        import traceback
        traceback.print_exc()

# 5. 测试完整的 RAG 管道
print("\n" + "="*80)
print("🔍 测试完整的 RAG 检索管道")
print("="*80)

try:
    print(f"\n测试问题: '{test_query}'")
    
    # 使用 RAG Manager 的 retrieve 方法
    results = rag_manager.retrieve(test_query, k=3)
    
    print(f"\n检索结果: {len(results)} 个")
    if len(results) == 0:
        print("❌ 未检索到任何结果！")
        print("\n可能的原因:")
        print("   1. 相关性检查拒绝了查询")
        print("   2. 向量检索没有找到相似文档")
        print("   3. 重排序过滤掉了所有结果")
    else:
        for i, doc in enumerate(results, 1):
            if isinstance(doc, dict):
                text = doc.get("text", "")[:100]
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "未知")
            else:
                text = doc.page_content[:100] if hasattr(doc, 'page_content') else str(doc)[:100]
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                source = metadata.get("source", "未知")
            
            print(f"\n   结果 {i}:")
            print(f"   来源: {source}")
            print(f"   内容: {text}...")
except Exception as e:
    print(f"\n❌ RAG 管道测试失败: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("✅ 诊断完成")
print("="*80)
