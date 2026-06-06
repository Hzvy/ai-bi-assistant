"""
快速测试 Milvus 连接和基本功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pymilvus import connections, utility, Collection
from config import config

def test_connection():
    """测试连接"""
    print("=" * 60)
    print("测试 1: Milvus 连接")
    print("=" * 60)
    
    try:
        connections.connect(
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT
        )
        print(f"✅ 成功连接到 Milvus: {config.MILVUS_HOST}:{config.MILVUS_PORT}")
        
        # 列出所有集合
        collections = utility.list_collections()
        print(f"📋 现有集合: {collections}")
        
        return True
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False
    finally:
        try:
            connections.disconnect("default")
        except:
            pass


def test_bgem3():
    """测试 BGEM3 模型"""
    print("\n" + "=" * 60)
    print("测试 2: BGEM3 模型")
    print("=" * 60)
    
    try:
        from FlagEmbedding import BGEM3FlagModel
        
        print(f"📦 加载模型: {config.BGEM3_EMBEDDING_MODEL}")
        model = BGEM3FlagModel(
            config.BGEM3_EMBEDDING_MODEL,
            use_fp16=config.BGEM3_USE_FP16
        )
        
        # 测试编码
        test_text = "这是一个测试文本"
        result = model.encode(
            [test_text],
            return_dense=True,
            return_sparse=True
        )
        
        print(f"✅ 模型加载成功")
        print(f"   密集向量维度: {result['dense_vecs'].shape}")
        print(f"   稀疏向量: {type(result['lexical_weights'])}")
        
        return True
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_milvus_manager():
    """测试 MilvusManager"""
    print("\n" + "=" * 60)
    print("测试 3: MilvusManager")
    print("=" * 60)
    
    # 先清理可能存在的旧集合
    try:
        connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        if utility.has_collection("quick_test"):
            print("🗑️  删除旧的测试集合...")
            utility.drop_collection("quick_test")
        connections.disconnect("default")
    except Exception as e:
        print(f"⚠️  清理旧集合时出错（可忽略）: {e}")
    
    try:
        from src.vector_db.milvus_manager import MilvusManager
        
        print("🔧 初始化 MilvusManager...")
        manager_config = {
            'host': config.MILVUS_HOST,
            'port': config.MILVUS_PORT,
            'collection_name': "quick_test",
            'dense_dim': config.BGEM3_DENSE_DIM,
            'embedding_model': config.BGEM3_EMBEDDING_MODEL
        }
        manager = MilvusManager(manager_config)
        
        print("✅ MilvusManager 初始化成功")
        
        # 插入测试文档
        print("\n📥 插入测试文档...")
        test_docs = [
            {"text": "商业智能（BI）是一种数据分析技术", "metadata": {"source": "test1"}},
            {"text": "机器学习是人工智能的一个分支", "metadata": {"source": "test2"}},
            {"text": "数据可视化帮助理解复杂数据", "metadata": {"source": "test3"}}
        ]
        
        doc_ids = manager.insert(documents=test_docs)
        print(f"✅ 成功插入 {len(doc_ids)} 个文档")
        
        # 测试检索
        print("\n🔍 测试检索...")
        results = manager.search("什么是BI？", top_k=2)
        print(f"✅ 检索到 {len(results)} 个结果")
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n结果 {i}:")
                print(f"  文本: {result['text'][:50]}...")
                print(f"  分数: {result['score']:.4f}")
        
        # 清理
        print("\n🗑️  清理测试集合...")
        manager.close()
        
        connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        if utility.has_collection("quick_test"):
            utility.drop_collection("quick_test")
            print("✅ 测试集合已删除")
        connections.disconnect("default")
        
        return True
        
    except Exception as e:
        print(f"❌ MilvusManager 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 尝试清理
        try:
            connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
            if utility.has_collection("quick_test"):
                utility.drop_collection("quick_test")
            connections.disconnect("default")
        except:
            pass
        
        return False


def main():
    print("\n🚀 Milvus 快速测试\n")
    
    results = []
    results.append(("Milvus 连接", test_connection()))
    results.append(("BGEM3 模型", test_bgem3()))
    results.append(("MilvusManager", test_milvus_manager()))
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} - {name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！Milvus 集成成功！")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
