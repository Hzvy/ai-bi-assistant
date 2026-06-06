"""
测试分级权限知识库系统
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_db.milvus_manager import MilvusManager
from pymilvus import utility, connections
from config import config


def test_permission_system():
    """测试权限过滤功能"""
    print("=" * 60)
    print("测试分级权限知识库系统")
    print("=" * 60)
    
    # 配置
    milvus_config = {
        'host': config.MILVUS_HOST,
        'port': config.MILVUS_PORT,
        'collection_name': config.MILVUS_COLLECTION_NAME + "_test",  # 使用测试集合名
        'dense_dim': config.BGEM3_DENSE_DIM,
        'embedding_model': config.BGEM3_EMBEDDING_MODEL
    }
    
    try:
        # 初始化 Milvus
        print("\n1. 连接 Milvus...")
        milvus = MilvusManager(milvus_config)
        
        # 删除旧的测试集合（如果存在）
        print("\n2. 清理旧的测试集合...")
        if utility.has_collection(milvus_config['collection_name']):
            utility.drop_collection(milvus_config['collection_name'])
            print(f"   ✅ 已删除旧集合: {milvus_config['collection_name']}")
        
        # 创建新集合
        print("\n3. 创建新集合...")
        milvus.create_collection()
        
        # 插入测试文档（不同权限级别）
        print("\n4. 插入测试文档...")
        test_docs = [
            {
                'text': '这是公开文档，所有用户都可以看到',
                'metadata': {'source': 'public.txt'},
                'kb_level': 1,
                'kb_category': 'public',
                'department': ''
            },
            {
                'text': '这是内部文档，仅二级及以上权限可见',
                'metadata': {'source': 'internal.txt'},
                'kb_level': 2,
                'kb_category': 'internal',
                'department': '技术部'
            },
            {
                'text': '这是机密文档，仅三级管理员可见',
                'metadata': {'source': 'confidential.txt'},
                'kb_level': 3,
                'kb_category': 'confidential',
                'department': '技术部'
            }
        ]
        
        ids = milvus.insert(documents=test_docs)
        print(f"✅ 插入 {len(ids)} 个文档")
        
        # 测试不同权限级别的检索
        test_query = "文档"
        
        print("\n" + "=" * 60)
        print("5. 测试权限过滤")
        print("=" * 60)
        
        # Level 1 用户（仅公开）
        print("\n【Level 1 用户检索】")
        results_l1 = milvus.search(test_query, top_k=10, user_level=1)
        print(f"检索到 {len(results_l1)} 个结果：")
        for r in results_l1:
            print(f"  - Level {r['kb_level']}: {r['text'][:30]}...")
        
        # Level 2 用户（公开+内部）
        print("\n【Level 2 用户检索】")
        results_l2 = milvus.search(test_query, top_k=10, user_level=2)
        print(f"检索到 {len(results_l2)} 个结果：")
        for r in results_l2:
            print(f"  - Level {r['kb_level']}: {r['text'][:30]}...")
        
        # Level 3 用户（全部）
        print("\n【Level 3 用户检索】")
        results_l3 = milvus.search(test_query, top_k=10, user_level=3)
        print(f"检索到 {len(results_l3)} 个结果：")
        for r in results_l3:
            print(f"  - Level {r['kb_level']}: {r['text'][:30]}...")
        
        # 验证结果
        print("\n" + "=" * 60)
        print("6. 验证结果")
        print("=" * 60)
        
        assert len(results_l1) == 1, f"Level 1 应返回 1 个文档，实际 {len(results_l1)}"
        assert len(results_l2) == 2, f"Level 2 应返回 2 个文档，实际 {len(results_l2)}"
        assert len(results_l3) == 3, f"Level 3 应返回 3 个文档，实际 {len(results_l3)}"
        
        print("✅ Level 1 权限测试通过：仅返回公开文档")
        print("✅ Level 2 权限测试通过：返回公开+内部文档")
        print("✅ Level 3 权限测试通过：返回所有文档")
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！权限系统工作正常")
        print("=" * 60)
        
        # 清理测试集合
        print("\n7. 清理测试集合...")
        utility.drop_collection(milvus_config['collection_name'])
        print(f"✅ 测试集合已删除: {milvus_config['collection_name']}")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理测试集合（即使失败也要清理）
        try:
            if utility.has_collection(milvus_config['collection_name']):
                utility.drop_collection(milvus_config['collection_name'])
                print(f"✅ 测试集合已清理: {milvus_config['collection_name']}")
        except:
            pass
            
        return False
    
    return True


if __name__ == "__main__":
    success = test_permission_system()
    sys.exit(0 if success else 1)
