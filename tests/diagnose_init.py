"""
诊断知识库初始化问题
"""
import os
import sys
import traceback

# 设置环境
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🔍 知识库初始化诊断工具")
print("=" * 60)

# 1. 检查配置
print("\n【1】检查配置文件")
try:
    from config import config
    print(f"✅ Config 加载成功")
    print(f"   VECTOR_DB_TYPE: {config.VECTOR_DB_TYPE}")
    print(f"   BGEM3_EMBEDDING_MODEL: {config.BGEM3_EMBEDDING_MODEL}")
    print(f"   BGEM3_DENSE_DIM: {config.BGEM3_DENSE_DIM}")
    print(f"   MILVUS_HOST: {config.MILVUS_HOST}")
    print(f"   MILVUS_PORT: {config.MILVUS_PORT}")
except Exception as e:
    print(f"❌ Config 加载失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 2. 检查 Milvus 连接
print("\n【2】检查 Milvus 连接")
if config.VECTOR_DB_TYPE == "milvus":
    try:
        from pymilvus import connections, utility
        
        print(f"   正在连接 {config.MILVUS_HOST}:{config.MILVUS_PORT}...")
        connections.connect(
            alias="default",
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT
        )
        print(f"✅ Milvus 连接成功")
        
        # 列出所有集合
        collections = utility.list_collections()
        print(f"   现有集合: {collections}")
        
        connections.disconnect("default")
    except Exception as e:
        print(f"❌ Milvus 连接失败: {e}")
        traceback.print_exc()
else:
    print(f"⚠️ 跳过（当前配置: {config.VECTOR_DB_TYPE}）")

# 3. 检查 BGEM3 模型
print("\n【3】检查 BGEM3 模型")
if config.VECTOR_DB_TYPE == "milvus":
    try:
        print(f"   正在加载模型: {config.BGEM3_EMBEDDING_MODEL}...")
        from FlagEmbedding import BGEM3FlagModel
        
        model = BGEM3FlagModel(
            config.BGEM3_EMBEDDING_MODEL,
            use_fp16=config.BGEM3_USE_FP16
        )
        print(f"✅ BGEM3 模型加载成功")
        
        # 测试编码
        print(f"   测试编码...")
        result = model.encode(
            ["测试文本"],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        print(f"   密集向量维度: {len(result['dense_vecs'][0])}")
        print(f"   稀疏向量: {len(result['lexical_weights'][0])} 个非零元素")
        print(f"✅ 编码测试成功")
        
    except Exception as e:
        print(f"❌ BGEM3 模型加载失败: {e}")
        traceback.print_exc()
else:
    print(f"⚠️ 跳过（当前配置: {config.VECTOR_DB_TYPE}）")

# 4. 检查增强型 RAG 组件
print("\n【4】检查增强型 RAG 组件")
try:
    from src.rag.enhanced_rag import EnhancedRAGPipeline
    print(f"✅ EnhancedRAGPipeline 导入成功")
except Exception as e:
    print(f"❌ EnhancedRAGPipeline 导入失败: {e}")
    traceback.print_exc()

# 5. 测试完整初始化流程
print("\n【5】测试完整初始化流程")
try:
    from tools.tools_rag import rag_manager
    from langchain.schema import Document
    
    # 创建测试文档
    test_docs = [
        Document(
            page_content="这是第一个测试文档，用于验证知识库初始化。",
            metadata={"source": "test1.txt"}
        ),
        Document(
            page_content="这是第二个测试文档，用于验证向量存储功能。",
            metadata={"source": "test2.txt"}
        )
    ]
    
    print(f"   正在初始化知识库（{len(test_docs)} 个测试文档）...")
    rag_manager.initialize(test_docs)
    
    if rag_manager.is_kb_available():
        print(f"✅ 知识库初始化成功")
        print(f"   模式: {'增强型 (Milvus)' if rag_manager.is_enhanced_mode() else '标准 (Chroma)'}")
        
        # 测试检索
        print(f"\n   测试检索...")
        results = rag_manager.retrieve("测试", k=2)
        print(f"   检索到 {len(results)} 个结果")
        
        if results:
            print(f"   第一个结果: {results[0].page_content[:50]}...")
            print(f"✅ 检索测试成功")
        else:
            print(f"⚠️ 检索结果为空")
    else:
        print(f"❌ 知识库不可用")
        
except Exception as e:
    print(f"❌ 初始化失败: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("🏁 诊断完成")
print("=" * 60)
