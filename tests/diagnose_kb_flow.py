"""
诊断脚本：检查知识库完整流程（加载 → 存储 → 检索）
"""
from pathlib import Path
from tools.kb_loader import load_pdf_file, clean_text
from langchain.schema import Document
from config import config

print("="*80)
print("🔍 知识库完整流程诊断")
print("="*80)

# 第一步：读取 PDF
pdf_path = Path("data/knowledge_base/简历-黄瑞楠-1.pdf")
print(f"\n📄 步骤1：读取 PDF 文件: {pdf_path}")

if not pdf_path.exists():
    print(f"❌ 文件不存在")
    exit(1)

content = load_pdf_file(pdf_path)
print(f"✅ 读取成功: {len(content)} 字符")
print(f"📄 前200字符:\n{content[:200]}")

# 第二步：清洗文本
print(f"\n🧹 步骤2：清洗文本")
cleaned_content = clean_text(content, debug=True)
print(f"✅ 清洗完成: {len(cleaned_content)} 字符")
print(f"📄 清洗后前200字符:\n{cleaned_content[:200]}")

# 第三步：创建 Document 对象
print(f"\n📦 步骤3：创建 Document 对象")
doc = Document(
    page_content=cleaned_content,
    metadata={
        "source": f"《{pdf_path.stem}》",
        "file_path": str(pdf_path),
        "file_type": ".pdf",
        "chunk_index": 0,
        "total_chunks": 1
    }
)
print(f"✅ Document 创建成功")
print(f"   page_content 长度: {len(doc.page_content)}")
print(f"   page_content 前200字符: {doc.page_content[:200]}")
print(f"   metadata: {doc.metadata}")

# 第四步：存储到 Milvus
print(f"\n💾 步骤4：存储到 Milvus")
try:
    from tools.tools_rag import rag_manager
    
    # 初始化知识库
    rag_manager.initialize([doc])
    print(f"✅ 存储成功")
    
    # 检查知识库状态
    if rag_manager.is_kb_available():
        print(f"✅ 知识库可用")
        print(f"   模式: {'增强' if rag_manager.is_enhanced_mode() else '标准'}")
    else:
        print(f"❌ 知识库不可用")
        exit(1)
    
except Exception as e:
    print(f"❌ 存储失败: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# 第五步：从 Milvus 读取
print(f"\n🔍 步骤5：从 Milvus 读取数据")
try:
    from pymilvus import connections, Collection
    
    connections.connect(
        alias="default",
        host=config.MILVUS_HOST,
        port=config.MILVUS_PORT
    )
    
    collection = Collection(config.MILVUS_COLLECTION_NAME)
    collection.load()
    
    # 查询刚刚存储的文档
    results = collection.query(
        expr=f'metadata["source"] == "《{pdf_path.stem}》"',
        output_fields=["text", "metadata"],
        limit=10
    )
    
    print(f"✅ 查询成功: 找到 {len(results)} 个分片")
    
    for i, item in enumerate(results, 1):
        text = item.get("text", "")
        metadata = item.get("metadata", {})
        print(f"\n   分片 {i}:")
        print(f"   - 来源: {metadata.get('source', '未知')}")
        print(f"   - 文本长度: {len(text)}")
        print(f"   - 文本前200字符: {text[:200]}")
        
except Exception as e:
    print(f"❌ 读取失败: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# 第六步：使用 RAG 检索
print(f"\n🔍 步骤6：使用 RAG 检索")
try:
    test_queries = [
        "黄瑞楠",
        "简历",
        "项目",
        "工场安全检测"
    ]
    
    for query in test_queries:
        print(f"\n   查询: '{query}'")
        results = rag_manager.retrieve(query, k=3)
        print(f"   结果: {len(results)} 个")
        
        if results:
            for i, doc in enumerate(results, 1):
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                print(f"      {i}. [{len(content)}字符] {content[:100]}...")
        else:
            print(f"      ❌ 未找到结果")
            
except Exception as e:
    print(f"❌ 检索失败: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("✅ 诊断完成")
print("="*80)
