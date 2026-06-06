"""
诊断 PDF 上传和知识库加载问题
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("🔍 PDF 上传诊断工具")
print("=" * 60)

# 1. 检查知识库目录
print("\n【1】检查知识库目录")
kb_path = Path("data/knowledge_base")
print(f"   路径: {kb_path.absolute()}")
print(f"   存在: {kb_path.exists()}")

if kb_path.exists():
    files = list(kb_path.glob("**/*.*"))
    print(f"   文件数: {len(files)}")
    
    if files:
        print("\n   文件列表:")
        for file in files:
            size_kb = file.stat().st_size / 1024
            print(f"      - {file.name} ({size_kb:.1f} KB)")
    else:
        print("   ⚠️ 目录为空")
else:
    print("   ❌ 目录不存在")

# 2. 测试文件读取
print("\n【2】测试文件读取")
if kb_path.exists():
    pdf_files = list(kb_path.glob("**/*.pdf"))
    txt_files = list(kb_path.glob("**/*.txt"))
    md_files = list(kb_path.glob("**/*.md"))
    
    print(f"   PDF: {len(pdf_files)} 个")
    print(f"   TXT: {len(txt_files)} 个")
    print(f"   MD: {len(md_files)} 个")
    
    # 测试读取第一个 PDF
    if pdf_files:
        test_pdf = pdf_files[0]
        print(f"\n   测试读取: {test_pdf.name}")
        
        try:
            from tools.kb_loader import load_pdf_file
            content = load_pdf_file(test_pdf)
            
            if content:
                print(f"   ✅ 读取成功")
                print(f"   内容长度: {len(content)} 字符")
                print(f"   前100字符: {content[:100]}...")
            else:
                print(f"   ❌ 读取失败（内容为空）")
        except Exception as e:
            print(f"   ❌ 读取失败: {e}")
            import traceback
            traceback.print_exc()

# 3. 测试知识库加载
print("\n【3】测试知识库加载")
try:
    from tools.kb_loader import load_kb_from_files
    
    if kb_path.exists():
        print("   正在加载知识库...")
        num_docs = load_kb_from_files(str(kb_path))
        print(f"   加载结果: {num_docs} 个文档")
        
        if num_docs == 0:
            print("   ⚠️ 没有加载任何文档！")
            print("\n   可能原因:")
            print("      1. PDF 文件无法解析")
            print("      2. 文件内容为空")
            print("      3. 文本分割后为空")
            print("      4. 向量库插入失败")
    else:
        print("   ❌ 知识库目录不存在")
        
except Exception as e:
    print(f"   ❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 检查向量库状态
print("\n【4】检查向量库状态")
try:
    from tools.tools_rag import rag_manager, get_kb_status
    
    status = get_kb_status()
    print(f"   available: {status['available']}")
    print(f"   mode: {status['mode']}")
    print(f"   vector_db: {status['vector_db']}")
    print(f"   message: {status['message']}")
    
    if status['available']:
        # 测试检索
        print("\n   测试检索...")
        results = rag_manager.retrieve("测试查询", k=3)
        print(f"   检索结果数: {len(results)}")
        
        if results:
            print(f"\n   第一个结果:")
            print(f"      内容: {results[0].page_content[:100]}...")
        else:
            print("   ⚠️ 检索结果为空")
            
            # 如果是 Milvus，检查集合文档数
            if rag_manager.is_enhanced_mode():
                try:
                    collection = rag_manager.enhanced_pipeline.vector_db.collection
                    if collection:
                        count = collection.num_entities
                        print(f"\n   Milvus 集合文档数: {count}")
                        
                        if count == 0:
                            print("   ⚠️ Milvus 集合为空！")
                            print("      可能原因: 文档插入失败")
                except Exception as e:
                    print(f"   ❌ 无法检查集合: {e}")
    else:
        print("   ⚠️ 知识库不可用")
        
except Exception as e:
    print(f"   ❌ 检查失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🏁 诊断完成")
print("=" * 60)
