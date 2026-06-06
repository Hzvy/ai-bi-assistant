"""
清空 Milvus 知识库（删除所有测试数据）
"""
import os
import sys

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

print("=" * 60)
print("🗑️  清空 Milvus 知识库")
print("=" * 60)

from config import config
from pymilvus import connections, Collection, utility

# 连接 Milvus
print("\n【1】连接 Milvus")
connections.connect(
    alias="default",
    host=config.MILVUS_HOST,
    port=config.MILVUS_PORT
)
print(f"   ✅ 已连接: {config.MILVUS_HOST}:{config.MILVUS_PORT}")

collection_name = config.MILVUS_COLLECTION_NAME
print(f"\n【2】检查集合: {collection_name}")

if utility.has_collection(collection_name):
    collection = Collection(collection_name)
    collection.load()
    
    count_before = collection.num_entities
    print(f"   清理前文档数: {count_before}")
    
    # 确认删除
    print(f"\n   ⚠️  即将删除集合 '{collection_name}' 的所有数据！")
    confirm = input("   确认删除? (输入 yes 继续): ")
    
    if confirm.lower() == 'yes':
        print("\n【3】删除集合...")
        
        # 删除集合
        utility.drop_collection(collection_name)
        print(f"   ✅ 集合已删除")
        
        print(f"\n   💡 下次启动 Streamlit 时，上传文档会自动创建新集合")
    else:
        print("\n   ❌ 取消删除")
else:
    print(f"   ⚠️  集合不存在，无需清理")

connections.disconnect("default")

print("\n" + "=" * 60)
print("🏁 完成")
print("=" * 60)
