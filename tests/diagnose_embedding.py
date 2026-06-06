"""
诊断嵌入模型输出维度
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from FlagEmbedding import BGEM3FlagModel
from config import config

print("=" * 60)
print("🔍 嵌入模型诊断")
print("=" * 60)

print(f"\n配置信息:")
print(f"  模型路径: {config.BGEM3_EMBEDDING_MODEL}")
print(f"  配置维度: {config.BGEM3_DENSE_DIM}")

print("\n加载模型...")
model = BGEM3FlagModel(
    config.BGEM3_EMBEDDING_MODEL,
    use_fp16=True
)
print("✅ 模型加载完成")

# 测试编码
test_text = "这是测试文本"
print(f"\n编码测试文本: '{test_text}'")

result = model.encode(
    [test_text],
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False
)

print("\n编码结果:")
print(f"  Keys: {result.keys()}")

if 'dense_vecs' in result:
    dense = result['dense_vecs'][0]
    print(f"  Dense 向量维度: {len(dense)}")
    print(f"  Dense 向量类型: {type(dense)}")
    print(f"  Dense 向量前5维: {dense[:5]}")
    
if 'lexical_weights' in result:
    sparse = result['lexical_weights'][0]
    print(f"  Sparse 向量非零个数: {len(sparse)}")
    print(f"  Sparse 向量类型: {type(sparse)}")

# 测试转换为列表
dense_list = result['dense_vecs'][0].tolist()
print(f"\n转换为列表后:")
print(f"  类型: {type(dense_list)}")
print(f"  长度: {len(dense_list)}")

print("\n" + "=" * 60)
print("✅ 诊断完成")
print("=" * 60)
