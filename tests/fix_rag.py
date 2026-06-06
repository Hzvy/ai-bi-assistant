"""
快速修复脚本 - 更新 tools_rag.py 以支持 Milvus
"""
import os

# 读取当前文件
file_path = r"D:\PyProjects\LLM\工业大模型\ai-bi-assistant-v3\tools\tools_rag.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 替换有问题的部分
# 1. 修复初始化逻辑
old_init = '''    def __init__(self):
        self.vectordb = None
        self.embeddings = None
        self._kb_available = False
        self.enhanced_pipeline = None
        
        # 如果启用增强型 RAG 且 Milvus 可用
        if ENHANCED_RAG_AVAILABLE and config.VECTOR_DB_TYPE == "milvus":
            try:
                self.enhanced_pipeline = EnhancedRAGPipeline()
                print("✅ 增强型 RAG 管道已启用")
            except Exception as e:
                print(f"⚠️ 增强型 RAG 初始化失败，回退到标准 RAG: {e}")
                self.enhanced_pipeline = None'''

new_init = '''    def __init__(self):
        self.vectordb = None
        self.embeddings = None
        self._kb_available = False
        self.enhanced_pipeline = None
        self._llm = None  # 保存 LLM 实例用于延迟初始化
        
    def _init_enhanced_pipeline(self):
        """延迟初始化增强型 RAG 管道"""
        if ENHANCED_RAG_AVAILABLE and config.VECTOR_DB_TYPE == "milvus" and not self.enhanced_pipeline:
            try:
                # 获取 LLM 实例
                if not self._llm:
                    from tools.llm_manager import llm_manager
                    self._llm = llm_manager.get_llm()
                
                from src.rag.enhanced_rag import EnhancedRAGPipeline
                self.enhanced_pipeline = EnhancedRAGPipeline(config, self._llm)
                print("✅ 增强型 RAG 管道已启用")
                return True
            except Exception as e:
                print(f"⚠️ 增强型 RAG 初始化失败，回退到标准 RAG: {e}")
                import traceback
                traceback.print_exc()
                self.enhanced_pipeline = None
                return False
        return self.enhanced_pipeline is not None'''

content = content.replace(old_init, new_init)

# 2. 修复 initialize 方法中的数据库选择逻辑
old_db_logic = '''            # 根据配置选择向量数据库
            if self.is_enhanced_mode():'''

new_db_logic = '''            # 根据配置选择向量数据库
            if config.VECTOR_DB_TYPE == "milvus":
                # 初始化增强型 RAG
                self._init_enhanced_pipeline()
                
            if self.is_enhanced_mode():'''

content = content.replace(old_db_logic, new_db_logic)

# 保存文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ tools_rag.py 已更新")
print("   - 添加了延迟初始化逻辑")
print("   - 修复了 Milvus 集成")
print("\n请重新启动应用测试")
