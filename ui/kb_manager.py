"""
知识库管理 UI 组件 - 文件上传、知识库更新

【功能说明】
Streamlit 知识库管理界面，提供文件上传、状态查看、知识库重载功能。

【核心功能】
1. **文件上传**
   - 支持格式: .txt, .md, .pdf
   - 批量上传（多文件）
   - 自动保存到 data/knowledge_base/
   - 上传后自动更新知识库

2. **知识库状态**
   - 显示文档总数
   - 显示向量数据库状态
   - 显示已上传文件列表
   - 支持单文件删除

3. **知识库管理**
   - 重新加载知识库
   - 清空知识库
   - 文本清洗开关
   - 智能分割开关

【数据流】
```
上传文件 → 保存到 data/knowledge_base/ → load_kb_from_files()
    ↓
文本清洗（可选）→ 智能分割（可选）→ 向量化 → Milvus.insert()
    ↓
更新 Session State → 重新初始化 Agent
```

【UI 结构】
- Tab 1: 📤 上传文件
  - 文件选择器（multi-file uploader）
  - 文件列表预览
  - 上传按钮
  - 进度提示

- Tab 2: 📊 知识库状态
  - 文档统计（总数、向量数）
  - 文件列表（带删除按钮）
  - 刷新按钮

- Tab 3: ⚙️ 管理
  - 重新加载知识库
  - 清空知识库
  - 文本清洗开关
  - 智能分割开关

【配置项】
- config.TEXT_CLEANING_ENABLED: 是否启用文本清洗
- config.SMART_SPLIT_ENABLED: 是否启用智能分割
- config.CUSTOM_CLEAN_PATTERNS: 自定义清洗规则

【使用方式】
```python
# 在 main.py 侧边栏调用
from ui.kb_manager import render_kb_manager

with st.sidebar:
    render_kb_manager()
```

【注意事项】
- 上传后需点击"重新加载知识库"才能生效
- 删除文件会同时删除向量数据
- 清空知识库不可恢复，需确认操作

【版本】
V3 - 2025-10
"""
import streamlit as st
import os
import time
from pathlib import Path
from typing import List
import shutil
from tools.kb_loader import load_kb_from_files, get_kb_status
from tools.tools_rag import rag_manager
from tools.system_state_manager import SystemStateManager
from langchain.schema import Document
from config import config


def render_kb_manager():
    """
    渲染知识库管理界面
    
    功能说明:
        Streamlit 知识库管理 UI 的主入口函数。
        创建三个标签页：上传文件、知识库状态、管理。
    
    UI 组件:
        - Tab 1: 文件上传器 + 保存逻辑
        - Tab 2: 知识库状态展示 + 文件删除
        - Tab 3: 重载/清空知识库 + 配置开关
    
    Session State 交互:
        - kb_initialized: 知识库是否已初始化
        - need_reinit_agent: 是否需要重新初始化 Agent
        - last_kb_state: 上次知识库状态
    
    使用示例:
        >>> # 在 main.py 侧边栏调用
        >>> with st.sidebar:
        ...     render_kb_manager()
    
    注意事项:
        - 此函数包含 Streamlit 组件，仅在 Streamlit 环境运行
        - 文件操作会触发 Session State 变更
        - 需确保 data/knowledge_base/ 目录存在
    """
    
    st.markdown("## 📚 知识库管理")
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["📤 上传文件", "📊 知识库状态", "⚙️ 管理"])
    
    # ===== Tab 1: 上传文件 =====
    with tab1:
        st.markdown("### 上传文件到知识库")
        
        # ===== 新增：知识库级别选择器 =====
        kb_level = st.selectbox(
            "🔐 知识库级别",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "🌐 一级 - 公开 (所有用户可见)",
                2: "🔒 二级 - 内部 (二级及以上权限)",
                3: "🔐 三级 - 机密 (仅三级管理员)"
            }[x],
            help="选择上传文档的访问权限级别"
        )
        
        # 显示级别说明
        level_desc = {
            1: "📖 公开文档：产品手册、常见问题、公开资料",
            2: "📚 内部文档：内部流程、技术文档、培训资料",
            3: "🔒 机密文档：战略规划、财务数据、核心技术"
        }
        st.info(level_desc[kb_level])
        
        # 权限验证
        user_level = st.session_state.get("user_info", {}).get("access_level", 1)
        if kb_level > user_level:
            st.error(f"❌ 您的权限级别 ({user_level}) 不足以上传 {kb_level} 级文档！")
            st.stop()
        
        st.markdown("---")
        
        # 文件上传器
        uploaded_files = st.file_uploader(
            "选择要上传的文件 (支持 .txt, .md, .pdf)",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
            help="可以一次上传多个文件"
        )
        
        if uploaded_files:
            st.info(f"✅ 已选择 {len(uploaded_files)} 个文件")
            
            # 显示文件列表
            with st.expander("📋 查看已选择的文件"):
                for file in uploaded_files:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📄 {file.name} ({file.size / 1024:.1f} KB)")
                    with col2:
                        st.caption(file.type)
            
            # 上传按钮
            if st.button("🚀 上传并更新知识库", use_container_width=True, type="primary"):
                with st.spinner("正在处理文件..."):
                    try:
                        # 获取知识库目录
                        kb_path = Path("data/knowledge_base")
                        kb_path.mkdir(parents=True, exist_ok=True)
                        
                        # 保存上传的文件
                        saved_files = []
                        errors = []
                        
                        for uploaded_file in uploaded_files:
                            try:
                                file_path = kb_path / uploaded_file.name
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                saved_files.append(str(file_path))
                                st.success(f"✅ 已保存: {uploaded_file.name}")
                            except Exception as save_error:
                                errors.append(f"{uploaded_file.name}: {str(save_error)}")
                                st.error(f"❌ 保存失败: {uploaded_file.name}")
                        
                        # 重新加载知识库（启用文本清洗和智能分割 + 权限级别）
                        if saved_files:
                            st.info(f"🔄 正在加载知识库（Level {kb_level}）...")
                            
                            # 获取用户部门信息
                            user_info = st.session_state.get("user_info", {})
                            user_department = user_info.get("department", "")
                            
                            num_docs = load_kb_from_files(
                                str(kb_path),
                                enable_cleaning=config.TEXT_CLEANING_ENABLED,
                                enable_smart_split=config.SMART_SPLIT_ENABLED,
                                custom_clean_patterns=config.CUSTOM_CLEAN_PATTERNS if config.CUSTOM_CLEAN_PATTERNS else None,
                                kb_level=kb_level,  # ← 传递权限级别
                                department=user_department  # ← 传递部门信息
                            )
                            
                            if num_docs > 0:
                                # 调用状态管理器更新系统状态
                                success, message = SystemStateManager.on_knowledge_base_updated(num_docs)
                                
                                if success:
                                    st.success(f"✅ 知识库已更新！加载了 {num_docs} 篇文档")
                                    st.balloons()
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.warning(f"⚠️ {message}")
                            else:
                                st.warning("⚠️ 没有成功加载任何文档")
                                if errors:
                                    with st.expander("📋 查看加载错误"):
                                        for error in errors:
                                            st.error(error)
                        else:
                            st.error("❌ 没有成功保存任何文件")
                    
                    except Exception as e:
                        st.error(f"❌ 上传失败: {str(e)}")
    
    # ===== Tab 2: 知识库状态 =====
    with tab2:
        st.markdown("### 📊 知识库状态")
        
        col1, col2, col3 = st.columns(3)
        
        # 知识库初始化状态
        kb_initialized = st.session_state.get("kb_initialized", False)
        with col1:
            status = "✅ 已初始化" if kb_initialized else "❌ 未初始化"
            st.write(f"**初始化**: {status}")
        
        # 文档数量
        doc_count = st.session_state.get("kb_doc_count", 0)
        with col2:
            st.write(f"**文档数**: {doc_count}")
        
        # 最后更新时间
        last_update = st.session_state.get("kb_last_update", "从未更新")
        with col3:
            st.write(f"**更新时间**: {last_update}")
        
        st.divider()
        
        # 知识库文件列表
        st.markdown("### 📁 知识库文件")
        kb_path = Path("data/knowledge_base")
        
        if kb_path.exists():
            files = list(kb_path.glob("**/*.*"))
            if files:
                for file in files:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"📄 {file.name}")
                    with col2:
                        size_kb = file.stat().st_size / 1024
                        st.caption(f"{size_kb:.1f} KB")
                    with col3:
                        if st.button("🗑️", key=f"delete_{file}", help="删除此文件"):
                            try:
                                file.unlink()
                                st.success(f"已删除: {file.name}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"删除失败: {str(e)}")
            else:
                st.info("📭 知识库为空，请上传文件")
        else:
            st.warning("📭 知识库目录不存在")
        
        st.divider()
        
        # 向量库文档列表
        st.markdown("### 📄 向量库文档")
        
        if kb_initialized and config.VECTOR_DB_TYPE.lower() == "milvus":
            try:
                from pymilvus import connections, Collection
                
                # 连接 Milvus
                connections.connect(
                    alias="default",
                    host=config.MILVUS_HOST,
                    port=config.MILVUS_PORT
                )
                
                collection = Collection(config.MILVUS_COLLECTION_NAME)
                collection.load()
                
                # 查询所有文档
                results = collection.query(
                    expr="",
                    output_fields=["text", "metadata"],
                    limit=100
                )
                
                if results:
                    # 按来源分组
                    from collections import defaultdict
                    sources = defaultdict(list)
                    for item in results:
                        metadata = item.get("metadata", {})
                        source = metadata.get("source", "未知来源")
                        text = item.get("text", "")
                        sources[source].append(text[:100])  # 只保留前100字符
                    
                    st.info(f"共 {len(results)} 个文档分片，来自 {len(sources)} 个来源")
                    
                    # 显示每个来源的文档
                    for idx, (source, texts) in enumerate(sources.items()):
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            with st.expander(f"📄 {source} ({len(texts)}个分片)"):
                                for i, text in enumerate(texts[:3], 1):  # 只显示前3个
                                    st.caption(f"{i}. {text}...")
                                if len(texts) > 3:
                                    st.caption(f"... 还有 {len(texts) - 3} 个分片")
                        
                        with col2:
                            # 删除按钮
                            if st.button("🗑️ 删除", key=f"delete_source_{idx}", help=f"删除来源: {source}"):
                                try:
                                    # 1. 从向量库中删除
                                    expr = f'metadata["source"] == "{source}"'
                                    collection.delete(expr)
                                    
                                    # 2. 从文件系统中删除源文件
                                    # source 可能是完整路径或文件名，需要处理
                                    file_to_delete = None
                                    
                                    # 尝试从 source 中提取文件名
                                    if "\\" in source or "/" in source:
                                        # source 是完整路径
                                        file_to_delete = Path(source)
                                    else:
                                        # source 只是文件名，在 knowledge_base 目录中查找
                                        kb_path = Path("data/knowledge_base")
                                        file_to_delete = kb_path / source
                                    
                                    # 删除文件
                                    if file_to_delete and file_to_delete.exists():
                                        file_to_delete.unlink()
                                        st.success(f"✅ 已删除文件: {file_to_delete.name}")
                                    else:
                                        st.warning(f"⚠️ 源文件未找到: {source}")
                                    
                                    # 3. 从向量库删除成功的提示
                                    st.success(f"✅ 已从向量库删除: {source}")
                                    
                                    # 4. 更新知识库状态
                                    remaining = collection.num_entities
                                    SystemStateManager.on_knowledge_base_updated(remaining)
                                    
                                    time.sleep(0.5)
                                    st.rerun()
                                except Exception as delete_error:
                                    st.error(f"❌ 删除失败: {str(delete_error)}")
                                    import traceback
                                    st.error(traceback.format_exc())
                else:
                    st.info("📭 向量库为空")
                    
            except Exception as e:
                st.warning(f"⚠️ 无法读取向量库: {str(e)}")
        elif kb_initialized and config.VECTOR_DB_TYPE.lower() == "chroma":
            st.info("💡 ChromaDB 模式下暂不支持文档列表显示")
        else:
            st.info("💡 请先初始化知识库")
    
    # ===== Tab 3: 管理操作 =====
    with tab3:
        st.markdown("### ⚙️ 管理操作")
        
        # RAG策略选择器
        st.markdown("#### 🎯 RAG检索策略")
        
        # 初始化RAG策略
        if "rag_strategy" not in st.session_state:
            st.session_state.rag_strategy = "simple"
        
        # 策略选项配置
        strategy_options = {
            "simple": {
                "label": "⚡ 简单",
                "desc": "基础向量搜索 - 最快速度，成本最低",
                "cost": "💰",
                "quality": "⭐⭐"
            },
            "hybrid": {
                "label": "🎯 融合",
                "desc": "向量+关键词混合搜索 - 平衡性能和成本",
                "cost": "💰💰",
                "quality": "⭐⭐⭐⭐"
            },
            "enhanced": {
                "label": "🚀 增强",
                "desc": "融合搜索+上下文增强 - 高质量答案",
                "cost": "💰💰💰",
                "quality": "⭐⭐⭐⭐⭐"
            },
            "adaptive": {
                "label": "🤖 自适应",
                "desc": "智能策略选择 - 自动优化",
                "cost": "💰💰💰",
                "quality": "⭐⭐⭐⭐"
            },
            "full": {
                "label": "⭐ 完整",
                "desc": "全流程优化 - 最高质量",
                "cost": "💰💰💰💰",
                "quality": "⭐⭐⭐⭐⭐"
            }
        }
        
        current_strategy = st.session_state.rag_strategy
        current_config = strategy_options[current_strategy]
        
        # 策略选择器
        selected = st.selectbox(
            f"当前策略: **成本 {current_config['cost']} | 质量 {current_config['quality']}**",
            options=list(strategy_options.keys()),
            format_func=lambda x: f"{strategy_options[x]['label']} - {strategy_options[x]['desc']}",
            index=list(strategy_options.keys()).index(current_strategy),
            key="rag_strategy_selector"
        )
        
        # 检测策略变化
        if selected != st.session_state.rag_strategy:
            st.session_state.rag_strategy = selected
            st.success(f"✅ 已切换到: {strategy_options[selected]['label']}")
            st.rerun()
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 重新加载知识库", use_container_width=True):
                with st.spinner("正在重新加载（启用文本清洗和智能分割）..."):
                    try:
                        kb_path = Path("data/knowledge_base")
                        if kb_path.exists():
                            num_docs = load_kb_from_files(
                                str(kb_path),
                                enable_cleaning=config.TEXT_CLEANING_ENABLED,
                                enable_smart_split=config.SMART_SPLIT_ENABLED,
                                custom_clean_patterns=config.CUSTOM_CLEAN_PATTERNS if config.CUSTOM_CLEAN_PATTERNS else None
                            )
                            
                            # 调用状态管理器更新系统状态
                            success, message = SystemStateManager.on_knowledge_base_updated(num_docs)
                            
                            if success:
                                st.success(f"✅ 重新加载成功！共 {num_docs} 篇文档")
                                st.info(message)
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.warning(message)
                        else:
                            st.warning("⚠️ 知识库目录不存在")
                    except Exception as e:
                        st.error(f"❌ 加载失败: {str(e)}")
        
        with col2:
            if st.button("🗑️ 清空知识库", use_container_width=True):
                if st.session_state.get("confirm_clear_kb", False):
                    try:
                        kb_path = Path("data/knowledge_base")
                        if kb_path.exists():
                            shutil.rmtree(kb_path)
                        
                        # 调用状态管理器清空知识库状态
                        success, message = SystemStateManager.on_knowledge_base_cleared()
                        
                        if success:
                            st.success("✅ 知识库已清空")
                            st.info(message)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning(message)
                    except Exception as e:
                        st.error(f"❌ 清空失败: {str(e)}")
                else:
                    st.warning("⚠️ 确认要清空知识库吗？")
                    if st.button("🔴 确认清空", use_container_width=True, key="confirm_clear"):
                        st.session_state.confirm_clear_kb = True
                        st.rerun()
        
        st.divider()
        
        # 配置信息
        st.markdown("### 📋 配置信息")
        
        # 从环境变量或配置获取嵌入模型信息
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
        
        # 获取具体的模型名称
        if embedding_provider == "huggingface":
            embedding_model = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "moka-ai/m3e-base")
            device = os.getenv("HUGGINGFACE_DEVICE", "cpu")
            model_display = f"{embedding_model} ({device})"
        elif embedding_provider == "openai":
            embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            model_display = embedding_model
        elif embedding_provider == "ollama":
            embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
            model_display = embedding_model
        elif embedding_provider == "deepseek":
            embedding_model = os.getenv("DEEPSEEK_EMBEDDING_MODEL", "text-embedding-3-small")
            model_display = embedding_model
        elif embedding_provider == "zhipuai":
            embedding_model = os.getenv("ZHIPUAI_EMBEDDING_MODEL", "embedding-2")
            model_display = embedding_model
        else:
            model_display = embedding_provider
        
        # 动态读取向量数据库类型
        vector_db_type = config.VECTOR_DB_TYPE.upper()  # milvus -> MILVUS
        vector_db_display = {
            "MILVUS": "Milvus (混合检索)",
            "CHROMA": "ChromaDB"
        }.get(vector_db_type, vector_db_type)
        
        # 文本处理配置
        cleaning_status = "✅ 启用" if config.TEXT_CLEANING_ENABLED else "❌ 禁用"
        smart_split_status = "✅ 启用" if config.SMART_SPLIT_ENABLED else "❌ 禁用"
        
        config_info = f"""
        **知识库路径**: `data/knowledge_base/`
        
        **Embedding 配置**:
        - 提供者: `{embedding_provider}`
        - 模型: `{model_display}`
        
        **向量库**: `{vector_db_display}`
        
        **文本处理**:
        - 文本清洗: `{cleaning_status}` (去除页码、URL、噪音)
        - 智能分割: `{smart_split_status}` (按章节/条款自动分割)
        - 分块大小: `300 tokens`
        """
        st.info(config_info)
