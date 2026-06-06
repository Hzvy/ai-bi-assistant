"""
优化版侧边栏 - 增强的视觉设计和交互体验
"""
import streamlit as st
import pymysql
from agent import get_agent_executor
import time
from ui.login import render_user_info_sidebar


def test_db_connection(host, port, user, password, database):
    """
    测试数据库连接
    """
    try:
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database if database else None,
            charset='utf8mb4',
            connect_timeout=5
        )
        connection.close()
        return True, "连接成功！"
    except pymysql.Error as e:
        return False, f"连接失败: {str(e)}"
    except Exception as e:
        return False, f"错误: {str(e)}"


def render_sidebar():
    """渲染优化后的侧边栏"""
    
    # ===== 用户信息面板（置顶） =====
    render_user_info_sidebar()
    
    # ===== 系统状态面板 - 显示工作模式和资源状态 =====
    with st.sidebar:
        db_active = st.session_state.get("db_config_active", False)
        kb_active = st.session_state.get("kb_initialized", False)
        agent_mode = st.session_state.get("agent_mode", "纯对话模式")
        
        # 数据库状态
        db_status = "✅ 已连接" if db_active else "⚠️ 未连接"
        db_color = "green" if db_active else "orange"
        
        # 知识库状态和文档数量
        kb_doc_count = st.session_state.get("kb_doc_count", 0)
        kb_status = f"✅ {kb_doc_count}篇" if kb_active else "⚠️ 未加载"
        kb_color = "green" if kb_active else "orange"
           
        
        # 显示资源状态
        st.markdown("**📊 系统状态**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div style='background-color: rgba(0,255,0,0.1) if db_active else rgba(255,165,0,0.1); 
                           border-radius: 8px; padding: 10px; text-align: center;'>
                    <div style='font-size: 18px;'>🗄️</div>
                    <div style='font-size: 10px; color: #666; margin-top: 4px;'>数据库</div>
                    <div style='font-size: 11px; font-weight: bold;'>{db_status}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div style='background-color: rgba(100,150,255,0.1); 
                           border-radius: 8px; padding: 10px; text-align: center;'>
                    <div style='font-size: 18px;'>📚</div>
                    <div style='font-size: 10px; color: #666; margin-top: 4px;'>知识库</div>
                    <div style='font-size: 11px; font-weight: bold;'>{kb_status}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""
                <div style='background-color: rgba(100,200,100,0.1); 
                           border-radius: 8px; padding: 10px; text-align: center;'>
                    <div style='font-size: 18px;'>🤖</div>
                    <div style='font-size: 10px; color: #666; margin-top: 4px;'>Agent</div>
                    <div style='font-size: 11px; font-weight: bold;'>✅ 就绪</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.divider()
    
    # ===== Tab 式导航 =====
    tab1, tab2, tab3, tab4 = st.sidebar.tabs(["🗄️ 数据库", "📚 知识库", "⚙️ 设置", "ℹ️ 帮助"])
    
    # ===== Tab 1: 数据库配置 =====
    with tab1:
        st.subheader("数据库连接设置")
        
        # 快速连接预设
        st.markdown("**快速连接预设:**")
        preset_col1, preset_col2 = st.columns(2)
        
        with preset_col1:
            if st.button("🌐 本地 (localhost)", use_container_width=True):
                st.session_state.db_host = "localhost"
                st.session_state.db_port = 3306
                st.toast("✅ 已加载本地连接预设")
        
        with preset_col2:
            if st.button("☁️ 云端", use_container_width=True):
                st.session_state.db_host = ""
                st.toast("✅ 请输入云端地址")
        
        st.markdown("---")
        
        # 连接信息输入
        st.markdown("**连接信息:**")
        
        col1, col2 = st.columns(2)
        with col1:
            db_host = st.text_input(
                "主机地址",
                value=st.session_state.get("db_host", "localhost"),
                placeholder="例如: localhost",
                key="sidebar_host"
            )
            db_port = st.number_input(
                "端口",
                value=st.session_state.get("db_port", 3306),
                min_value=1,
                max_value=65535,
                step=1,
                key="sidebar_port"
            )
        
        with col2:
            db_user = st.text_input(
                "用户名",
                value=st.session_state.get("db_user", ""),
                placeholder="例如: root",
                key="sidebar_user"
            )
            db_password = st.text_input(
                "密码",
                type="password",
                value=st.session_state.get("db_password", ""),
                key="sidebar_password"
            )
        
        db_name = st.text_input(
            "数据库名 (可选)",
            value=st.session_state.get("db_name", ""),
            placeholder="例如: test_db",
            key="sidebar_dbname"
        )
        
        # 字符集选择
        db_charset = st.selectbox(
            "字符集",
            ["utf8mb4", "utf8", "latin1"],
            index=0,
            key="sidebar_charset"
        )
        
        st.markdown("---")
        
        # 操作按钮
        st.markdown("**操作:**")
        
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            if st.button("🔍 测试连接", use_container_width=True, key="test_db"):
                with st.spinner("正在连接..."):
                    success, message = test_db_connection(
                        host=db_host,
                        port=int(db_port),
                        user=db_user,
                        password=db_password,
                        database=db_name if db_name else None
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
        
        with btn_col2:
            if st.button("💾 保存配置", use_container_width=True, key="save_db"):
                if not db_host or not db_user:
                    st.error("❌ 主机地址和用户名不能为空")
                else:
                    with st.spinner("正在保存配置..."):
                        # 保存字符集到 session
                        st.session_state.db_charset = db_charset
                        
                        # 使用状态管理器更新系统状态
                        from tools.system_state_manager import SystemStateManager
                        
                        success, message = SystemStateManager.on_database_connected(
                            host=db_host,
                            port=int(db_port),
                            user=db_user,
                            password=db_password,
                            database=db_name,
                            charset=db_charset
                        )
                        
                        if success:
                            st.success(f"✅ 配置已保存 (字符集: {db_charset})")
                            st.info(message)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
        
        if st.button("🔄 重置配置", use_container_width=True, key="reset_db"):
            with st.spinner("正在重置配置..."):
                from tools.system_state_manager import SystemStateManager
                
                st.session_state.db_host = ""
                st.session_state.db_port = 3306
                st.session_state.db_user = ""
                st.session_state.db_password = ""
                st.session_state.db_name = ""
                st.session_state.db_charset = "utf8mb4"
                st.session_state.db_config_active = False
                
                # 更新系统状态
                success, message = SystemStateManager.update_system_state(trigger="database")                # 更新系统状态
                
                if success:
                    st.info(f"✅ 配置已重置\n{message}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(message)
        
        # 显示当前配置
        if st.session_state.get("db_config_active", False):
            st.markdown("---")
            st.markdown("**📋 当前配置:**")
            st.code(
                f"""
主机: {st.session_state.get('db_host')}
端口: {st.session_state.get('db_port')}
用户: {st.session_state.get('db_user')}
数据库: {st.session_state.get('db_name', '(未指定)')}
字符集: {st.session_state.get('db_charset', 'utf8mb4')}
                """,
                language="ini"
            )
    
    # ===== Tab 2: 知识库管理 =====
    with tab2:
        # 导入知识库管理器
        from ui.kb_manager import render_kb_manager
        render_kb_manager()
    
    # ===== Tab 3: 应用设置 =====
    with tab3:
        st.subheader("应用设置")
        
        # 导入配置
        from config import config
        import os
        
        # 显示当前配置信息
        st.markdown("### 📊 当前模型配置")
        
        # 创建配置卡片
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🤖 LLM 模型")
            llm_provider = os.getenv("LLM_PROVIDER", "openai").upper()
            llm_model = os.getenv(
                f"{llm_provider}_MODEL",
                os.getenv("LLM_MODEL", "未配置")
            )
            st.info(f"""
            **提供商**: {llm_provider}
            **模型**: {llm_model}
            """)
        
        with col2:
            st.markdown("#### 🔍 Embedding 模型")
            embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").upper()
            if embedding_provider == "HUGGINGFACE":
                embedding_model = os.getenv(
                    "HUGGINGFACE_EMBEDDING_MODEL",
                    "moka-ai/m3e-base"
                )
                device = os.getenv("HUGGINGFACE_DEVICE", "cpu")
                st.info(f"""
                **提供商**: {embedding_provider}
                **模型**: {embedding_model}
                **设备**: {device.upper()}
                """)
            else:
                embedding_model = os.getenv(
                    f"{embedding_provider}_EMBEDDING_MODEL",
                    "未配置"
                )
                st.info(f"""
                **提供商**: {embedding_provider}
                **模型**: {embedding_model}
                """)
        
        st.divider()
        
        # 显示所有可用的 LLM 提供商
        st.markdown("### 📋 可用的 LLM 提供商")
        llm_providers = {
            "openai": "OpenAI (需要 OPENAI_API_KEY)",
            "deepseek": "Deepseek (需要 DEEPSEEK_API_KEY)",
            "qwen": "阿里通义千问 (需要 QWEN_API_KEY)",
            "anthropic": "Anthropic Claude (需要 ANTHROPIC_API_KEY)",
            "ollama": "Ollama 本地模型 (需要本地部署)",
            "zhipuai": "智谱 GLM (需要 ZHIPUAI_API_KEY)",
        }
        
        for provider, description in llm_providers.items():
            status = "✅ 已配置" if os.getenv(f"{provider.upper()}_API_KEY") or provider == "ollama" else "⚠️ 未配置"
            st.write(f"**{provider.upper()}**: {status} - {description}")
        
        st.divider()
        
        # 显示所有可用的 Embedding 提供商
        st.markdown("### 📋 可用的 Embedding 提供商")
        embedding_providers = {
            "huggingface": "✅ HuggingFace (推荐！本地离线)",
            "ollama": "Ollama (本地部署，离线运行)",
            "openai": "OpenAI (云端服务，需要 API Key)",
            "deepseek": "Deepseek (云端服务，便宜)",
            "zhipuai": "智谱 GLM (云端服务)",
            "baichuan": "百川 (云端服务)",
        }
        
        for provider, description in embedding_providers.items():
            st.write(f"**{provider.upper()}**: {description}")
        
        st.divider()
        
        st.markdown("### � 配置说明")
        st.info("""
        **如何修改配置**:
        1. 编辑项目根目录的 `.env` 文件
        2. 修改 `LLM_PROVIDER` 和 `EMBEDDING_PROVIDER` 变量
        3. 重启应用使配置生效
        
        **推荐配置** (免费，无需 API Key):
        ```
        LLM_PROVIDER=ollama
        EMBEDDING_PROVIDER=huggingface
        ```
        
        **详细配置说明**，请参考：`.env.example` 文件
        """)
    
    # ===== Tab 4: 帮助 & 文档 =====
    with tab4:
        st.subheader("使用帮助")
        
        st.markdown("""
        ### 快速开始
        
        1. **配置数据库** (可选)
           - 在"🗄️ 数据库"标签页中填写连接信息
           - 点击"🔍 测试连接"验证
           - 点击"💾 保存配置"激活
        
        2. **上传知识库** (可选)
           - 在"📚 知识库"标签页上传文档
           - 系统自动向量化并存储
        
        3. **开始提问**
           - 在主窗口输入问题
           - Agent 自动选择最佳模式
           - 等待响应即可
        
        ### 工作模式说明
        
        **完整模式** ⭐
        - 数据库已连接 ✅
        - 知识库已初始化 ✅
        - 可用: 数据查询 + 图表生成 + 文档检索
        
        **数据分析模式** 📊
        - 数据库已连接 ✅
        - 知识库未初始化 ⚠️
        - 可用: 数据查询 + 图表生成
        
        **对话模式** 📚
        - 数据库未连接 ⚠️
        - 知识库已初始化 ✅
        - 可用: 文档检索 + 基础对话
        
        **纯对话模式** 💬
        - 数据库未连接 ⚠️
        - 知识库未初始化 ⚠️
        - 可用: 基础对话
        
        ### 常见问题
        
        **Q: 数据库连接失败？**
        - 检查数据库服务是否启动
        - 验证连接信息是否正确
        - 确保防火墙允许连接
        
        **Q: 知识库加载缓慢？**
        - 第一次加载需要下载 Embedding 模型
        - 文件过大时可能需要等待较长时间
        - 建议在网络稳定时执行
        
        **Q: Agent 未初始化？**
        - 检查 API Key 配置是否正确
        - 验证网络连接是否正常
        - 查看应用启动日志获取详细信息
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📚 相关文档
        
        - [完整用户指南](README.md)
        - [快速修复指南](说明文档/Agent问题快速修复.md)
        - [架构文档](说明文档/AI_BI_Assistant_V1_架构与最佳实践.md)
        
        ### 📞 反馈
        
        遇到问题? 查看 Copilot 指导: `.github/copilot-instructions.md`
        """)
    
    st.sidebar.divider()
    
    # 底部信息
    st.sidebar.markdown(
        """
        <div style='text-align: center; font-size: 11px; color: #888; margin-top: 20px;'>
            <p>AI BI Assistant V4</p>
            <p>v4.0.0 | Built by Hzvy</p>
        </div>
        """,
        unsafe_allow_html=True
    )
