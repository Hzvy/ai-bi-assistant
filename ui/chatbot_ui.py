"""
聊天机器人 UI 组件
"""
import streamlit as st
import streamlit_highcharts as hct
import json
import re
import logging

logger = logging.getLogger(__name__)


def extract_json_code_blocks(text):
    """
    从文本中提取所有 Markdown JSON 代码块
    
    功能说明:
        使用正则表达式匹配 ```json...``` 格式的代码块。
        这是 Agent 返回 Highcharts 配置的标准格式。
    
    参数:
        text (str): 待解析的文本内容
            示例: "分析结果如下\n```json\n{...}\n```\n详情..."
    
    返回:
        dict or None:
            成功时:
                {
                    'blocks': [  # JSON 代码块列表
                        {
                            'before': str,  # 代码块前的文本
                            'json': str,    # JSON 字符串（已去首尾空格）
                            'start': int,   # 代码块起始位置
                            'end': int      # 代码块结束位置
                        },
                        ...
                    ],
                    'after': str  # 最后一个代码块后的文本
                }
            失败时: None（未找到代码块）
    
    正则模式:
        ```python
        pattern = r'```json([\\s\\S]*?)```'
        ```
        - ```json: 起始标记
        - ([\\s\\S]*?): 捕获组，匹配任意字符（包括换行）
        - ?: 非贪婪模式，匹配最短内容
        - ```: 结束标记
    
    匹配示例:
        输入:
        ```
        分析如下：
        ```json
        {"chart": {"type": "column"}}
        ```
        结论...
        ```
        
        输出:
        {
            'blocks': [
                {
                    'before': '分析如下：',
                    'json': '{"chart": {"type": "column"}}',
                    'start': 12,
                    'end': 58
                }
            ],
            'after': '结论...'
        }
    
    使用场景:
        1. **图表渲染流水线**
           Agent → 返回带 ```json``` 的响应 → 提取 JSON → 解析 → 渲染
        
        2. **文本与图表分离**
           提取前置文本（分析说明）和 JSON（图表配置）分别展示
        
        3. **多图表支持**
           单个响应包含多个 ```json``` 块，全部提取
    
    调试日志:
        - 匹配数量: logger.debug(f"找到 {len(matches)} 个匹配")
        - 未匹配时: 打印文本前200字符用于诊断
        - 文本长度和是否包含 '```json' 标记
    
    边界情况处理:
        1. **代码块紧跟其他文本**
           ```json...```Thought → 正确提取（[\s\S] 匹配所有字符）
        
        2. **空代码块**
           ```json\n\n``` → json 字段为空字符串
        
        3. **多行 JSON**
           ```json\n{\n  "a": 1\n}\n``` → 正确保留换行
        
        4. **无代码块**
           "纯文本" → 返回 None
    
    性能:
        - re.finditer() 迭代器，内存友好
        - re.DOTALL 标志，支持跨行匹配
        - 时间复杂度: O(n) - n 为文本长度
    
    使用示例:
        >>> text = '''
        ... 查询结果：
        ... ```json
        ... {"data": [1, 2, 3]}
        ... ```
        ... 说明：数据已更新
        ... '''
        
        >>> result = extract_json_code_blocks(text)
        >>> print(result)
        {
            'blocks': [
                {
                    'before': '查询结果：',
                    'json': '{"data": [1, 2, 3]}',
                    'start': 18,
                    'end': 58
                }
            ],
            'after': '说明：数据已更新'
        }
        
        >>> # 访问 JSON 内容
        >>> import json
        >>> config = json.loads(result['blocks'][0]['json'])
        >>> print(config['data'])
        [1, 2, 3]
    
    注意事项:
        - 仅提取 JSON 字符串，不自动解析
        - 调用者需要自行 json.loads() 并处理异常
        - 不验证 JSON 格式有效性
        - ```json 标记必须小写
    """
    # 正则表达式匹配 ```json ... ```
    # 关键：使用 [\s\S]*? 可以匹配任何字符（包括换行）
    # 这样可以处理 "```json...```Thought" 这样直接相连的情况
    pattern = r'```json([\s\S]*?)```'
    matches = list(re.finditer(pattern, text, re.DOTALL))
    
    logger.debug(f"📊 extract_json_code_blocks: 正则模式找到 {len(matches)} 个匹配")
    logger.debug(f"   文本长度: {len(text)}, 包含 '```json': {'```json' in text}")
    
    if not matches:
        logger.debug(f"⚠️ 未找到任何 JSON 代码块")
        logger.debug(f"   查看文本前200字符: {text[:200]}")
        return None
    
    results = []
    last_end = 0
    
    for match in matches:
        before_text = text[last_end:match.start()].strip()
        json_str = match.group(1).strip()  # 提取并清理 JSON 内容
        last_end = match.end()
        results.append({
            'before': before_text,
            'json': json_str,
            'start': match.start(),
            'end': match.end()
        })
    
    after_text = text[last_end:].strip()
    
    return {
        'blocks': results,
        'after': after_text
    }


def try_extract_chart(response_text: str) -> dict:
    """
    从 Agent 响应中提取 Highcharts 图表配置
    
    功能说明:
        智能解析 Agent 返回的文本，提取 Highcharts JSON 配置。
        支持两种格式：Markdown 代码块和纯 JSON。
    
    参数:
        response_text (str): Agent 的完整响应文本
            示例: "分析结果\n```json\n{...}\n```"
    
    返回:
        dict or None:
            - 成功: Highcharts 配置字典
            - 失败: None（无图表或解析失败）
    
    提取策略（按优先级）:
        
        **方法 1: Markdown 代码块**（推荐）
        - 匹配: ```json ... ```
        - 提取: JSON 字符串
        - 解析: json.loads()
        - 验证: isinstance(config, dict)
        - 优先级: 最高（标准格式）
        
        **方法 2: 直接 JSON 解析**（降级）
        - 尝试: json.loads(response_text)
        - 验证: 包含 'chart' 或 'series' 字段
        - 用途: 兼容纯 JSON 响应
    
    Highcharts 配置验证:
        有效配置需满足以下之一:
        - 包含 "chart" 字段（图表类型）
        - 包含 "series" 字段（数据系列）
        
        示例有效配置:
        ```json
        {
            "chart": {"type": "column"},
            "series": [{"name": "销售额", "data": [10, 20, 30]}]
        }
        ```
    
    工作流程:
        1. 验证输入非空且为字符串
        2. 尝试方法 1: 正则提取 ```json...```
        3. 找到匹配 → json.loads() → 验证 dict → 返回
        4. 未找到或解析失败 → 尝试方法 2
        5. 尝试直接解析整个文本为 JSON
        6. 解析成功 → 验证包含 chart/series → 返回
        7. 所有方法失败 → 返回 None
    
    错误处理:
        - response_text 为空 → 返回 None
        - 非字符串类型 → 返回 None
        - JSON 解析失败 → 捕获 JSONDecodeError，记录日志
        - 解析成功但非字典 → 忽略
        - 解析成功但无 chart/series → 忽略
    
    日志输出:
        - ✅ 成功: logger.info("从代码块成功提取图表配置")
        - ⚠️ 失败: logger.debug("代码块内 JSON 解析失败")
        - 日志包含错误信息的前100字符
    
    使用示例:
        >>> # 场景 1: Markdown 代码块格式
        >>> response = '''
        ... 分析结果如下：
        ... ```json
        ... {
        ...   "chart": {"type": "column"},
        ...   "series": [{"data": [10, 20, 30]}]
        ... }
        ... ```
        ... '''
        >>> config = try_extract_chart(response)
        ✅ 从代码块成功提取图表配置
        >>> print(config['chart']['type'])
        column
        
        >>> # 场景 2: 纯 JSON 格式
        >>> response = '{"chart": {"type": "pie"}, "series": [...]}'
        >>> config = try_extract_chart(response)
        ✅ 成功提取图表配置 (直接 JSON)
        
        >>> # 场景 3: 无图表内容
        >>> response = "这是纯文本分析结果"
        >>> config = try_extract_chart(response)
        >>> print(config)
        None
    
    返回值示例:
        ```python
        {
            "chart": {"type": "column"},
            "title": {"text": "销售统计"},
            "xAxis": {"categories": ["Q1", "Q2", "Q3"]},
            "yAxis": {"title": {"text": "金额"}},
            "series": [
                {"name": "销售额", "data": [100, 200, 150]}
            ]
        }
        ```
    
    调用链:
        render_chatbot() → extract_json_code_blocks() → try_extract_chart()
        → hct.streamlit_highcharts()
    
    注意事项:
        - 仅提取第一个 JSON 代码块（多个时返回首个）
        - 不验证 Highcharts 配置的完整性
        - 不检查必需字段（如 series 的 data）
        - 调用者需处理渲染时的错误
    
    性能:
        - 正则匹配: O(n) - n 为文本长度
        - JSON 解析: O(m) - m 为 JSON 长度
        - 适合实时 UI 渲染（< 10ms）
    """
    if not response_text or not isinstance(response_text, str):
        return None
    
    # 方法 1: 查找 ```json ... ``` 代码块
    pattern = r'```json([\s\S]*?)```'
    match = re.search(pattern, response_text, re.DOTALL)
    
    if match:
        json_str = match.group(1).strip()
        try:
            config = json.loads(json_str)
            if isinstance(config, dict):
                logger.info(f"✅ 从代码块成功提取图表配置")
                return config
        except json.JSONDecodeError as e:
            logger.debug(f"⚠️ 代码块内 JSON 解析失败: {str(e)[:100]}")
    
    # 方法 2: 尝试直接解析为 JSON
    try:
        config = json.loads(response_text)
        if isinstance(config, dict):
            has_chart = "chart" in config
            has_series = "series" in config
            
            if has_chart or has_series:
                logger.info(f"✅ 成功提取图表配置 (直接 JSON)")
                return config
    except (json.JSONDecodeError, TypeError):
        pass
    
    return None


def render_chatbot():
    
    st.title("💬 AI BI Assistant")
    st.markdown("---")
    
    # 初始化消息列表和欢迎标志
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "welcome_shown" not in st.session_state:
        st.session_state.welcome_shown = False
    
    # 首次加载时显示 AI 自我介绍
    if not st.session_state.welcome_shown and len(st.session_state.messages) == 0:
        welcome_message = """👋 **你好！我是 AI BI Assistant**

我是一个智能数据分析助手，专门帮助您：

🎯 **核心能力：**
- 📊 **数据查询** - 连接数据库查询业务数据
- 📈 **图表生成** - 自动生成各类数据可视化图表
- 📚 **知识检索** - 从知识库中快速获取信息
- 💬 **智能对话** - 解答您的数据和业务问题

🚀 **快速开始：**
1. 在左侧栏配置数据库连接（可选）
2. 初始化知识库（可选）
3. 开始提问！

💡 **示例问题：**
- "生成各种产品类型的柱状图"
- "生成销售额的趋势图表"

有什么我可以帮您的吗？🤔"""
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": welcome_message
        })
        st.session_state.welcome_shown = True
    
    # 显示聊天历史
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            content = message["content"]
            
            # 从原始内容中提取 JSON
            extracted = extract_json_code_blocks(content)
            
            # 提取 Final Answer 作为显示文本（如果有）
            display_text = content
            if message["role"] == "assistant" and "Final Answer:" in content:
                parts = content.split("Final Answer:", 1)
                display_text = parts[1].strip()
            
            logger.debug(f"📋 处理消息 (角色: {message['role']}, 内容长度: {len(content)})")
            
            if extracted and extracted['blocks']:
                logger.info(f"✅ 检测到 {len(extracted['blocks'])} 个图表代码块")
                
                # 提取并显示 JSON 前面的文本（分析部分）
                first_block = extracted['blocks'][0]
                text_before_json = first_block['before'].strip()
                
                logger.info(f"📝 JSON 前文字长度: {len(text_before_json)}")
                logger.info(f"📝 JSON 前文字内容: {text_before_json[:200] if len(text_before_json) > 200 else text_before_json}")
                
                if text_before_json:
                    logger.info(f"✅ 显示分析文本 ({len(text_before_json)} 字符)")
                    st.markdown(text_before_json)
                else:
                    logger.warning("⚠️ JSON 前面没有文字")
                
                # 渲染图表
                for idx, block in enumerate(extracted['blocks'], 1):
                    try:
                        logger.debug(f"📊 解析图表 {idx}, JSON 长度: {len(block['json'])}")
                        json_data = json.loads(block['json'])
                        if isinstance(json_data, dict):
                            logger.info(f"✅ 图表 {idx} 渲染中... 类型: {json_data.get('chart', {}).get('type', '未知')}")
                            hct.streamlit_highcharts(json_data, 400)
                            logger.info(f"✅ 图表 {idx} 渲染完成")
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ 图表 {idx} JSON 解析失败: {str(e)[:100]}")
                        st.error(f"❌ JSON 解析错误: {str(e)}")
                
                # 显示 JSON 后面的文本（如果有）
                text_after_json = extracted['after'].strip()
                if text_after_json:
                    logger.info(f"✅ 显示后续文本 ({len(text_after_json)} 字符)")
                    st.markdown(text_after_json)
                else:
                    logger.debug("ℹ️ JSON 后面没有文字")
            else:
                logger.debug(f"⚠️ 未找到图表代码块，直接显示文本")
                # 没有图表，直接显示文本
                st.markdown(display_text)
    
    # 检查数据库是否配置
    db_active = st.session_state.get("db_config_active", False)
    agent_mode = st.session_state.get("agent_mode", "纯对话模式")
    
    # 根据工作模式生成提示信息
    mode_info = {
        "完整模式": {
            "emoji": "⭐",
            "status": "✅ 数据库已连接 | ✅ 知识库已就绪",
            "placeholder": "输入您的问题... (例如: '帮我查询销售数据' 或 '销售额的定义是什么')",
            "tips": """### ✅ 完整功能已启用
您可以：
- 📊 查询数据库数据
- 📈 生成可视化图表
- 📚 检索知识库内容"""
        },
        "数据分析模式": {
            "emoji": "📊",
            "status": "✅ 数据库已连接 | ⚠️ 知识库未初始化",
            "placeholder": "输入您的问题... (例如: '帮我查询销售数据' 或 '生成销售额柱状图')",
            "tips": """### 📊 数据分析模式
您可以：
- 📊 查询数据库数据
- 📈 生成可视化图表

提示: 初始化知识库后可解锁更多功能"""
        },
        "对话模式": {
            "emoji": "📚",
            "status": "⚠️ 数据库未连接 | ✅ 知识库已就绪",
            "placeholder": "输入您的问题... (例如: '销售额的定义是什么' 或 '查询相关概念')",
            "tips": """### 📚 对话模式
您可以：
- � 检索知识库内容
- ❓ 了解数据定义和含义

提示: 配置数据库连接后可启用数据查询功能"""
        },
        "纯对话模式": {
            "emoji": "💬",
            "status": "⚠️ 数据库未连接 | ⚠️ 知识库未初始化",
            "placeholder": "输入您的问题... (AI 助手将直接对话)",
            "tips": """### 💬 纯对话模式
当前处于基础对话模式，更多功能已禁用：
- 📊 数据查询（需配置数据库）
- 📚 知识库检索（需初始化知识库）

请在左侧栏进行相应配置以解锁完整功能"""
        }
    }
    
    current_mode = mode_info.get(agent_mode, mode_info["纯对话模式"])
    
    # 显示工作模式和功能状态
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"**{current_mode['emoji']} 工作模式**: {agent_mode}\n"
            f"**状态**: {current_mode['status']}"
        )
    with col2:
        if st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            st.rerun()
    
    # 初始化RAG策略
    if "rag_strategy" not in st.session_state:
        st.session_state.rag_strategy = "simple"
    
    # 检查知识库是否已初始化并有内容
    kb_active = st.session_state.get("kb_initialized", False)
    
    st.markdown("---")
    
    # 聊天输入框 - 总是显示
    if prompt := st.chat_input(current_mode['placeholder']):
        
        # 添加用户消息到历史
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 获取 Agent 响应
        with st.chat_message("assistant"):
            with st.spinner("思考中... 🤔"):
                try:
                    # 获取 Agent
                    agent = st.session_state.get("agent")
                    
                    if agent is None:
                        assistant_message = """⚠️ Agent 未初始化

**可能的原因:**
1. 🔌 API 连接失败 (LLM 服务不可用)
2. 📦 依赖包缺失
3. 🔐 API Key 配置错误
4. 🌐 网络连接问题

**解决步骤:**
1. 检查 `.env` 文件配置是否正确
2. 验证 API Key 是否有效
3. 检查网络连接
4. 刷新页面重试
5. 查看应用启动日志获取错误详情

**临时解决方案:**
如果您只想进行基础对话，请：
- 在左侧栏配置 LLM 模型
- 刷新页面重新加载应用"""
                    elif not db_active and any(keyword in prompt for keyword in ["查询", "查", "统计", "多少"]):
                        # 没有数据库但用户要求数据库查询
                        assistant_message = """
                        ### ⚠️ 数据库未配置
                        
                        您的问题似乎需要数据库查询支持。请先在左侧栏的"🗄️ 数据库配置"中配置数据库连接。
                        
                        **目前可用功能:**
                        - 📄 知识库文件检索
                        - ❓ 数据定义查询
                        
                        **配置数据库后可用:**
                        - 📊 数据库数据查询
                        - 📈 自动生成图表
                        
                        **配置步骤:**
                        1. 点击左侧栏的"🗄️ 数据库配置"
                        2. 填入数据库连接信息
                        3. 点击"🔍 测试连接"验证
                        4. 点击"💾 保存配置"保存
                        """
                    else:
                        # 调用 Agent 获取响应 (LangGraph 格式)
                        try:
                            # 详细日志记录
                            logger.info(f"🔄 调用 LangGraph Agent... 输入: {prompt[:100]}")
                            
                            # LangGraph 调用格式
                            config = {"configurable": {"thread_id": "conversation_1"}}
                            result = agent.invoke(
                                {"messages": [("user", prompt)]},
                                config=config
                            )
                            
                            # 检查响应格式
                            if not isinstance(result, dict):
                                logger.warning(f"⚠️ Agent 响应不是字典: {type(result)}")
                                assistant_message = f"⚠️ 响应格式错误: {str(result)}"
                            elif "messages" not in result:
                                logger.warning(f"⚠️ Agent 响应缺少 'messages': {list(result.keys())}")
                                assistant_message = "⚠️ Agent 返回格式错误，请重试"
                            else:
                                logger.info(f"✅ Agent 响应包含 {len(result['messages'])} 条消息")
                                
                                # 获取最后一条消息
                                last_message = result["messages"][-1]
                                
                                # 提取内容
                                if hasattr(last_message, 'content'):
                                    output = last_message.content
                                elif isinstance(last_message, dict):
                                    output = last_message.get('content', '')
                                else:
                                    output = str(last_message)
                                
                                if output is None or len(str(output).strip()) == 0:
                                    logger.warning("⚠️ Agent 输出为空")
                                    assistant_message = "⚠️ Agent 未生成响应，请重试"
                                else:
                                    logger.info(f"✅ 获得有效响应 ({len(str(output))} 字符)")
                                    # LangGraph 直接使用输出内容，无需提取 "Final Answer:"
                                    assistant_message = str(output)
                        
                        except Exception as agent_error:
                            import traceback
                            error_detail = traceback.format_exc()
                            logger.error(f"❌ Agent 执行错误:\n{error_detail}")
                            
                            assistant_message = f"""❌ Agent 执行出错

**错误信息:** {str(agent_error)}

**可能的原因:**
- LLM 服务暂时不可用 (401/连接超时)
- 输入文本过长或格式不合法
- Agent 执行超时
- 工具调用失败

**建议:**
1. 尝试简化您的问题
2. 稍后重试
3. 检查左侧栏的日志和配置
4. 重新启动应用: `streamlit cache clear && streamlit run main.py`

**详细错误:**
```
{str(agent_error)[:500]}
```"""
                    
                except Exception as e:
                    import traceback
                    error_detail = traceback.format_exc()
                    print(f"❌ 聊天处理错误:\n{error_detail}")
                    assistant_message = f"""❌ 出错了: {str(e)}

请查看应用启动日志获取详细信息"""
                
                # 显示响应
                # 使用 ChatBI-main 的方式: 查找 ```json``` 代码块
                logger.info(f"📝 处理 Agent 响应 (长度: {len(assistant_message)})")
                logger.debug(f"   响应前200字符: {assistant_message[:200]}")
                
                # 保存原始响应用于提取 JSON
                original_response = assistant_message
                
                # 关键：先在原始 assistant_message 中提取 JSON（可能在 Final Answer 之前）
                extracted = extract_json_code_blocks(original_response)
                logger.info(f"📊 第一次提取: extracted={extracted is not None}, blocks={len(extracted.get('blocks', [])) if extracted else 0}")
                
                # 如果有 Final Answer，提取 Final Answer 后的内容作为显示文本
                display_message = assistant_message
                if "Final Answer:" in assistant_message:
                    parts = assistant_message.split("Final Answer:", 1)
                    display_message = parts[1].strip()
                    logger.info(f"✅ 提取 Final Answer 作为显示文本 (长度: {len(display_message)})")
                
                # 调试：在 UI 中显示提取信息
                # with st.expander("🔍 调试信息", expanded=False):
                #     st.code(f"原始响应长度: {len(original_response)}\n显示消息长度: {len(display_message)}\n已提取代码块: {len(extracted.get('blocks', [])) if extracted else 0}\n\n显示内容:\n{display_message[:500]}", language="text")
                
                if extracted and extracted['blocks']:
                    logger.info(f"✅ 检测到 {len(extracted['blocks'])} 个图表代码块")
                    
                    # 提取并显示第一个 JSON 前面的文字（分析部分）
                    first_block = extracted['blocks'][0]
                    text_before_json = first_block['before'].strip()
                    
                    if text_before_json:
                        logger.info(f"📝 显示 JSON 前的分析文字 ({len(text_before_json)} 字符)")
                        st.markdown(text_before_json)
                    else:
                        logger.warning("⚠️ JSON 前面没有分析文字")
                    
                    # 渲染所有图表
                    for idx, block in enumerate(extracted['blocks'], 1):
                        # 渲染图表
                        try:
                            logger.debug(f"📊 解析图表 {idx}, JSON 长度: {len(block['json'])}")
                            json_data = json.loads(block['json'])
                            if isinstance(json_data, dict):
                                chart_type = json_data.get('chart', {}).get('type', '未知')
                                logger.info(f"✅ 渲染图表 {idx} - 类型: {chart_type}")
                                hct.streamlit_highcharts(json_data, 500)
                                logger.info(f"✅ 图表 {idx} 渲染完成")
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ 图表 {idx} JSON 解析失败: {str(e)[:100]}")
                            st.error(f"❌ JSON 解析错误: {str(e)}")
                    
                    # 显示 JSON 后面的文字（如果有）
                    text_after_json = extracted['after'].strip()
                    if text_after_json:
                        logger.info(f"📝 显示 JSON 后的文字 ({len(text_after_json)} 字符)")
                        st.markdown(text_after_json)
                else:
                    # logger.debug(f"⚠️ Agent 响应中未找到图表代码块")
                    # st.warning(f"⚠️ 未找到图表代码块")
                    # 没有图表，直接显示文本
                    st.markdown(display_message)
                
                # 添加到历史 - 保存原始响应用于后续提取 JSON
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": original_response  # 保存原始响应，包含 JSON 代码块
                })   