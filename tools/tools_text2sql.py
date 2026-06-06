"""
自然语言转 SQL 工具
"""
from langchain.tools import tool
from tools.llm_manager import llm_manager
import json
import logging

logger = logging.getLogger(__name__)

@tool
def text_to_sql_tool(
    question: str,
    db_schema: str
) -> dict:
    """
    将自然语言转换为 MySQL SQL 查询（LangChain Tool）
    
    功能说明:
        使用 LLM 将用户的自然语言问题翻译为规范的 MySQL SQL 语句。
        包含 MySQL 特定的日期函数优化和错误预防提示。
    
    参数:
        question (str): 用户的自然语言问题
            示例: "查询最近30天各产品类型的销售额"
        db_schema (str): 数据库表结构信息
            包含: 表名、字段名、数据类型、注释
            示例: "products (id INT, name VARCHAR, type VARCHAR, price DECIMAL)"
    
    返回:
        dict: SQL 生成结果
            成功时:
                - success (bool): True
                - sql (str): 生成的 SQL 语句
                - confidence (float): 置信度 (0.0-1.0)
                - explanation (str): SQL 查询说明
            失败时:
                - success (bool): False
                - sql (str): ""
                - confidence (float): 0
                - explanation (str): 失败说明
                - error (str): 详细错误信息
    
    工作流程:
        1. 构建提示词（包含 MySQL 日期函数最佳实践）
        2. 调用 LLM 生成 SQL（通过 llm_manager）
        3. 解析 LLM 返回的 JSON 格式响应
        4. 验证 SQL 有效性
        5. 返回结构化结果
    
    提示词优化:
        包含以下 MySQL 特定指导:
        - ✅ 正确的日期函数: DATE_SUB(CURDATE(), INTERVAL N DAY)
        - ❌ 常见错误: CURDATE() - ', '-30 days' (错误语法)
        - 日期分组: GROUP BY DATE(date_column)
        - 日期格式化: DATE_FORMAT(date, '%Y-%m-%d')
    
    LLM 响应处理:
        支持多种 LLM 返回格式:
        - AIMessage.content (OpenAI, Qwen)
        - 字符串响应 (Ollama)
        - 字典格式 (自定义 LLM)
        - 自动提取嵌入的 JSON（正则匹配）
    
    JSON 解析策略:
        1. 直接解析: json.loads(response_text)
        2. 正则提取: 查找 {...} 模式
        3. 容错处理: 捕获 JSONDecodeError
    
    日期查询示例:
        问题: "最近30天的销售趋势"
        生成SQL:
        ```sql
        SELECT DATE(date) as day, SUM(amount) as total
        FROM sales
        WHERE date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
        GROUP BY DATE(date)
        ORDER BY day;
        ```
    
    错误处理:
        - JSON 解析失败 → 记录原始响应，返回 error 字段
        - LLM 调用失败 → 捕获异常，打印堆栈
        - SQL 为空 → 标记 success=False
    
    日志输出:
        - INFO: SQL 生成开始、LLM 类型、解析成功
        - WARNING: SQL 为空、JSON 直接解析失败
        - ERROR: 异常堆栈、JSON 解析错误
    
    使用示例:
        >>> result = text_to_sql_tool.invoke({
        ...     "question": "各产品类型的平均价格",
        ...     "db_schema": "products (id, name, type, price)"
        ... })
        🔄 开始生成 SQL... 问题: 各产品类型的平均价格
        ✅ LLM 获取成功: ChatOpenAI
        ✅ JSON 解析成功
        ✅ SQL 生成成功: SELECT type, AVG(price)...
        
        >>> print(result)
        {
            'success': True,
            'sql': 'SELECT type, AVG(price) as avg_price FROM products GROUP BY type;',
            'confidence': 0.95,
            'explanation': '按产品类型分组计算平均价格'
        }
    
    注意事项:
        - 不进行 SQL 注入验证（假设 LLM 输出安全）
        - 生产环境建议添加 SQL 语法解析验证
        - confidence 值由 LLM 自行判断，非实际检测
        - 依赖 llm_manager.get_llm() 已正确配置
    
    相关工具:
        - execute_sql_tool: 执行生成的 SQL
        - generate_chart_tool: 将结果可视化
    """
    
    prompt = f"""
    你是一个 MySQL SQL 专家。请将用户的自然语言问题转换为正确的 MySQL SQL 查询。
    
    ⚠️ **重要 MySQL 日期函数用法**:
    - 获取最近 N 天的数据: WHERE date >= DATE_SUB(CURDATE(), INTERVAL N DAY)
    - 获取最近 N 天的数据(包括时间): WHERE created_at >= DATE_SUB(NOW(), INTERVAL N DAY)
    - 按日期分组: GROUP BY DATE(date_column) 或 GROUP BY DATE(created_at)
    - 日期格式化: DATE_FORMAT(date_column, '%Y-%m-%d')
    - 比较日期: WHERE date > '2024-01-01' AND date < '2024-12-31'
    
    ❌ **常见错误** (不要这样做):
    - WHERE date >= CURDATE() - ', '-30 days'  ❌ 错误语法
    - 使用了 - ', ' 的错误日期减法  ❌ 错误
    - DATE_SUB 函数的参数不正确  ❌ 错误
    
    ✅ **正确的日期查询示例**:
    1. SELECT * FROM sales WHERE date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);
    2. SELECT DATE(date) as day, SUM(amount) FROM sales WHERE date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY) GROUP BY DATE(date) ORDER BY day;
    3. SELECT * FROM orders WHERE created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY);
    4. SELECT COUNT(*) FROM users WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY);
    
    数据库 Schema:
    {db_schema}
    
    用户问题: {question}
    
    请返回如下 JSON 格式的响应（仅返回 JSON，不要其他文本）：
    {{
        "sql": "SELECT ...",
        "confidence": 0.0-1.0,
        "explanation": "查询说明"
    }}
    """
    
    try:
        logger.info(f"🔄 开始生成 SQL... 问题: {question}")
        logger.info(f"数据库 schema: {db_schema[:100]}")
        
        llm = llm_manager.get_llm()
        logger.info(f"✅ LLM 获取成功: {type(llm).__name__}")
        
        response = llm.invoke(prompt)
        logger.info(f"✅ LLM 调用成功，响应类型: {type(response).__name__}")
        
        # 处理不同类型的响应
        if hasattr(response, 'content'):
            response_text = response.content
            logger.info(f"从 response.content 获取: {response_text[:200]}")
        elif isinstance(response, str):
            response_text = response
            logger.info(f"直接字符串响应: {response_text[:200]}")
        elif hasattr(response, 'get'):
            response_text = response.get('content', str(response))
            logger.info(f"从 dict.get 获取: {response_text[:200]}")
        else:
            response_text = str(response)
            logger.info(f"字符串转换: {response_text[:200]}")
        
        logger.info(f"📝 处理后的响应: {response_text[:300]}")
        
        # 尝试解析 JSON
        try:
            result = json.loads(response_text)
            logger.info(f"✅ JSON 解析成功: {list(result.keys())}")
        except json.JSONDecodeError:
            # 尝试从响应中提取 JSON
            logger.warning("❌ 直接 JSON 解析失败，尝试提取 JSON...")
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info(f"✅ 提取的 JSON 解析成功")
            else:
                raise json.JSONDecodeError("无法找到 JSON", response_text, 0)
        
        if "sql" in result and result["sql"]:
            result["success"] = True
            logger.info(f"✅ SQL 生成成功: {result['sql'][:100]}")
            return result
        else:
            logger.warning(f"⚠️ 结果中没有有效的 SQL: {result}")
            return {
                "success": False,
                "sql": "",
                "confidence": result.get("confidence", 0),
                "explanation": result.get("explanation", "未生成有效的 SQL 语句"),
                "error": "LLM 返回的 SQL 为空"
            }
    
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON 解析失败: {e}")
        logger.error(f"原始响应: {response_text[:500]}")
        
        return {
            "success": False,
            "sql": "",
            "confidence": 0,
            "explanation": "SQL 生成失败",
            "error": f"JSON 解析错误: {str(e)}"
        }
    
    except Exception as e:
        logger.error(f"❌ SQL 生成异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "success": False,
            "sql": "",
            "confidence": 0,
            "explanation": "SQL 生成异常",
            "error": f"异常: {str(e)}"
        }
