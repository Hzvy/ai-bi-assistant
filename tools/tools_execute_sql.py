"""
SQL 执行工具
"""
from langchain.tools import tool
import pymysql
from config import config
import json
from datetime import datetime, date
from decimal import Decimal
import streamlit as st

@tool
def execute_sql_tool(sql_query: str) -> dict:
    """
    执行 MySQL SQL 查询（LangChain Tool）
    
    功能说明:
        连接 MySQL 数据库执行 LLM 生成的 SQL 查询，并返回结构化结果。
        包含完整的连接管理、类型转换和错误处理。
    
    参数:
        sql_query (str): 待执行的 SQL 查询语句
            示例: "SELECT type, COUNT(*) FROM products GROUP BY type;"
    
    返回:
        dict: 查询执行结果
            成功时:
                - success (bool): True
                - data (List[List]): 查询结果，每行为一个列表
                    示例: [['Electronics', 150], ['Food', 230]]
                - rowcount (int): 返回的行数
                - columns (List[str]): 列名列表
                    示例: ['type', 'COUNT(*)']
            失败时:
                - success (bool): False
                - data (List): []
                - rowcount (int): 0
                - error (str): 错误信息（数据库连接失败/SQL语法错误等）
    
    工作流程:
        1. 检查数据库是否已配置 (config.has_db_config())
        2. 获取数据库连接配置 (host, port, user, password, database)
        3. 创建 PyMySQL 连接
        4. 执行 SQL 查询 (cursor.execute())
        5. 获取所有结果 (cursor.fetchall())
        6. 提取列名 (cursor.description)
        7. 类型转换（日期、Decimal → JSON 可序列化）
        8. 关闭连接
        9. 返回结构化结果
    
    数据库配置检查:
        配置来源:
        - Streamlit Session State (st.session_state.db_config)
        - 环境变量 (.env 文件)
        - config.py 默认值
        
        优先级: Session State > .env > config.py
        
        未配置时:
        → 返回友好提示: "请在左侧栏中的'🗄️ 数据库配置'中添加数据库连接信息"
    
    类型转换:
        MySQL 类型 → Python 类型:
        - DATE/DATETIME → datetime.date/datetime.datetime
        - DECIMAL → decimal.Decimal
        - INT → int
        - VARCHAR → str
        
        JSON 序列化转换:
        - datetime.date → ISO 格式字符串 "2024-01-15"
        - datetime.datetime → ISO 格式字符串 "2024-01-15T14:30:00"
        - Decimal → float (避免 JSON 序列化错误)
    
    连接管理:
        - 使用 pymysql 库（纯 Python MySQL 客户端）
        - 自动关闭连接（finally 块确保）
        - 不使用连接池（每次调用建立新连接）
    
    安全性:
        - ⚠️ 直接执行 LLM 生成的 SQL（无参数化查询）
        - 生产环境建议添加:
          1. SQL 语法白名单验证
          2. 危险关键字检测（DROP, DELETE, UPDATE）
          3. 查询超时限制
    
    错误类型:
        - 数据库未配置: "数据库未配置。请在左侧栏..."
        - 连接失败: "Can't connect to MySQL server..."
        - SQL 语法错误: "You have an error in your SQL syntax..."
        - 权限不足: "Access denied for user..."
    
    使用示例:
        >>> result = execute_sql_tool.invoke({
        ...     "sql_query": "SELECT type, AVG(price) FROM products GROUP BY type;"
        ... })
        
        >>> print(result)
        {
            'success': True,
            'data': [
                ['Electronics', 299.5],
                ['Food', 12.8],
                ['Clothing', 59.9]
            ],
            'rowcount': 3,
            'columns': ['type', 'AVG(price)']
        }
        
        >>> # 数据库未配置时
        >>> result = execute_sql_tool.invoke({"sql_query": "SELECT 1"})
        {
            'success': False,
            'data': [],
            'rowcount': 0,
            'error': '❌ 数据库未配置。请在左侧栏中的\'🗄️ 数据库配置\'中添加数据库连接信息。'
        }
    
    注意事项:
        - 每次调用创建新连接（无连接池），适合低频查询
        - 高并发场景建议改用连接池（如 DBUtils）
        - 结果全部加载到内存（大数据集可能OOM）
        - 仅支持 SELECT 查询（Agent 应避免 INSERT/UPDATE/DELETE）
    
    相关工具:
        - text_to_sql_tool: 生成此工具的 SQL 输入
        - generate_chart_tool: 将此工具的输出可视化
    
    依赖:
        - pymysql: Python MySQL 客户端库
        - config.get_db_config(): 数据库配置管理
    """
    
    try:
        # 检查数据库是否配置
        if not config.has_db_config():
            return {
                "success": False,
                "data": [],
                "rowcount": 0,
                "error": "❌ 数据库未配置。请在左侧栏中的'🗄️ 数据库配置'中添加数据库连接信息。"
            }
        
        db_config = config.get_db_config()
        connection = pymysql.connect(**db_config)
        
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            results = cursor.fetchall()
            
            # 类型转换
            def default_serializer(obj):
                if isinstance(obj, (date, datetime)):
                    return obj.isoformat()
                elif isinstance(obj, Decimal):
                    return float(obj)
                raise TypeError(f"Object {obj} is not JSON serializable")
            
            return {
                "success": True,
                "data": [list(row) for row in results],
                "rowcount": len(results),
                "columns": [desc[0] for desc in cursor.description] if cursor.description else []
            }
    
    except Exception as e:
        return {
            "success": False,
            "data": [],
            "rowcount": 0,
            "error": str(e)
        }
    
    finally:
        connection.close()
