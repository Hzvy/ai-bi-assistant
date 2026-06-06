"""
图表生成工具 - 基于 Highcharts JSON
"""
from langchain.tools import tool
from tools.llm_manager import llm_manager
import json

HIGHCHARTS_EXAMPLES = {
    "column": {
        "chart": {"type": "column"},
        "title": {"text": "示例标题"},
        "xAxis": {"categories": ["Jan", "Feb", "Mar"]},
        "yAxis": {"title": {"text": "数值"}},
        "series": [{"name": "数据", "data": [10, 20, 30]}]
    },
    "line": {
        "chart": {"type": "line"},
        "title": {"text": "示例标题"},
        "xAxis": {"categories": ["Jan", "Feb", "Mar"]},
        "yAxis": {"title": {"text": "数值"}},
        "series": [{"name": "数据", "data": [10, 20, 30]}]
    },
    "pie": {
        "chart": {"type": "pie"},
        "title": {"text": "示例标题"},
        "series": [{"name": "占比", "data": [{"name": "A", "y": 30}, {"name": "B", "y": 70}]}]
    }
}

@tool
def generate_chart_tool(
    data: str,
    chart_type: str = "column",
    title: str = "图表",
    x_label: str = "X 轴",
    y_label: str = "Y 轴"
) -> dict:
    """
    生成 Highcharts JSON 配置（LangChain Tool）
    
    功能说明:
        使用 LLM 将 SQL 查询结果转换为 Highcharts 可视化 JSON 配置。
        支持柱状图、折线图、饼图等多种图表类型。
    
    参数:
        data (str|list|dict): 查询结果数据
            - str: JSON 字符串 '[[\"A\", 10], [\"B\", 20]]'
            - list: Python 列表 [['A', 10], ['B', 20]]
            - dict: 单条数据字典 {'name': 'A', 'value': 10}
        chart_type (str): 图表类型，默认 "column"
            支持值: "column" (柱状图), "line" (折线图), "pie" (饼图)
        title (str): 图表标题，默认 "图表"
            示例: "各产品类型销售额统计"
        x_label (str): X 轴标签，默认 "X 轴"
            示例: "产品类型"
        y_label (str): Y 轴标签，默认 "Y 轴"
            示例: "销售额 (元)"
    
    返回:
        str: Markdown 代码块包裹的 JSON 配置
            格式: ```json\n{...}\n```
            失败时返回 dict: {"success": False, "error": "..."}
    
    工作流程:
        1. 数据格式标准化（字符串 → Python 对象）
        2. 图表类型标准化（小写、去空格）
        3. 根据 chart_type 选择示例模板
        4. 构建 LLM 提示词（数据 + 标签 + 模板）
        5. 调用 LLM 生成 Highcharts JSON
        6. 解析 JSON 并包裹在 ```json``` 标记中
        7. 返回字符串（UI 使用正则提取）
    
    Highcharts 模板:
        - column: {"chart": {"type": "column"}, "xAxis": {...}, "series": [...]}
        - line: {"chart": {"type": "line"}, ...}
        - pie: {"chart": {"type": "pie"}, "series": [{"data": [...]}]}
    
    数据格式转换:
        输入: execute_sql_tool 的 data 字段
        >>> [['Electronics', 299.5], ['Food', 12.8]]
        
        输出 (柱状图):
        {
          "chart": {"type": "column"},
          "xAxis": {"categories": ["Electronics", "Food"]},
          "series": [{"name": "价格", "data": [299.5, 12.8]}]
        }
        
        输出 (饼图):
        {
          "chart": {"type": "pie"},
          "series": [{
            "name": "占比",
            "data": [
              {"name": "Electronics", "y": 299.5},
              {"name": "Food", "y": 12.8}
            ]
          }]
        }
    
    关键设计: Markdown 代码块包装
        ✅ 正确返回:
        >>> return f"```json\n{json_str}\n```"
        
        UI 解析:
        >>> import re
        >>> pattern = r'```json([\\s\\S]*?)```'
        >>> match = re.search(pattern, response)
        >>> config = json.loads(match.group(1))
        
        ❌ 错误返回:
        >>> return config_dict  # UI 无法识别！
        >>> return json_str     # UI 无法识别！
    
    LLM 提示词结构:
        - 角色设定: "你是一位前端可视化专家"
        - 输入数据: SQL 结果列表
        - 图表参数: 标题、轴标签
        - 参考模板: Highcharts 标准格式
        - 输出要求: "仅返回 JSON，不要其他文本"
    
    错误处理:
        - 数据格式错误 → 转换为空列表
        - JSON 解析失败 → 返回 error 字典（带原始响应）
        - LLM 调用失败 → 捕获异常，返回堆栈信息
    
    使用示例:
        >>> data = [['电子产品', 150], ['食品', 230], ['服装', 89]]
        >>> result = generate_chart_tool.invoke({
        ...     "data": json.dumps(data),
        ...     "chart_type": "column",
        ...     "title": "各类别产品数量",
        ...     "x_label": "产品类别",
        ...     "y_label": "数量"
        ... })
        
        >>> print(result)
        ```json
        {
          "chart": {"type": "column"},
          "title": {"text": "各类别产品数量"},
          "xAxis": {"categories": ["电子产品", "食品", "服装"], "title": {"text": "产品类别"}},
          "yAxis": {"title": {"text": "数量"}},
          "series": [{"name": "数量", "data": [150, 230, 89]}]
        }
        ```
    
    UI 渲染流程:
        1. Agent 返回带 ```json``` 的字符串
        2. ui/chatbot_ui.py::extract_json_code_blocks() 提取 JSON
        3. tools/chart_renderer.py::render_chart() 渲染 Highcharts
        4. Streamlit 组件显示交互式图表
    
    注意事项:
        - ⚠️ 必须返回字符串而非字典（UI 依赖正则匹配）
        - 图表类型拼写错误会回退到 "column"
        - 数据为空时 LLM 可能生成示例数据
        - ensure_ascii=False 保留中文字符
    
    相关工具:
        - execute_sql_tool: 提供 data 输入
        - text_to_sql_tool: 生成 SQL 查询
    
    依赖:
        - llm_manager.get_llm(): LLM 实例
        - HIGHCHARTS_EXAMPLES: 图表模板库
    """
    
    try:
        # 处理数据格式：支持 JSON 字符串或已解析的列表
        if isinstance(data, str):
            data_list = json.loads(data)
        elif isinstance(data, list):
            data_list = data
        elif isinstance(data, dict):
            # 如果是字典，尝试转换为列表
            data_list = [data] if data else []
        else:
            data_list = []
        
        # 标准化 chart_type
        chart_type = chart_type.lower().strip() if chart_type else "column"
        
        example = HIGHCHARTS_EXAMPLES.get(chart_type, HIGHCHARTS_EXAMPLES["column"])
        
        prompt = f"""
        你是一位前端可视化专家，请根据数据生成 Highcharts {chart_type} 图的 JSON 配置。
        
        数据: {data_list}
        标题: {title}
        X 轴标签: {x_label}
        Y 轴标签: {y_label}
        
        参考示例:
        {json.dumps(example, ensure_ascii=False, indent=2)}
        
        请返回完整的 Highcharts JSON 配置（仅返回 JSON，不要其他文本）。
        """
        
        llm = llm_manager.get_llm()
        response = llm.invoke(prompt)
        
        try:
            config_dict = json.loads(response.content)
            # 返回包裹在 ```json``` 中的字符串，以便 UI 能够识别和渲染
            json_str = json.dumps(config_dict, ensure_ascii=False, indent=2)
            return f"```json\n{json_str}\n```"
        except json.JSONDecodeError as json_error:
            return {
                "success": False,
                "error": f"JSON 解析失败: {str(json_error)}",
                "raw_response": response.content[:200]
            }
    
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"图表生成失败: {str(e)}",
            "traceback": traceback.format_exc()[:200]
        }
