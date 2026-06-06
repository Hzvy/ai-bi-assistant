# """
# 工具调用格式验证模块

# 用于验证 Agent 生成的工具调用是否符合格式规范。
# """

# import json
# import re
# from typing import Dict, List, Union, Any


# class ToolCallValidator:
#     """工具调用格式验证器"""
    
#     # 允许的工具列表
#     ALLOWED_TOOLS = {
#         "database_query": "执行 SQL 数据库查询",
#         "generate_visualization": "生成数据可视化图表",
#         "execute_sql": "执行 SQL 语句（与 database_query 别名）",
#         "create_chart": "创建图表（与 generate_visualization 别名）",
#     }
    
#     # 工具别名映射
#     TOOL_ALIASES = {
#         "execute_sql": "database_query",
#         "create_chart": "generate_visualization",
#         "sql_query": "database_query",
#         "chart": "generate_visualization",
#     }
    
#     @classmethod
#     def validate_json_format(cls, tool_call_str: str) -> Dict[str, Any]:
#         """
#         验证工具调用的 JSON 格式
        
#         Args:
#             tool_call_str: 工具调用的 JSON 字符串
            
#         Returns:
#             {
#                 "valid": bool,
#                 "error": str (如果验证失败),
#                 "errors": List[str] (所有错误),
#                 "tool_call": Dict (解析后的工具调用),
#                 "normalized_tool_call": Dict (规范化后的工具调用)
#             }
#         """
        
#         errors = []
#         tool_call = None
        
#         # 1. 检查是否有有效的 JSON
#         try:
#             tool_call = json.loads(tool_call_str)
#         except json.JSONDecodeError as e:
#             error_msg = f"JSON 格式错误: {e}"
#             return {
#                 "valid": False,
#                 "error": error_msg,
#                 "errors": [error_msg]
#             }
        
#         # 2. 检查 tool_call 是否为字典
#         if not isinstance(tool_call, dict):
#             error_msg = f"工具调用必须是 JSON 对象，得到: {type(tool_call).__name__}"
#             return {
#                 "valid": False,
#                 "error": error_msg,
#                 "errors": [error_msg]
#             }
        
#         # 3. 检查必需的键（严格验证大小写）
#         action_key = None
#         action_input_key = None
        
#         # 首先尝试查找正确的键名
#         if "Action" in tool_call:
#             action_key = "Action"
#         elif "action" in tool_call:
#             errors.append("键名 'action' 应该是大写 'Action'")
        
#         if "Action Input" in tool_call:
#             action_input_key = "Action Input"
#         elif "action_input" in tool_call or "action_input" in tool_call:
#             errors.append("键名应该是 'Action Input'（大写首字母，用空格分隔）")
        
#         if not action_key:
#             errors.append("缺少 'Action' 键（必须大写首字母）")
        
#         if not action_input_key:
#             errors.append("缺少 'Action Input' 键（必须大写首字母）")
        
#         # 4. 检查工具名称
#         if action_key and action_key in tool_call:
#             action = tool_call[action_key]
            
#             # 检查类型
#             if not isinstance(action, str):
#                 errors.append(f"工具名称必须是字符串，得到: {type(action).__name__}")
#             else:
#                 # 检查是否为空
#                 if not action.strip():
#                     errors.append("工具名称不能为空")
#                 else:
#                     # 检查是否全小写
#                     if action != action.lower():
#                         errors.append(f"工具名称 '{action}' 包含大写字母，应该全小写")
                    
#                     # 检查是否在允许列表中
#                     normalized_tool = action.lower()
                    
#                     # 首先检查别名
#                     if normalized_tool in cls.TOOL_ALIASES:
#                         normalized_tool = cls.TOOL_ALIASES[normalized_tool]
                    
#                     if normalized_tool not in cls.ALLOWED_TOOLS:
#                         allowed = list(cls.ALLOWED_TOOLS.keys())
#                         errors.append(
#                             f"工具 '{action}' 不在允许列表中。允许的工具: {allowed}"
#                         )
        
#         # 5. 检查 Action Input
#         if action_input_key and action_input_key in tool_call:
#             action_input = tool_call[action_input_key]
            
#             if action_input is None:
#                 errors.append("Action Input 不能为 null")
#             elif isinstance(action_input, str):
#                 if len(action_input) == 0:
#                     errors.append("Action Input 不能为空字符串")
#             elif isinstance(action_input, dict):
#                 # 对于复杂的输入（如 JSON 对象）也是可以的
#                 pass
#             elif isinstance(action_input, list):
#                 # 对于列表输入也是可以的
#                 pass
#             else:
#                 # 其他类型也可以接受，但会转换为字符串
#                 pass
        
#         # 如果有错误，返回失败
#         if errors:
#             return {
#                 "valid": False,
#                 "errors": errors,
#                 "error": "; ".join(errors),
#                 "tool_call": tool_call
#             }
        
#         # 6. 生成规范化的工具调用
#         normalized_tool_call = None
#         if action_key and action_input_key:
#             action = tool_call[action_key].lower()
            
#             # 应用别名
#             if action in cls.TOOL_ALIASES:
#                 action = cls.TOOL_ALIASES[action]
            
#             normalized_tool_call = {
#                 "tool": action,
#                 "input": tool_call[action_input_key]
#             }
        
#         return {
#             "valid": True,
#             "tool_call": tool_call,
#             "normalized_tool_call": normalized_tool_call,
#             "errors": []
#         }
    
#     @classmethod
#     def validate_and_fix(cls, tool_call_str: str) -> Dict[str, Any]:
#         """
#         验证并尝试修复工具调用格式
        
#         Returns:
#             {
#                 "valid": bool,
#                 "original": str,
#                 "fixed": str (修复后的 JSON),
#                 "issues_found": List[str],
#                 "issues_fixed": List[str],
#                 "errors": List[str] (无法修复的错误)
#             }
#         """
        
#         # 首先进行标准验证
#         validation_result = cls.validate_json_format(tool_call_str)
        
#         if validation_result["valid"]:
#             return {
#                 "valid": True,
#                 "original": tool_call_str,
#                 "fixed": tool_call_str,
#                 "issues_found": [],
#                 "issues_fixed": [],
#                 "errors": []
#             }
        
#         # 尝试修复常见问题
#         issues_found = validation_result.get("errors", [])
#         issues_fixed = []
#         fixed_str = tool_call_str
        
#         try:
#             # 尝试处理不同的键名变体
#             tool_call = validation_result.get("tool_call")
            
#             if tool_call and isinstance(tool_call, dict):
#                 fixed_call = {}
                
#                 # 修复键名 - 必须严格转换为大写首字母
#                 for key, value in tool_call.items():
#                     key_lower = key.lower()
                    
#                     if key_lower == "action":
#                         fixed_call["Action"] = value
#                         if key != "Action":
#                             issues_fixed.append(f"修复键名: '{key}' → 'Action'")
#                     elif key_lower in ["action input", "action_input", "actioninput"]:
#                         fixed_call["Action Input"] = value
#                         if key != "Action Input":
#                             issues_fixed.append(f"修复键名: '{key}' → 'Action Input'")
#                     else:
#                         fixed_call[key] = value
                
#                 # 修复工具名称大小写 - 必须全小写
#                 if "Action" in fixed_call:
#                     action = fixed_call["Action"]
#                     if isinstance(action, str):
#                         lower_action = action.lower()
#                         fixed_call["Action"] = lower_action
#                         if action != lower_action:
#                             issues_fixed.append(f"修复工具名称大小写: '{action}' → '{lower_action}'")
                
#                 fixed_str = json.dumps(fixed_call, ensure_ascii=False, indent=2)
                
#                 # 再次验证修复后的版本
#                 re_validation = cls.validate_json_format(fixed_str)
                
#                 return {
#                     "valid": re_validation["valid"],
#                     "original": tool_call_str,
#                     "fixed": fixed_str,
#                     "issues_found": issues_found,
#                     "issues_fixed": issues_fixed,
#                     "errors": re_validation.get("errors", []) if not re_validation["valid"] else []
#                 }
        
#         except Exception as e:
#             return {
#                 "valid": False,
#                 "original": tool_call_str,
#                 "fixed": fixed_str,
#                 "issues_found": issues_found,
#                 "issues_fixed": issues_fixed,
#                 "errors": [f"修复过程中出错: {str(e)}"]
#             }
    
#     @classmethod
#     def get_format_example(cls, tool_name: str = None) -> str:
#         """
#         获取工具调用格式示例
        
#         Args:
#             tool_name: 工具名称，如果为 None 则返回所有工具的示例
            
#         Returns:
#             格式示例字符串
#         """
        
#         if tool_name:
#             # 规范化工具名称
#             normalized = tool_name.lower()
#             if normalized in cls.TOOL_ALIASES:
#                 normalized = cls.TOOL_ALIASES[normalized]
        
#         examples = {
#             "database_query": '''{
#   "Action": "database_query",
#   "Action Input": "SELECT COUNT(*) FROM orders WHERE created_at > '2024-01-01'"
# }''',
#             "generate_visualization": '''{
#   "Action": "generate_visualization",
#   "Action Input": "{\\"type\\": \\"bar\\", \\"title\\": \\"2024年月度销售额\\", \\"x_column\\": \\"month\\", \\"y_column\\": \\"sales\\"}"
# }''',
#         }
        
#         if tool_name and normalized in examples:
#             return examples[normalized]
        
#         # 返回所有示例
#         all_examples = "# 所有工具调用格式示例\n\n"
#         for tool, example in examples.items():
#             desc = cls.ALLOWED_TOOLS.get(tool, "")
#             all_examples += f"## {tool}\n说明：{desc}\n```json\n{example}\n```\n\n"
        
#         return all_examples
    
#     @classmethod
#     def validate_and_explain(cls, tool_call_str: str) -> str:
#         """
#         验证并生成详细的解释文本
        
#         Args:
#             tool_call_str: 工具调用字符串
            
#         Returns:
#             详细的验证报告
#         """
        
#         result = cls.validate_json_format(tool_call_str)
        
#         report = "# 工具调用验证报告\n\n"
#         report += f"## 验证结果: {'✅ 通过' if result['valid'] else '❌ 失败'}\n\n"
        
#         if not result["valid"]:
#             report += "## 发现的错误\n"
#             for i, error in enumerate(result.get("errors", []), 1):
#                 report += f"{i}. {error}\n"
#             report += "\n"
        
#         if result.get("tool_call"):
#             report += "## 解析后的工具调用\n"
#             report += "```json\n"
#             report += json.dumps(result["tool_call"], ensure_ascii=False, indent=2)
#             report += "\n```\n\n"
        
#         if result.get("normalized_tool_call"):
#             report += "## 规范化后的工具调用\n"
#             report += "```json\n"
#             report += json.dumps(result["normalized_tool_call"], ensure_ascii=False, indent=2)
#             report += "\n```\n\n"
        
#         if result["valid"]:
#             report += "## 建议\n"
#             report += "工具调用格式正确，可以执行。\n"
#         else:
#             report += "## 修复建议\n"
#             report += f"运行 `validate_and_fix()` 方法可以自动修复大部分格式问题。\n"
        
#         return report


#     @classmethod
#     def validate_key_value_format(cls, text: str) -> Dict[str, Any]:
#         """
#         验证键值对格式的工具调用
        
#         格式示例：
#         Action: database_query
#         Action Input: SELECT * FROM orders
        
#         Args:
#             text: 包含工具调用的文本（键值对格式）
            
#         Returns:
#             验证结果
#         """
        
#         errors = []
#         action = None
#         action_input = None
        
#         # 检查 Action: 前缀
#         action_match = re.search(r'Action:\s*(\S+)', text)
#         if not action_match:
#             errors.append("缺少 'Action:' 前缀（冒号后需要有空格）")
#         else:
#             action = action_match.group(1)
            
#             # 检查工具名称
#             if action not in cls.ALLOWED_TOOLS and action.lower() not in cls.ALLOWED_TOOLS:
#                 allowed = list(cls.ALLOWED_TOOLS.keys())
#                 errors.append(f"工具 '{action}' 不在允许列表中。允许的工具: {allowed}")
            
#             # 检查是否全小写
#             if action != action.lower():
#                 errors.append(f"工具名称 '{action}' 包含大写字母，应该全小写")
        
#         # 检查 Action Input: 前缀
#         input_match = re.search(r'Action\s+Input:\s*(.+?)(?:\n|$)', text)
#         if not input_match:
#             errors.append("缺少 'Action Input:' 前缀（注意中间有空格）")
#         else:
#             action_input = input_match.group(1).strip()
#             if not action_input:
#                 errors.append("Action Input 不能为空")
        
#         if errors:
#             return {
#                 "valid": False,
#                 "errors": errors,
#                 "error": "; ".join(errors)
#             }
        
#         return {
#             "valid": True,
#             "action": action.lower() if action else None,
#             "input": action_input,
#             "errors": []
#         }


# def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
#     """
#     从文本中提取所有工具调用（支持 JSON 和键值对格式）
    
#     Args:
#         text: 包含工具调用的文本
        
#     Returns:
#         提取到的工具调用列表
#     """
    
#     tool_calls = []
    
#     # 首先尝试提取键值对格式
#     kv_pattern = r'Action:\s*(\w+)\s*(?:\n|.+?\n)?Action\s+Input:\s*(.+?)(?:\n|$)'
#     kv_matches = re.finditer(kv_pattern, text, re.MULTILINE)
    
#     for match in kv_matches:
#         action = match.group(1)
#         action_input = match.group(2).strip()
        
#         result = ToolCallValidator.validate_key_value_format(match.group(0))
#         if result["valid"]:
#             tool_calls.append({
#                 "format": "key_value",
#                 "raw": match.group(0),
#                 "action": action.lower(),
#                 "input": action_input
#             })
    
#     # 然后尝试提取 JSON 格式
#     json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
#     json_matches = re.finditer(json_pattern, text)
    
#     for match in json_matches:
#         json_str = match.group()
#         result = ToolCallValidator.validate_json_format(json_str)
        
#         if result["valid"]:
#             tool_calls.append({
#                 "format": "json",
#                 "raw": json_str,
#                 "parsed": result["tool_call"],
#                 "normalized": result.get("normalized_tool_call")
#             })
    
#     return tool_calls


# # 使用示例和测试
# if __name__ == "__main__":
#     print("=" * 60)
#     print("工具调用格式验证工具 - 测试")
#     print("=" * 60)
#     print()
    
#     # 测试 1：正确的格式
#     print("测试 1️⃣  - 正确的工具调用格式")
#     print("-" * 60)
#     valid_call = '''{
#   "Action": "database_query",
#   "Action Input": "SELECT COUNT(*) FROM orders"
# }'''
    
#     result = ToolCallValidator.validate_json_format(valid_call)
#     print(f"验证结果: {'✅ 通过' if result['valid'] else '❌ 失败'}")
#     if result['valid']:
#         print(f"规范化工具调用: {result['normalized_tool_call']}")
#     print()
    
#     # 测试 2：错误的键名
#     print("测试 2️⃣  - 错误的键名（小写）")
#     print("-" * 60)
#     invalid_call = '''{
#   "action": "database_query",
#   "action_input": "SELECT COUNT(*) FROM orders"
# }'''
    
#     result = ToolCallValidator.validate_json_format(invalid_call)
#     print(f"验证结果: {'✅ 通过' if result['valid'] else '❌ 失败'}")
#     if not result['valid']:
#         print("错误:")
#         for error in result['errors']:
#             print(f"  - {error}")
    
#     # 尝试自动修复
#     print("\n尝试自动修复...")
#     fix_result = ToolCallValidator.validate_and_fix(invalid_call)
#     print(f"修复结果: {'✅ 成功' if fix_result['valid'] else '❌ 失败'}")
#     if fix_result['issues_fixed']:
#         print("已修复的问题:")
#         for issue in fix_result['issues_fixed']:
#             print(f"  - {issue}")
#     if fix_result['valid']:
#         print(f"修复后的工具调用:\n{fix_result['fixed']}")
#     print()
    
#     # 测试 3：工具名称大小写混乱
#     print("测试 3️⃣  - 工具名称大小写混乱")
#     print("-" * 60)
#     mixed_case_call = '''{
#   "Action": "Database_Query",
#   "Action Input": "SELECT COUNT(*) FROM orders"
# }'''
    
#     result = ToolCallValidator.validate_json_format(mixed_case_call)
#     print(f"验证结果: {'✅ 通过' if result['valid'] else '❌ 失败'}")
#     if not result['valid']:
#         print("错误:")
#         for error in result['errors']:
#             print(f"  - {error}")
#     print()
    
#     # 测试 4：格式示例
#     print("测试 4️⃣  - 获取格式示例")
#     print("-" * 60)
#     example = ToolCallValidator.get_format_example("database_query")
#     print("database_query 的示例:")
#     print(example)
#     print()
    
#     # 测试 5：详细验证报告
#     print("测试 5️⃣  - 生成详细验证报告")
#     print("-" * 60)
#     report = ToolCallValidator.validate_and_explain(valid_call)
#     print(report)
