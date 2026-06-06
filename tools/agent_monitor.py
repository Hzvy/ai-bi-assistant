"""
Agent 性能监控工具
用于跟踪和诊断 Agent 执行的性能问题
"""

import time
import json
from typing import Dict, List, Optional
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler


class PerformanceMetrics:
    """性能指标收集"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.iteration_count = 0
        self.tool_calls = []  # 记录每个工具调用
        self.errors = []  # 记录错误
        self.final_output = None
    
    @property
    def total_time(self) -> float:
        """总耗时（秒）"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0
    
    @property
    def average_iteration_time(self) -> float:
        """平均每次迭代时间"""
        if self.iteration_count > 0:
            return self.total_time / self.iteration_count
        return 0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "total_time": f"{self.total_time:.2f}s",
            "iteration_count": self.iteration_count,
            "average_iteration_time": f"{self.average_iteration_time:.2f}s",
            "tool_calls": len(self.tool_calls),
            "errors": len(self.errors),
            "tool_breakdown": self._get_tool_breakdown(),
        }
    
    def _get_tool_breakdown(self) -> Dict:
        """获取工具使用统计"""
        breakdown = {}
        for tool_call in self.tool_calls:
            tool_name = tool_call.get("tool")
            if tool_name not in breakdown:
                breakdown[tool_name] = {"count": 0, "total_time": 0}
            breakdown[tool_name]["count"] += 1
            breakdown[tool_name]["total_time"] += tool_call.get("duration", 0)
        
        # 计算平均时间
        for tool_name in breakdown:
            count = breakdown[tool_name]["count"]
            total = breakdown[tool_name]["total_time"]
            breakdown[tool_name]["average_time"] = f"{total/count:.2f}s" if count > 0 else "0s"
            breakdown[tool_name]["total_time"] = f"{total:.2f}s"
        
        return breakdown


class AgentPerformanceMonitor(BaseCallbackHandler):
    """Agent 性能监控回调"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.metrics = PerformanceMetrics()
        self._current_tool_start = None
    
    def on_agent_start(self, **kwargs):
        """Agent 开始执行"""
        self.metrics.start_time = time.time()
        if self.verbose:
            print("\n" + "="*60)
            print("🚀 Agent 开始执行")
            print("="*60)
    
    def on_agent_action(self, action, **kwargs):
        """Agent 执行 Action"""
        self.metrics.iteration_count += 1
        elapsed = time.time() - self.metrics.start_time
        
        self._current_tool_start = time.time()
        
        if self.verbose:
            print(f"\n📍 迭代 #{self.metrics.iteration_count} | 耗时: {elapsed:.1f}s")
            print(f"   🔧 工具: {action.tool}")
            print(f"   📝 输入: {str(action.tool_input)[:100]}...")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """工具开始执行"""
        pass
    
    def on_tool_end(self, output, **kwargs):
        """工具执行完成"""
        if self._current_tool_start:
            duration = time.time() - self._current_tool_start
            tool_name = kwargs.get("tool", "unknown")
            
            self.metrics.tool_calls.append({
                "tool": tool_name,
                "duration": duration,
                "output_length": len(str(output)),
            })
            
            if self.verbose:
                print(f"   ✅ 结果: {str(output)[:80]}... (耗时: {duration:.2f}s)")
    
    def on_tool_error(self, error: Exception, **kwargs):
        """工具执行出错"""
        self.metrics.errors.append({
            "tool": kwargs.get("tool", "unknown"),
            "error": str(error),
        })
        
        if self.verbose:
            print(f"   ❌ 错误: {str(error)[:100]}")
    
    def on_agent_finish(self, finish, **kwargs):
        """Agent 执行完成"""
        self.metrics.end_time = time.time()
        self.metrics.final_output = finish.output
        
        if self.verbose:
            self._print_summary()
    
    def _print_summary(self):
        """打印总结"""
        metrics_dict = self.metrics.to_dict()
        
        print("\n" + "="*60)
        print("📊 执行总结")
        print("="*60)
        print(f"⏱️  总耗时:        {metrics_dict['total_time']}")
        print(f"🔄 迭代次数:      {metrics_dict['iteration_count']}")
        print(f"⚡ 平均迭代时间:  {metrics_dict['average_iteration_time']}")
        print(f"🔧 工具调用数:    {metrics_dict['tool_calls']}")
        print(f"❌ 错误数:        {metrics_dict['errors']}")
        
        # 打印工具统计
        if metrics_dict['tool_breakdown']:
            print(f"\n📈 工具使用统计:")
            for tool_name, stats in metrics_dict['tool_breakdown'].items():
                print(f"   • {tool_name}:")
                print(f"     - 调用次数: {stats['count']}")
                print(f"     - 总耗时: {stats['total_time']}")
                print(f"     - 平均耗时: {stats['average_time']}")
        
        # 性能评级
        print(f"\n⭐ 性能评级: ", end="")
        rating = self._calculate_rating(metrics_dict)
        print(rating)
        
        # 优化建议
        suggestions = self._get_suggestions(metrics_dict)
        if suggestions:
            print(f"\n💡 优化建议:")
            for suggestion in suggestions:
                print(f"   • {suggestion}")
        
        print("="*60 + "\n")
    
    def _calculate_rating(self, metrics_dict: Dict) -> str:
        """计算性能评级"""
        total_time = float(metrics_dict['total_time'].replace('s', ''))
        iterations = metrics_dict['iteration_count']
        errors = metrics_dict['errors']
        
        # 评分逻辑
        if total_time < 5 and iterations <= 3 and errors == 0:
            return "⭐⭐⭐⭐⭐ 优秀"
        elif total_time < 15 and iterations <= 5 and errors <= 1:
            return "⭐⭐⭐⭐ 良好"
        elif total_time < 30 and iterations <= 8 and errors <= 2:
            return "⭐⭐⭐ 中等"
        elif total_time < 60 and iterations <= 10:
            return "⭐⭐ 一般"
        else:
            return "⭐ 需要优化"
    
    def _get_suggestions(self, metrics_dict: Dict) -> List[str]:
        """获取优化建议"""
        suggestions = []
        
        total_time = float(metrics_dict['total_time'].replace('s', ''))
        iterations = metrics_dict['iteration_count']
        errors = metrics_dict['errors']
        
        if iterations >= 10:
            suggestions.append("迭代次数较多，考虑优化系统提示词或工具设计")
        
        if total_time > 30:
            suggestions.append("总耗时较长，检查是否有慢工具（数据库查询、API 调用）")
        
        if errors > 0:
            suggestions.append("执行中有错误，改进错误处理或工具可靠性")
        
        # 工具分析
        tool_breakdown = metrics_dict['tool_breakdown']
        for tool_name, stats in tool_breakdown.items():
            avg_time = float(stats['average_time'].replace('s', ''))
            if avg_time > 10:
                suggestions.append(f"工具 '{tool_name}' 平均耗时较长，考虑添加缓存或优化")
        
        return suggestions


class SimpleAgentMonitor(BaseCallbackHandler):
    """简单的 Agent 监控（在 Streamlit UI 中使用）"""
    
    def __init__(self):
        self.iteration_count = 0
        self.start_time = None
        self.status_placeholder = None
    
    def on_agent_start(self, **kwargs):
        """Agent 开始"""
        self.iteration_count = 0
        self.start_time = time.time()
    
    def on_agent_action(self, action, **kwargs):
        """Agent 执行 Action"""
        self.iteration_count += 1
        elapsed = time.time() - self.start_time
        status = f"🔄 处理中... (迭代: {self.iteration_count}, 耗时: {elapsed:.1f}s)"
        
        # 如果在 Streamlit 中，更新状态
        try:
            import streamlit as st
            st.session_state.agent_status = status
        except:
            pass
    
    def on_agent_finish(self, finish, **kwargs):
        """Agent 完成"""
        elapsed = time.time() - self.start_time
        status = f"✅ 完成 (迭代: {self.iteration_count}, 耗时: {elapsed:.1f}s)"
        
        try:
            import streamlit as st
            st.session_state.agent_status = status
        except:
            pass


def create_performance_monitor(verbose: bool = True) -> AgentPerformanceMonitor:
    """创建性能监控"""
    return AgentPerformanceMonitor(verbose=verbose)


def create_simple_monitor() -> SimpleAgentMonitor:
    """创建简单监控（用于 UI）"""
    return SimpleAgentMonitor()


if __name__ == "__main__":
    # 测试性能监控
    print("性能监控工具已就绪")
    print("使用方法:")
    print("  from tools.agent_monitor import create_performance_monitor")
    print("  monitor = create_performance_monitor()")
    print("  executor.invoke(input_dict, callbacks=[monitor])")
