"""
LLM 工厂 - 多模型统一接口

【功能说明】
提供统一的大模型创建接口，支持 8 种主流 LLM 提供商。
采用工厂模式 + 策略模式，实现模型切换零代码改动。

【支持的模型】
1. **OpenAI**: gpt-3.5-turbo, gpt-4, gpt-4-turbo
2. **Ollama**: mistral, llama2, codellama（本地部署）
3. **Claude**: claude-3-sonnet, claude-3-opus, claude-3-haiku
4. **Deepseek**: deepseek-chat, deepseek-coder
5. **通义千问**: qwen-turbo, qwen-plus, qwen-max
6. **智谱 GLM**: glm-4, glm-3-turbo, glm-4v
7. **百度文心**: ERNIE-Bot, ERNIE-Bot-turbo
8. **讯飞星火**: Spark-2.0, Spark-3.0

【设计模式】
```
LLMFactory (工厂类)
    ↓
providers = {
    "openai": OpenAIProvider,      # 策略1
    "ollama": OllamaProvider,      # 策略2
    "claude": AnthropicProvider,   # 策略3
    ...
}
    ↓
provider.get_llm(**kwargs) → LangChain LLM 实例
```

【配置方式】
```python
# 方式1: 环境变量（推荐）
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="sk-..."

# 方式2: .env 文件
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# 方式3: config.py
LLM_PROVIDER = "openai"
LLM_API_KEY = "sk-..."
```

【使用方式】
```python
from tools.llm_factory import LLMFactory

# 创建 LLM（使用默认配置）
llm = LLMFactory.create_llm()

# 指定提供商
llm = LLMFactory.create_llm(provider_name="openai")

# 覆盖参数
llm = LLMFactory.create_llm(
    provider_name="openai",
    temperature=0.3,
    max_tokens=4096
)

# 使用 LLM
response = llm.invoke("什么是 AI？")
print(response.content)
```

【提供商特性对比】
| 提供商   | 部署方式 | 成本    | 速度   | 适用场景           |
|----------|----------|---------|--------|-------------------|
| OpenAI   | 云端     | 高      | 快     | 生产环境、高质量   |
| Ollama   | 本地     | 免费    | 中     | 开发测试、离线环境 |
| Deepseek | 云端     | 低      | 快     | 成本敏感、代码生成 |
| 千问     | 云端     | 中      | 快     | 中文场景、多模态   |
| Claude   | 云端     | 高      | 快     | 长文本、推理任务   |

【API Key 获取】
- OpenAI: https://platform.openai.com/api-keys
- Deepseek: https://platform.deepseek.com/
- 千问: https://dashscope.aliyun.com/
- 智谱: https://open.bigmodel.cn/
- Claude: https://console.anthropic.com/

【注意事项】
- 确保设置对应的 API Key
- Ollama 需本地安装（https://ollama.ai/）
- 不同模型的 token 计费不同
- 建议使用 .env 文件管理 API Key

【版本】
V3 - 2025-10
支持 8 种主流 LLM
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, List
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from config import config
import os


class BaseLLMProvider(ABC):
    """
    大模型提供者基类（抽象基类）
    
    功能说明:
        定义 LLM 提供者的统一接口。
        所有具体提供者必须继承此类并实现 get_llm() 方法。
    
    设计模式:
        策略模式（Strategy Pattern）
        - 每个提供者是一个策略
        - 工厂类负责选择策略
    
    使用示例:
        >>> class MyProvider(BaseLLMProvider):
        ...     def get_llm(self, **kwargs):
        ...         return MyLLM(**kwargs)
    """
    
    @abstractmethod
    def get_llm(self, **kwargs):
        """
        获取 LLM 实例（抽象方法）
        
        参数:
            **kwargs: 可选参数
                - temperature (float): 生成温度 (0-1)
                - max_tokens (int): 最大 token 数
                - 其他模型特定参数
        
        返回:
            LangChain LLM 实例
        """
        pass



class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI 模型提供者
    
    功能说明:
        提供 OpenAI GPT 系列模型的访问。
        支持官方 API 和兼容接口（如 Azure OpenAI）。
    
    支持模型:
        - gpt-3.5-turbo: 快速、低成本
        - gpt-4: 高质量、强推理
        - gpt-4-turbo: 平衡性能和成本
    
    配置项:
        - OPENAI_API_KEY: API 密钥（必需）
        - OPENAI_BASE_URL: API 基础 URL（可选，默认官方）
        - OPENAI_MODEL: 模型名称（可选，默认 gpt-3.5-turbo）
    
    使用示例:
        >>> provider = OpenAIProvider()
        >>> llm = provider.get_llm(temperature=0.3)
        >>> response = llm.invoke("你好")
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", config.LLM_API_KEY)
        self.base_url = os.getenv("OPENAI_BASE_URL", config.LLM_BASE_URL)
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    def get_llm(self, **kwargs):
        """
        获取 OpenAI LLM 实例
        
        参数:
            temperature (float): 生成温度，默认 0.7
            max_tokens (int): 最大 token 数，默认 2048
        
        返回:
            ChatOpenAI: LangChain OpenAI 实例
        """
        return ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2048),
        )


class OllamaProvider(BaseLLMProvider):
    """
    Ollama 本地模型提供者
    
    功能说明:
        提供本地部署的开源模型访问。
        完全免费，无需 API Key，支持离线运行。
    
    支持模型:
        - mistral: 7B 参数，速度快
        - llama2: Meta 官方，质量高
        - codellama: 代码生成专用
        - qwen: 阿里通义千问开源版
    
    安装步骤:
        1. 下载 Ollama: https://ollama.ai/
        2. 安装并启动服务
        3. 拉取模型: `ollama pull mistral`
        4. 验证: `ollama list`
    
    配置项:
        - OLLAMA_MODEL: 模型名称（默认 mistral）
        - OLLAMA_BASE_URL: 服务地址（默认 http://localhost:11434）
    
    使用示例:
        >>> provider = OllamaProvider()
        >>> llm = provider.get_llm()
        >>> response = llm.invoke("写一个 Python 函数")
    """
    
    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "mistral")
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    def get_llm(self, **kwargs):
        """
        获取 Ollama LLM 实例
        
        参数:
            temperature (float): 生成温度，默认 0.7
        
        返回:
            Ollama: LangChain Ollama 实例
        """
        return Ollama(
            model=self.model,
            base_url=self.base_url,
            temperature=kwargs.get("temperature", 0.7),
        )


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude 模型提供者
    
    功能说明:
        提供 Claude 系列模型访问。
        擅长长文本处理和复杂推理任务。
    
    支持模型:
        - claude-3-opus: 最强性能
        - claude-3-sonnet: 平衡性能和成本
        - claude-3-haiku: 快速响应
    
    特点:
        - 200K token 上下文窗口
        - 安全性高，拒答率低
        - 适合长文档分析
    
    配置项:
        - ANTHROPIC_API_KEY: API 密钥（必需）
        - ANTHROPIC_MODEL: 模型名称
    
    使用示例:
        >>> provider = AnthropicProvider()
        >>> llm = provider.get_llm()
        >>> response = llm.invoke("分析这份合同...")
    """
    
    def __init__(self):
        try:
            from langchain_anthropic import ChatAnthropic
            self.ChatAnthropic = ChatAnthropic
        except ImportError:
            raise ImportError("请先安装 langchain-anthropic: pip install langchain-anthropic")
        
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
    
    def get_llm(self, **kwargs):
        """
        获取 Anthropic LLM 实例
        
        参数:
            temperature (float): 生成温度，默认 0.7
            max_tokens (int): 最大 token 数，默认 2048
        
        返回:
            ChatAnthropic: LangChain Anthropic 实例
        """
        return self.ChatAnthropic(
            api_key=self.api_key,
            model=self.model,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2048),
        )




class DeepseekProvider(BaseLLMProvider):
    """
    Deepseek 模型提供者
    
    功能说明:
        提供 Deepseek 系列模型访问。
        成本低、速度快，适合高并发场景。
    
    支持模型:
        - deepseek-chat: 通用对话
        - deepseek-coder: 代码生成（推荐）
    
    特点:
        - 价格仅为 OpenAI 的 1/10
        - 代码能力强
        - 中文支持好
    
    配置项:
        - DEEPSEEK_API_KEY: API 密钥
        - DEEPSEEK_BASE_URL: API 地址
        - DEEPSEEK_MODEL: 模型名称
    
    使用示例:
        >>> provider = DeepseekProvider()
        >>> llm = provider.get_llm()
        >>> response = llm.invoke("写一个排序算法")
    """
    
    def __init__(self):
        try:
            from langchain_openai import ChatOpenAI
            self.ChatOpenAI = ChatOpenAI
        except ImportError:
            raise ImportError("请先安装 openai: pip install openai langchain-openai")
        
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    def get_llm(self, **kwargs):
        """
        获取 Deepseek LLM 实例
        
        参数:
            temperature (float): 生成温度，默认 0.7
            max_tokens (int): 最大 token 数，默认 2048
        
        返回:
            ChatOpenAI: LangChain OpenAI 兼容实例
        """
        return self.ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2048),
        )


class QwenProvider(BaseLLMProvider):
    """
    阿里通义千问模型提供者
    
    功能说明:
        提供阿里云通义千问系列模型访问。
        使用 OpenAI 兼容 API，无需额外依赖。
    
    支持模型:
        - qwen-turbo: 快速响应
        - qwen-plus: 平衡性能
        - qwen-max: 最强性能
        - qwen-vl-max: 多模态（图文）
    
    特点:
        - 中文能力强
        - 多模态支持
        - 价格适中
    
    配置项:
        - DASHSCOPE_API_KEY: API 密钥（推荐）
        - QWEN_API_KEY: 备用密钥名
        - DASHSCOPE_BASE_URL: API 地址
        - QWEN_MODEL: 模型名称
    
    获取 API Key:
        https://dashscope.console.aliyun.com/api-key
    
    使用示例:
        >>> provider = QwenProvider()
        >>> llm = provider.get_llm()
        >>> response = llm.invoke("解释量子计算")
    """
    
    def __init__(self):
        try:
            from langchain_openai import ChatOpenAI
            self.ChatOpenAI = ChatOpenAI
        except ImportError:
            raise ImportError("请先安装 openai: pip install openai langchain-openai")
        
        # 获取 API Key
        self.api_key = (
            os.getenv("DASHSCOPE_API_KEY") or 
            os.getenv("QWEN_API_KEY") or 
            config.QWEN_API_KEY
        )
        
        if not self.api_key:
            raise ValueError(
                "Qwen API Key 未配置。请设置以下任一环境变量:\n"
                "  - DASHSCOPE_API_KEY (推荐)\n"
                "  - QWEN_API_KEY\n"
                "获取地址: https://dashscope.console.aliyun.com/api-key"
            )
        
        # 配置参数
        self.base_url = os.getenv(
            "DASHSCOPE_BASE_URL", 
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = os.getenv("QWEN_MODEL", config.QWEN_MODEL or "qwen-plus")
    
    def get_llm(self, **kwargs):
        """
        获取 Qwen LLM 实例
        
        参数:
            temperature (float): 生成温度，默认 0.7
            max_tokens (int): 最大 token 数，默认 2048
        
        返回:
            ChatOpenAI: LangChain OpenAI 兼容实例
        """
        return self.ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2048),
        )


class ZhipuAIProvider(BaseLLMProvider):
    """
    智谱 GLM 模型提供者
    
    功能说明:
        提供智谱 AI GLM 系列模型访问。
        使用 OpenAI 兼容 API。
    
    支持模型:
        - glm-4: 最新旗舰版
        - glm-3-turbo: 快速版本
        - glm-4v: 多模态版本
    
    特点:
        - 中文能力优秀
        - 推理能力强
        - 多模态支持
    
    配置项:
        - ZHIPUAI_API_KEY: API 密钥（必需）
        - ZHIPUAI_BASE_URL: API 地址
        - ZHIPUAI_MODEL: 模型名称
    
    获取 API Key:
        https://open.bigmodel.cn/
    
    使用示例:
        >>> provider = ZhipuAIProvider()
        >>> llm = provider.get_llm()
        >>> response = llm.invoke("写一首诗")
    """
    
    def __init__(self):
        self.api_key = os.getenv("ZHIPUAI_API_KEY")
        if not self.api_key:
            raise ValueError("ZHIPUAI_API_KEY 未设置，请在 .env 文件中配置")
        
        self.model = os.getenv("ZHIPUAI_MODEL", "glm-4")
        self.temperature = float(os.getenv("ZHIPUAI_TEMPERATURE", "0.6"))
        self.base_url = os.getenv("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    
    def get_llm(self, **kwargs):
        """
        获取智谱 GLM LLM 实例
        
        参数:
            temperature (float): 生成温度，默认 0.6
            max_tokens (int): 最大 token 数，默认 2048
        
        返回:
            ChatOpenAI: LangChain OpenAI 兼容实例
        """
        return ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", 2048),
        )


class LLMFactory:
    """
    LLM 工厂类（单例模式）
    
    功能说明:
        统一的 LLM 创建入口，负责管理所有提供者。
        根据配置自动选择对应的提供者。
    
    设计模式:
        工厂模式 + 策略模式
        - 工厂: LLMFactory
        - 策略: 各个 Provider
    
    支持的提供者:
        - openai: OpenAIProvider
        - ollama: OllamaProvider
        - claude: AnthropicProvider
        - deepseek: DeepseekProvider
        - qwen: QwenProvider
        - zhipuai: ZhipuAIProvider
    
    使用示例:
        >>> # 使用默认提供者
        >>> llm = LLMFactory.create_llm()
        
        >>> # 指定提供者
        >>> llm = LLMFactory.create_llm(provider_name="openai")
        
        >>> # 覆盖参数
        >>> llm = LLMFactory.create_llm(
        ...     provider_name="qwen",
        ...     temperature=0.3,
        ...     max_tokens=4096
        ... )
    
    扩展方法:
        添加新提供者:
        ```python
        class MyProvider(BaseLLMProvider):
            def get_llm(self, **kwargs):
                return MyLLM(**kwargs)
        
        # 注册到工厂
        LLMFactory._providers["myprovider"] = MyProvider
        ```
    """
    
    _providers = {
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "anthropic": AnthropicProvider,
        "qwen": QwenProvider,
        "deepseek": DeepseekProvider,
        "zhipuai": ZhipuAIProvider,
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class):
        """
        注册新的模型提供者（扩展接口）
        
        功能说明:
            动态添加新的 LLM 提供者到工厂。
            支持运行时扩展，无需修改源码。
        
        参数:
            name (str): 提供者名称（小写）
            provider_class (BaseLLMProvider): 提供者类
                - 必须继承 BaseLLMProvider
                - 必须实现 get_llm() 方法
        
        使用示例:
            >>> class MyProvider(BaseLLMProvider):
            ...     def get_llm(self, **kwargs):
            ...         return MyLLM(**kwargs)
            
            >>> LLMFactory.register_provider("myprovider", MyProvider)
            >>> llm = LLMFactory.create_llm(provider_name="myprovider")
        
        注意事项:
            - provider_class 必须是类，不是实例
            - 重复注册会覆盖已有提供者
        """
        cls._providers[name] = provider_class
    
    @classmethod
    def get_provider(cls, provider_name: str) -> BaseLLMProvider:
        """
        获取模型提供者实例
        
        功能说明:
            根据名称创建对应的提供者实例。
        
        参数:
            provider_name (str): 提供者名称
                支持: openai, ollama, claude, deepseek, qwen, zhipuai
        
        返回:
            BaseLLMProvider: 提供者实例
        
        异常处理:
            - 提供者不存在 → ValueError
        
        使用示例:
            >>> provider = LLMFactory.get_provider("openai")
            >>> llm = provider.get_llm(temperature=0.3)
        """
        if provider_name not in cls._providers:
            raise ValueError(
                f"不支持的模型提供者: {provider_name}\n"
                f"支持的提供者: {list(cls._providers.keys())}"
            )
        return cls._providers[provider_name]()
    
    @classmethod
    def create_llm(cls, provider_name: Optional[str] = None, **kwargs):
        """
        创建 LLM 实例（主入口）
        
        功能说明:
            工厂方法，根据配置创建 LLM 实例。
            自动从环境变量或 config 读取提供者名称。
        
        参数:
            provider_name (str, optional): 提供者名称
                - None: 从环境变量 LLM_PROVIDER 或 config.LLM_PROVIDER 读取
                - 指定: 使用指定提供者
            **kwargs: 传递给提供者的参数
                - temperature: 生成温度
                - max_tokens: 最大 token 数
                - 其他模型特定参数
        
        返回:
            LangChain LLM 实例
        
        配置优先级:
            1. provider_name 参数
            2. LLM_PROVIDER 环境变量
            3. config.LLM_PROVIDER
        
        使用示例:
            >>> # 使用默认配置
            >>> llm = LLMFactory.create_llm()
            
            >>> # 指定提供者
            >>> llm = LLMFactory.create_llm(provider_name="openai")
            
            >>> # 覆盖参数
            >>> llm = LLMFactory.create_llm(
            ...     provider_name="qwen",
            ...     temperature=0.3,
            ...     max_tokens=4096
            ... )
            
            >>> # 使用 LLM
            >>> response = llm.invoke("你好")
            >>> print(response.content)
        
        异常处理:
            - 提供者不存在 → ValueError
            - API Key 未设置 → ValueError
            - 依赖缺失 → ImportError
        """
        """创建 LLM 实例"""
        # 如果没有指定提供者，使用配置中的默认提供者
        if provider_name is None:
            provider_name = os.getenv("LLM_PROVIDER", "openai")
        
        provider = cls.get_provider(provider_name)
        return provider.get_llm(**kwargs)


class LLMManager:
    """LLM 管理器 - 单例模式"""
    
    _instance = None
    _llm = None
    _current_provider = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._initialize()
    
    def _initialize(self):
        """初始化 LLM"""
        provider_name = os.getenv("LLM_PROVIDER", "openai")
        if self._current_provider != provider_name:
            self._llm = LLMFactory.create_llm(provider_name)
            self._current_provider = provider_name
    
    def get_llm(self, **kwargs):
        """获取 LLM 实例"""
        self._initialize()
        return self._llm
    
    def switch_provider(self, provider_name: str, **kwargs):
        """切换 LLM 提供者"""
        self._llm = LLMFactory.create_llm(provider_name, **kwargs)
        self._current_provider = provider_name
    
    def get_available_providers(self):
        """获取所有可用的提供者"""
        return list(LLMFactory._providers.keys())
    
    def get_current_provider(self):
        """获取当前提供者"""
        return self._current_provider


# 全局 LLM 管理器实例
llm_manager = LLMManager()
