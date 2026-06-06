"""
向量数据库抽象基类
提供统一的向量数据库接口，支持多种向量数据库实现
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseVectorDB(ABC):
    """向量数据库抽象基类"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        初始化向量数据库
        
        Args:
            config: 数据库配置字典
        """
        pass
    
    @abstractmethod
    def create_collection(self, collection_name: str, **kwargs) -> bool:
        """
        创建集合
        
        Args:
            collection_name: 集合名称
            **kwargs: 其他参数
            
        Returns:
            创建是否成功
        """
        pass
    
    @abstractmethod
    def insert(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """
        插入文档
        
        Args:
            collection_name: 集合名称
            documents: 文档列表，每个文档包含 text, metadata 等字段
            
        Returns:
            插入的文档ID列表
        """
        pass
    
    @abstractmethod
    def search(
        self, 
        collection_name: str, 
        query: str, 
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        检索文档
        
        Args:
            collection_name: 集合名称
            query: 查询文本
            top_k: 返回结果数量
            **kwargs: 其他检索参数
            
        Returns:
            检索结果列表，每个结果包含 text, score, metadata 等字段
        """
        pass
    
    @abstractmethod
    def delete(self, collection_name: str, ids: List[str]) -> bool:
        """
        删除文档
        
        Args:
            collection_name: 集合名称
            ids: 文档ID列表
            
        Returns:
            删除是否成功
        """
        pass
    
    @abstractmethod
    def update(self, collection_name: str, documents: List[Dict[str, Any]]) -> bool:
        """
        更新文档
        
        Args:
            collection_name: 集合名称
            documents: 文档列表，必须包含 id 字段
            
        Returns:
            更新是否成功
        """
        pass
    
    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """
        检查集合是否存在
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合是否存在
        """
        pass
    
    @abstractmethod
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            统计信息字典（文档数量、大小等）
        """
        pass
    
    @abstractmethod
    def close(self):
        """关闭数据库连接"""
        pass
