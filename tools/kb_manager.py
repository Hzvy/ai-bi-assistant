"""
知识库管理器 - 处理文件上传、更新和重新加载等业务功能
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from langchain.schema import Document
from tools.tools_rag import rag_manager
from tools.kb_loader import load_kb_from_files, get_kb_status
from tools.embedding_factory import EmbeddingFactory


class KBManager:
    """知识库管理器"""
    
    # 知识库目录
    KB_DIR = Path("data/kb_files")
    KB_SOURCE = Path("data/knowledge_base.txt")
    BACKUP_DIR = Path("data/kb_backup")
    
    @classmethod
    def ensure_dirs_exist(cls):
        """确保必要的目录存在"""
        cls.KB_DIR.mkdir(parents=True, exist_ok=True)
        cls.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def upload_file(cls, uploaded_file) -> Tuple[bool, str]:
        """
        上传文件到知识库
        
        Args:
            uploaded_file: Streamlit 上传的文件对象
        
        Returns:
            (success, message)
        """
        try:
            cls.ensure_dirs_exist()
            
            # 获取文件信息
            file_name = uploaded_file.name
            file_content = uploaded_file.read()
            file_path = cls.KB_DIR / file_name
            
            # 保存文件
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            return True, f"✅ 文件已上传: {file_name} ({len(file_content) / 1024:.1f} KB)"
        
        except Exception as e:
            return False, f"❌ 上传失败: {str(e)}"
    
    @classmethod
    def merge_files_to_kb(cls) -> Tuple[bool, str]:
        """
        将 kb_files 目录中的所有文件合并到 knowledge_base.txt
        
        Returns:
            (success, message)
        """
        try:
            cls.ensure_dirs_exist()
            
            merged_content = []
            file_count = 0
            total_size = 0
            
            # 遍历所有支持的文件类型
            for ext in ['*.txt', '*.md', '*.pdf']:
                for file_path in cls.KB_DIR.glob(ext):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # 添加文件分隔符和来源信息
                            merged_content.append(f"\n\n{'='*50}")
                            merged_content.append(f"来源: {file_path.name}")
                            merged_content.append(f"{'='*50}\n")
                            merged_content.append(content)
                            
                            file_count += 1
                            total_size += len(content)
                    except Exception as e:
                        print(f"⚠️ 读取文件失败 {file_path}: {str(e)}")
            
            if not merged_content:
                return False, "❌ 没有找到要合并的文件"
            
            # 备份原有知识库
            if cls.KB_SOURCE.exists():
                backup_path = cls.BACKUP_DIR / f"knowledge_base_backup_{cls._get_timestamp()}.txt"
                shutil.copy(cls.KB_SOURCE, backup_path)
            
            # 写入合并后的内容
            merged_text = "".join(merged_content)
            with open(cls.KB_SOURCE, 'w', encoding='utf-8') as f:
                f.write(merged_text)
            
            return True, f"✅ 已合并 {file_count} 个文件到知识库 ({total_size / 1024:.1f} KB)"
        
        except Exception as e:
            return False, f"❌ 合并失败: {str(e)}"
    
    @classmethod
    def reload_kb(cls) -> Tuple[bool, str]:
        """
        重新加载知识库
        
        Returns:
            (success, message)
        """
        try:
            # 检查知识库源文件
            if not cls.KB_SOURCE.exists():
                return False, "❌ 知识库文件不存在"
            
            # 读取知识库文件
            with open(cls.KB_SOURCE, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 创建 Document 对象
            documents = [Document(
                page_content=content,
                metadata={"source": str(cls.KB_SOURCE)}
            )]
            
            # 重新初始化 RAG
            rag_manager.initialize(documents)
            
            if rag_manager.is_kb_available():
                return True, "✅ 知识库已重新加载"
            else:
                return False, "❌ 知识库加载失败"
        
        except Exception as e:
            return False, f"❌ 重新加载失败: {str(e)}"
    
    @classmethod
    def get_kb_stats(cls) -> dict:
        """
        获取知识库统计信息
        
        Returns:
            包含统计信息的字典
        """
        try:
            stats = {
                "exists": cls.KB_SOURCE.exists(),
                "size": 0,
                "last_modified": None,
                "is_available": rag_manager.is_kb_available(),
                "file_count": 0,
                "upload_dir_size": 0
            }
            
            if cls.KB_SOURCE.exists():
                import datetime
                stat = cls.KB_SOURCE.stat()
                stats["size"] = stat.st_size
                stats["last_modified"] = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            # 统计上传目录中的文件
            cls.ensure_dirs_exist()
            for file_path in cls.KB_DIR.glob("**/*"):
                if file_path.is_file():
                    stats["file_count"] += 1
                    stats["upload_dir_size"] += file_path.stat().st_size
            
            return stats
        
        except Exception as e:
            print(f"❌ 获取统计信息失败: {str(e)}")
            return {}
    
    @classmethod
    def clear_kb_cache(cls) -> Tuple[bool, str]:
        """
        清除知识库缓存
        
        Returns:
            (success, message)
        """
        try:
            # 清除 vectordb
            rag_manager.vectordb = None
            rag_manager._kb_available = False
            
            # 清除 ChromaDB 持久化目录（可选）
            chroma_dir = Path("./chroma_db")
            if chroma_dir.exists():
                shutil.rmtree(chroma_dir)
            
            return True, "✅ 知识库缓存已清除"
        
        except Exception as e:
            return False, f"❌ 清除缓存失败: {str(e)}"
    
    @classmethod
    def _get_timestamp(cls) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @classmethod
    def list_uploaded_files(cls) -> List[dict]:
        """
        列出上传目录中的所有文件
        
        Returns:
            文件列表，每个元素包含 name 和 size
        """
        try:
            cls.ensure_dirs_exist()
            files = []
            
            for file_path in sorted(cls.KB_DIR.glob("**/*")):
                if file_path.is_file():
                    files.append({
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                        "path": str(file_path)
                    })
            
            return files
        
        except Exception as e:
            print(f"❌ 获取文件列表失败: {str(e)}")
            return []
    
    @classmethod
    def delete_file(cls, file_name: str) -> Tuple[bool, str]:
        """
        删除上传的文件
        
        Args:
            file_name: 要删除的文件名
        
        Returns:
            (success, message)
        """
        try:
            file_path = cls.KB_DIR / file_name
            
            if not file_path.exists():
                return False, f"❌ 文件不存在: {file_name}"
            
            file_path.unlink()
            return True, f"✅ 文件已删除: {file_name}"
        
        except Exception as e:
            return False, f"❌ 删除失败: {str(e)}"
