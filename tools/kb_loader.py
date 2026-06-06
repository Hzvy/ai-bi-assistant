"""
知识库加载工具

用于初始化、加载和管理知识库的工具函数
支持 TXT、MD、PDF 等格式
包含文本清洗和智能分割功能
"""

import os
import re
from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from tools.tools_rag import rag_manager, get_kb_status


def load_text_file(file_path: Path) -> Optional[str]:
    """
    读取文本文件（支持多编码自动检测）
    
    功能说明:
        使用多种编码方式尝试读取文本文件，解决中文编码问题。
        支持 TXT、MD（Markdown）等纯文本格式。
    
    参数:
        file_path (Path): 文件路径对象
            示例: Path("data/kb_files/document.txt")
    
    返回:
        Optional[str]:
            - 成功: 文件内容字符串
            - 失败: None（所有编码都失败）
    
    编码尝试顺序:
        1. utf-8: 国际标准，优先尝试
        2. utf-8-sig: UTF-8 with BOM（Windows 记事本）
        3. gbk: 中文 Windows 常用编码
        4. gb2312: 旧版中文编码
        5. latin-1: 西欧编码（几乎不会失败，但可能乱码）
    
    工作原理:
        依次尝试每种编码，遇到 UnicodeDecodeError 则尝试下一个。
        某种编码成功后立即返回，不再尝试后续编码。
    
    错误处理:
        - UnicodeDecodeError: 编码不匹配，尝试下一个
        - LookupError: 编码名称无效，尝试下一个
        - 所有编码失败: 打印警告，返回 None
    
    使用示例:
        >>> from pathlib import Path
        >>> content = load_text_file(Path("data/kb_files/readme.txt"))
        >>> if content:
        ...     print(f"成功读取 {len(content)} 字符")
        成功读取 1523 字符
        
        >>> # 编码失败时
        >>> content = load_text_file(Path("binary_file.bin"))
        ⚠️ 无法读取文本文件 binary_file.bin，尝试了所有编码方式
        >>> print(content)
        None
    
    注意事项:
        - latin-1 作为兜底编码，几乎不会失败但可能产生乱码
        - 二进制文件（如图片、PDF）会失败
        - 建议上游调用者检查返回值是否为 None
    """
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    
    print(f"⚠️ 无法读取文本文件 {file_path}，尝试了所有编码方式")
    return None


def clean_text(text: str, custom_patterns: Optional[List[str]] = None, debug: bool = False) -> str:
    """
    清洗文本内容，去除噪音数据
    
    功能说明:
        从原始文本中移除页码、URL、特殊符号行等干扰信息。
        保留有效的中英文内容、代码块、结构化数据。
    
    参数:
        text (str): 原始文本内容
            示例: 包含页码、URL 的 PDF 提取文本
        custom_patterns (List[str], optional): 自定义要删除的固定字符串
            示例: ["公司机密", "内部文档", "Draft Version"]
        debug (bool): 是否打印详细清洗日志，默认 False
    
    返回:
        str: 清洗后的文本
    
    清洗规则（6 大类）:
        1. **自定义模式移除**
           - 删除 custom_patterns 中的所有字符串
           - 用途: 去除文档页眉页脚、水印
        
        2. **页码格式移除**
           - 格式1: "- 1 -", "- 23 -"
           - 格式2: "第1页", "第 23 页"
           - 格式3: "Page 1", "page 45"
        
        3. **URL 移除**
           - http/https 链接
           - www 开头的链接
           - 保留文本中的域名（如公司名）
        
        4. **特殊参数移除**
           - docId=123456
           - itemId=789
           - 其他系统生成的元数据
        
        5. **空白字符规范化**
           - 多个空行 → 最多2个换行符
           - 行首行尾空格清除
           - 保留段落间的单空行
        
        6. **纯符号行移除**
           - 删除: "---", "===", "..." (长度>10)
           - 保留: 短标题（如 "# 概述"）
           - 保留: 代码中的符号
    
    调试模式输出:
        开启 debug=True 时打印:
        - 移除的页码示例（前3个）
        - 移除的 URL 示例（前3个）
        - 移除的符号行示例（前3个）
        - 总计移除的字符数
    
    文本保留策略:
        ✅ 保留:
        - 中英文内容
        - 代码块（Python, SQL, JSON）
        - 短标题和列表项
        - 段落分隔的空行
        - 结构化数据（表格、列表）
        
        ❌ 删除:
        - 页码、页眉页脚
        - URL 和超链接
        - 长串符号（分隔线）
        - 系统生成的元数据
    
    使用示例:
        >>> raw_text = '''
        ... RK3588技术手册
        ... - 1 -
        ... 
        ... ## 性能参数
        ... CPU: 8核心
        ... -----------------------------------
        ... 详情访问: https://example.com/doc
        ... 第2页
        ... '''
        
        >>> cleaned = clean_text(raw_text, debug=True)
           🧹 清洗详情:
              - 页码格式1: ['- 1 -']
              - 页码格式2: ['第2页']
              - URL: ['https://example.com/doc']
              - 纯符号行: ['-----------------------------------']
              总计移除: 78 字符
        
        >>> print(cleaned)
        RK3588技术手册
        
        ## 性能参数
        CPU: 8核心
    
    注意事项:
        - 清洗会改变原始文本，建议保留备份
        - 正则匹配可能误删有效内容（如代码中的 URL）
        - debug 模式仅显示前3个示例，避免日志过长
        - 符号行长度阈值为 10，避免误删短分隔线
    """
    if not text:
        return ""
    
    original_text = text
    removed_items = []
    
    # 1. 去除自定义的固定字符串（如页眉页脚）
    if custom_patterns:
        for pattern in custom_patterns:
            if pattern in text:
                removed_items.append(f"自定义模式: '{pattern}'")
            text = text.replace(pattern, "")
    
    # 2. 去除常见的页码格式
    # "- 1 -", "第1页", "Page 1" 等
    page_nums = re.findall(r"-\s*\d+\s*-", text)
    if page_nums and debug:
        removed_items.append(f"页码格式1: {page_nums[:3]}")
    text = re.sub(r"-\s*\d+\s*-", "", text)
    
    page_nums2 = re.findall(r"第\s*\d+\s*页", text)
    if page_nums2 and debug:
        removed_items.append(f"页码格式2: {page_nums2[:3]}")
    text = re.sub(r"第\s*\d+\s*页", "", text)
    
    page_nums3 = re.findall(r"Page\s*\d+", text, flags=re.IGNORECASE)
    if page_nums3 and debug:
        removed_items.append(f"页码格式3: {page_nums3[:3]}")
    text = re.sub(r"Page\s*\d+", "", text, flags=re.IGNORECASE)
    
    # 3. 去除 URL
    urls = re.findall(r'https?://[^\s]+', text)
    if urls and debug:
        removed_items.append(f"URL: {urls[:3]}")
    text = re.sub(r'https?://[^\s]+', '', text)
    
    urls2 = re.findall(r'www\.[^\s]+', text)
    if urls2 and debug:
        removed_items.append(f"www链接: {urls2[:3]}")
    text = re.sub(r'www\.[^\s]+', '', text)
    
    # 4. 去除特殊参数（如 docId=xxx）
    doc_ids = re.findall(r'docId=\d+(&itemId=\d+)?', text)
    if doc_ids and debug:
        removed_items.append(f"文档ID: {doc_ids[:3]}")
    text = re.sub(r'docId=\d+(&itemId=\d+)?', '', text)
    
    # 5. 去除多余的空白字符
    # 将多个空行合并为一个
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    # 去除行首行尾空格
    text = '\n'.join(line.strip() for line in text.split('\n'))
    
    # 6. 去除纯符号行（如 "---", "===", "..."）
    # 但保留有实际内容的短行（如标题）
    lines = text.split('\n')
    cleaned_lines = []
    removed_symbol_lines = []
    
    for line in lines:
        stripped = line.strip()
        # 保留空行（用于段落分隔）
        if not stripped:
            cleaned_lines.append(line)
            continue
        # 删除纯符号行（只包含 -_=.*# 等符号）
        if re.match(r'^[-_=.*#\s]+$', stripped) and len(stripped) > 10:
            if debug:
                removed_symbol_lines.append(stripped[:30])
            continue
        # 保留其他所有行（包括短标题）
        cleaned_lines.append(line)
    
    if removed_symbol_lines and debug:
        removed_items.append(f"纯符号行: {removed_symbol_lines[:3]}")
    
    text = '\n'.join(cleaned_lines)
    
    # 输出调试信息
    if debug and removed_items:
        print("   🧹 清洗详情:")
        for item in removed_items:
            print(f"      - {item}")
        removed_chars = len(original_text) - len(text)
        if removed_chars > 0:
            print(f"      总计移除: {removed_chars} 字符")
    
    return text.strip()


def split_by_pattern(
    content: str, 
    pattern: str = r"第\S*条",
    min_chunk_size: int = 50
) -> List[str]:
    """
    根据正则表达式模式智能分割文本
    
    功能说明:
        使用正则表达式在文本中查找特定模式（如章节标题），
        并在这些位置分割文本，生成结构化的文档块。
    
    参数:
        content (str): 待分割的文本内容
            示例: 法律文档、技术规范、FAQ 文档
        pattern (str): 正则表达式匹配模式，默认 r"第\S*条"
            示例:
            - r"第\S*条": 匹配"第一条"、"第23条"
            - r"第\S*章": 匹配"第一章"、"第2章"
            - r"Q\d+:": 匹配"Q1:", "Q23:"
            - r"## ": 匹配 Markdown 二级标题
        min_chunk_size (int): 最小分块大小（字符数），默认 50
            小于此值的块会合并到前一个块
    
    返回:
        List[str]: 分割后的文本块列表
            - 每个块以匹配的模式开头
            - 包含该部分的完整内容
            - 过短的块已合并
    
    工作流程:
        1. 使用 re.finditer() 查找所有匹配位置
        2. 未找到匹配 → 返回整个文本作为单个块
        3. 找到匹配 → 在每个匹配位置切分
        4. 提取每个块的内容（从当前匹配到下一个匹配）
        5. 过滤或合并小于 min_chunk_size 的块
        6. 返回分块列表
    
    分割规则:
        - 每个块的起始位置: 匹配模式的开始
        - 每个块的结束位置: 下一个匹配的开始（或文本末尾）
        - 块内容包含: 标题 + 正文
    
    短块处理:
        - 长度 < min_chunk_size → 合并到前一个块
        - 第一个块太短 → 保留（无前置块可合并）
        - 合并方式: 使用双换行符连接 "\n\n"
    
    适用场景:
        1. **法规文档**
           >>> pattern = r"第\S*条"
           >>> chunks = split_by_pattern(law_text, pattern)
           ['第一条 总则...', '第二条 适用范围...']
        
        2. **技术手册**
           >>> pattern = r"第\S*章"
           >>> chunks = split_by_pattern(manual, pattern)
           ['第一章 概述...', '第二章 安装说明...']
        
        3. **FAQ 文档**
           >>> pattern = r"Q\d+:"
           >>> chunks = split_by_pattern(faq, pattern)
           ['Q1: 如何安装...', 'Q2: 如何配置...']
        
        4. **Markdown 文档**
           >>> pattern = r"^## "
           >>> chunks = split_by_pattern(md_text, pattern, min_chunk_size=100)
           ['## 快速开始\n内容...', '## API 参考\n内容...']
    
    使用示例:
        >>> text = '''
        ... 第一条 本规范适用于所有项目。
        ... 详细说明...
        ... 
        ... 第二条 项目需遵守以下要求：
        ... 1. 代码规范
        ... 2. 测试覆盖
        ... '''
        
        >>> chunks = split_by_pattern(text, pattern=r"第\S*条", min_chunk_size=20)
        >>> for i, chunk in enumerate(chunks, 1):
        ...     print(f"块{i}: {chunk[:30]}...")
        块1: 第一条 本规范适用于所有项目。详细说明...
        块2: 第二条 项目需遵守以下要求：1. 代码规范...
    
    注意事项:
        - pattern 需要使用 ^ 锚点匹配行首（re.MULTILINE 模式）
        - 如果文档无匹配模式，返回原文作为单个块
        - min_chunk_size 过大可能导致过度合并
        - 合并到前一个块时不会重新检查大小
    
    性能考虑:
        - 使用 re.finditer() 而非 re.split()，保留匹配内容
        - 一次遍历完成分割和过滤
        - 时间复杂度: O(n) - n 为文本长度
    """
    if not content or not content.strip():
        return []
    
    # 匹配所有符合模式的位置
    matches = list(re.finditer(rf"^{pattern}", content, re.MULTILINE))
    
    # 如果没有匹配到，返回原文
    if not matches:
        return [content.strip()] if content.strip() else []
    
    result = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        part = content[start:end].strip()
        
        # 过滤太短的分块
        if part and len(part) >= min_chunk_size:
            result.append(part)
        elif part and result:
            # 如果分块太短，合并到上一个分块
            result[-1] = result[-1] + "\n\n" + part
    
    return result


def auto_detect_split_pattern(text: str) -> Optional[str]:
    """
    自动检测文本的最佳分割模式
    
    功能说明:
        扫描文本内容，识别常见的结构化模式（章节、列表、FAQ等）。
        返回最适合的正则表达式分割模式。
    
    参数:
        text (str): 待检测的文本内容
    
    返回:
        Optional[str]:
            - 成功: 检测到的正则表达式模式
            - 失败: None（未检测到有效模式）
    
    检测模式列表:
        按优先级检测以下模式（至少出现3次才认为有效）:
        
        1. r"第\S*条" - 法规条款
           匹配: "第一条"、"第23条"、"第一百零八条"
           
        2. r"第\S*章" - 章节
           匹配: "第一章"、"第2章"、"第十五章"
           
        3. r"第\S*节" - 小节
           匹配: "第一节"、"第3节"
           
        4. r"^\d+\." - 数字列表
           匹配行首: "1. ", "23. ", "456. "
           
        5. r"^[A-Z]\." - 字母列表
           匹配行首: "A. ", "B. ", "Z. "
           
        6. r"^Q[:\s]" - 问答Q
           匹配行首: "Q: ", "Q1: ", "Q "
           
        7. r"^问[:\s]" - 问答中文
           匹配行首: "问: ", "问1: ", "问 "
    
    检测逻辑:
        - 使用 re.findall() 计数每种模式的出现次数
        - 出现次数 >= 3 → 认为是有效模式
        - 返回第一个满足条件的模式（按优先级）
        - 所有模式都不满足 → 返回 None
    
    有效性阈值:
        - 最小出现次数: 3
        - 原因: 避免偶然匹配（如单个 "第一条" 不构成结构）
        - 适用场景: 多章节文档、长 FAQ、法规文本
    
    使用示例:
        >>> law_text = '''
        ... 第一条 总则
        ... 本规范适用于...
        ... 
        ... 第二条 适用范围
        ... 适用于所有...
        ... 
        ... 第三条 实施日期
        ... 自发布之日起...
        ... '''
        
        >>> pattern = auto_detect_split_pattern(law_text)
           🔍 检测到分割模式: 法规条款 (共 3 处)
        >>> print(pattern)
        第\S*条
        
        >>> # 未检测到模式
        >>> plain_text = "这是一段普通文本，没有特殊结构"
        >>> pattern = auto_detect_split_pattern(plain_text)
        >>> print(pattern)
        None
    
    配合 split_by_pattern 使用:
        >>> pattern = auto_detect_split_pattern(text)
        >>> if pattern:
        ...     chunks = split_by_pattern(text, pattern)
        ...     print(f"自动分割为 {len(chunks)} 个块")
        ... else:
        ...     print("未检测到结构，保持原文")
    
    注意事项:
        - 仅检测常见中文模式，英文模式较少
        - 优先级固定（法规 > 章节 > 数字列表）
        - 阈值 3 适合大多数文档，短文档可能无法检测
        - 不会同时返回多个模式（返回第一个匹配）
    
    性能:
        - 时间复杂度: O(n * p) - n 为文本长度，p 为模式数量（7个）
        - 空间复杂度: O(1)
        - 适合在文档上传时调用一次
    """
    # 常见的分割模式
    patterns = [
        (r"第\S*条", "法规条款"),      # 第一条、第二条
        (r"第\S*章", "章节"),          # 第一章、第二章  
        (r"第\S*节", "小节"),          # 第一节、第二节
        (r"^\d+\.", "数字列表"),        # 1. 2. 3.
        (r"^[A-Z]\.", "字母列表"),      # A. B. C.
        (r"^Q[:\s]", "问答Q"),         # Q: 或 Q 
        (r"^问[:\s]", "问答中文"),      # 问: 或 问
    ]
    
    for pattern, name in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if len(matches) >= 3:  # 至少出现3次才认为是有效模式
            print(f"   🔍 检测到分割模式: {name} (共 {len(matches)} 处)")
            return pattern
    
    return None


def load_pdf_file(file_path: Path) -> Optional[str]:
    """
    读取 PDF 文件内容（多库支持）
    
    功能说明:
        使用 pdfplumber（优先）或 PyPDF2（降级）提取 PDF 文本。
        自动处理多页文档，合并所有页面内容。
    
    参数:
        file_path (Path): PDF 文件路径
            示例: Path("data/kb_files/manual.pdf")
    
    返回:
        Optional[str]:
            - 成功: 提取的文本内容
            - 失败: None（库未安装或提取失败）
    
    提取库优先级:
        1. **pdfplumber** (推荐)
           - 优点: 更准确，保留格式，支持表格
           - 适合: 技术文档、报告、表格数据
        
        2. **PyPDF2** (降级)
           - 优点: 轻量级，纯 Python
           - 缺点: 格式可能混乱，表格难处理
        
        3. 都未安装
           - 打印警告提示
           - 返回 None
    
    工作流程:
        1. 尝试 import pdfplumber
        2. 成功 → 打开 PDF，遍历所有页面
        3. 调用 page.extract_text() 提取文本
        4. 合并所有页面文本
        5. ImportError → 尝试 PyPDF2
        6. PyPDF2 也失败 → 返回 None
    
    文本提取特性:
        - pdfplumber:
          - 保留表格结构（转为文本表格）
          - 识别列布局
          - 保留段落换行
        
        - PyPDF2:
          - 基础文本提取
          - 可能丢失格式
          - 多列布局可能混乱
    
    错误处理:
        - ImportError: 库未安装，尝试下一个库
        - 文件损坏: 捕获异常，打印错误信息
        - 文本为空: 返回 None（可能是扫描版 PDF）
    
    使用示例:
        >>> from pathlib import Path
        >>> text = load_pdf_file(Path("manual.pdf"))
        >>> if text:
        ...     print(f"提取 {len(text)} 字符")
        ... else:
        ...     print("提取失败或文本为空")
        提取 15234 字符
        
        >>> # 库未安装时
        >>> text = load_pdf_file(Path("doc.pdf"))
        ⚠️ 无法读取 PDF 文件 doc.pdf（需要 pdfplumber 或 PyPDF2）
    
    安装依赖:
        # 推荐安装 pdfplumber
        pip install pdfplumber
        
        # 或安装 PyPDF2
        pip install PyPDF2
    
    注意事项:
        - 扫描版 PDF（图片型）无法提取文本，需 OCR
        - 加密 PDF 可能提取失败
        - 复杂排版可能导致文本顺序混乱
        - 大文件提取可能较慢
    
    性能:
        - pdfplumber: 较慢但准确（~1-2秒/页）
        - PyPDF2: 较快但粗糙（~0.5秒/页）
        - 建议: 技术文档用 pdfplumber，简单文档用 PyPDF2
    """
    try:
        # 尝试使用 pdfplumber
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text if text else None
        except ImportError:
            pass
        
        # 降级方案：使用 PyPDF2
        try:
            from PyPDF2 import PdfReader
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text if text else None
        except ImportError:
            pass
        
        print(f"⚠️ 无法读取 PDF 文件 {file_path}（需要 pdfplumber 或 PyPDF2）")
        return None
        
    except Exception as e:
        print(f"⚠️ 读取 PDF 文件失败 {file_path}: {str(e)}")
        return None


def load_kb_from_files(
    kb_path: str, 
    extensions: Optional[List[str]] = None,
    enable_cleaning: bool = True,
    enable_smart_split: bool = True,
    custom_clean_patterns: Optional[List[str]] = None,
    kb_level: int = 1,  # ← 新增：知识库权限级别
    department: str = ""  # ← 新增：部门限制
) -> int:
    """
    从文件系统批量加载知识库文档（增强版）
    
    功能说明:
        扫描指定目录下的所有文档文件，提取文本，
        执行清洗和智能分割，最终初始化向量数据库。
    
    参数:
        kb_path (str): 知识库文件所在的目录路径
            示例: "data/kb_files"
        extensions (List[str], optional): 要加载的文件扩展名
            默认: ['.txt', '.md', '.pdf']
            示例: ['.txt', '.docx', '.html']
        enable_cleaning (bool): 是否启用文本清洗，默认 True
            清洗内容: 页码、URL、纯符号行
        enable_smart_split (bool): 是否启用智能分割，默认 True
            分割方式: 自动检测章节模式（第X条、第X章等）
        custom_clean_patterns (List[str], optional): 自定义清洗模式
            示例: ["公司内部资料", "机密文档"]
    
    返回:
        int: 成功加载的文档数量
    
    工作流程:
        1. **路径检查**
           - 验证 kb_path 目录存在
           - 不存在 → 打印错误，返回 0
        
        2. **文件扫描**
           - 遍历 extensions 中的每个扩展名
           - 使用 Path.glob(f"**/*{ext}") 递归查找
           - 支持子目录
        
        3. **文本提取**
           - TXT/MD: load_text_file()（多编码）
           - PDF: load_pdf_file()（pdfplumber/PyPDF2）
        
        4. **文本清洗**（可选）
           - enable_cleaning=True → clean_text()
           - 移除页码、URL、符号行
           - 应用 custom_clean_patterns
        
        5. **智能分割**（可选）
           - enable_smart_split=True → auto_detect_split_pattern()
           - 检测到模式 → split_by_pattern()
           - 未检测到 → 保持原文
        
        6. **创建 Document 对象**
           - page_content: 文本内容
           - metadata: {source, file_type, chunk_index}
        
        7. **初始化向量库**
           - 调用 rag_manager.initialize(documents)
           - 嵌入向量化并写入 Milvus/Chroma
    
    文本处理流水线:
        原始文本 → 清洗 → 智能分割 → Document 对象 → 向量化 → 入库
    
    支持的文件类型:
        - .txt: 纯文本文件
        - .md: Markdown 文档
        - .pdf: PDF 文档（需 pdfplumber 或 PyPDF2）
    
    清洗效果示例:
        输入:
        ```
        技术手册
        - 1 -
        访问 https://example.com
        =========================
        第一章 概述
        ```
        
        输出（enable_cleaning=True）:
        ```
        技术手册
        
        第一章 概述
        ```
    
    智能分割示例:
        检测到 "第X章" 模式:
        ```
        原文 → ['第一章 概述\n内容...', '第二章 安装\n内容...']
        ```
        
        未检测到模式:
        ```
        原文 → ['完整文档内容']
        ```
    
    使用示例:
        >>> # 基础用法
        >>> count = load_kb_from_files("data/kb_files")
        🔧 文本清洗: ✅ 启用
        🔧 智能分割: ✅ 启用
        📄 正在加载: manual.pdf
           🔍 检测到分割模式: 章节 (共 5 处)
           ✂️  智能分割: 5 个块
        ✅ 成功加载 5 个文档
        >>> print(count)
        5
        
        >>> # 自定义配置
        >>> count = load_kb_from_files(
        ...     kb_path="docs",
        ...     extensions=['.txt', '.md'],
        ...     enable_cleaning=True,
        ...     enable_smart_split=False,
        ...     custom_clean_patterns=["内部资料"]
        ... )
    
    性能考虑:
        - 大文件处理较慢（清洗 + 分割）
        - PDF 提取可能耗时（取决于页数）
        - 向量化是最慢的步骤（调用嵌入模型）
        - 建议: 分批上传大量文档
    
    错误处理:
        - 目录不存在 → 返回 0
        - 单个文件失败 → 跳过，继续处理其他文件
        - 所有文件失败 → 返回 0
    
    注意事项:
        - enable_cleaning=False 可能导致噪音数据入库
        - enable_smart_split=False 可能产生过长的文档块
        - 默认配置（都启用）适合大多数场景
        - custom_clean_patterns 需要精确匹配整个字符串
    
    调试建议:
        设置 debug=True 查看清洗详情:
        >>> # 修改 clean_text() 调用
        >>> content = clean_text(content, custom_clean_patterns, debug=True)
           🧹 清洗详情:
              - 页码格式1: ['-1-', '-2-']
              - URL: ['https://...']
              总计移除: 245 字符
    """
    if extensions is None:
        extensions = ['.txt', '.md', '.pdf']
    
    kb_path = Path(kb_path)
    if not kb_path.exists():
        print(f"❌ 知识库路径不存在: {kb_path}")
        return 0
    
    documents = []
    
    try:
        print(f"🔧 文本清洗: {'✅ 启用' if enable_cleaning else '❌ 禁用'}")
        print(f"🔧 智能分割: {'✅ 启用' if enable_smart_split else '❌ 禁用'}")
        print(f"🔐 权限级别: Level {kb_level} ({['', '公开', '内部', '机密'][kb_level]})")
        
        # 级别到分类的映射
        category_map = {1: "public", 2: "internal", 3: "confidential"}
        kb_category = category_map.get(kb_level, "public")
        
        for ext in extensions:
            files = list(kb_path.glob(f"**/*{ext}"))
            for file in files:
                try:
                    content = None
                    
                    # 1. 根据文件类型读取内容
                    if ext.lower() == '.pdf':
                        content = load_pdf_file(file)
                    else:  # .txt, .md
                        content = load_text_file(file)
                    
                    if not content or not content.strip():
                        print(f"⚠️ 文件为空或无法解析: {file.name}")
                        continue
                    
                    print(f"📄 处理文件: {file.name} (原始: {len(content)} 字符)")
                    
                    # 2. 文本清洗（可选）
                    if enable_cleaning:
                        original_len = len(content)
                        content = clean_text(content, custom_clean_patterns, debug=True)
                        removed = original_len - len(content)
                        if removed > 0:
                            print(f"   📊 总计移除: {removed} 字符")
                    
                    # 3. 智能分割（可选）
                    text_chunks = []
                    if enable_smart_split:
                        # 自动检测分割模式
                        pattern = auto_detect_split_pattern(content)
                        if pattern:
                            text_chunks = split_by_pattern(content, pattern)
                            print(f"   ✂️  智能分割: {len(text_chunks)} 个语义块")
                        else:
                            # 未检测到模式，使用标准分割
                            text_chunks = [content]
                            print(f"   ℹ️  未检测到分割模式，使用整体内容")
                    else:
                        text_chunks = [content]
                    
                    # 4. 创建 Document 对象（添加权限元数据）
                    for i, chunk in enumerate(text_chunks):
                        if not chunk.strip():
                            continue
                        
                        metadata = {
                            "source": f"《{file.stem}》",
                            "file_path": str(file),
                            "file_type": ext,
                            "chunk_index": i,
                            "total_chunks": len(text_chunks),
                            # ===== 新增：权限控制元数据 =====
                            "kb_level": kb_level,
                            "kb_category": kb_category,
                            "department": department
                        }
                        
                        doc = Document(
                            page_content=chunk,
                            metadata=metadata
                        )
                        documents.append(doc)
                    
                    print(f"   ✅ 生成 {len(text_chunks)} 个文档块")
                
                except Exception as e:
                    print(f"⚠️ 无法处理文件 {file.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        if documents:
            print(f"\n📚 共加载 {len(documents)} 个文档块（来自 {len(files)} 个文件）")
            rag_manager.initialize(documents)
            
            status = get_kb_status()
            if status["available"]:
                print(f"✅ 知识库初始化成功: {status['message']}")
                return len(documents)
            else:
                print(f"⚠️ 知识库初始化失败: {status['message']}")
                return 0
        else:
            print(f"❌ 在 {kb_path} 中未找到相关文件")
            return 0
    
    except Exception as e:
        print(f"❌ 加载知识库出错: {str(e)}")
        return 0


def load_kb_from_directory(kb_dir: str = "knowledge_base") -> bool:
    """
    从默认知识库目录加载
    
    参数:
        kb_dir: 知识库目录名，默认为 "knowledge_base"
    
    返回:
        True: 成功加载
        False: 加载失败或目录不存在
    """
    
    if os.path.exists(kb_dir):
        count = load_kb_from_files(kb_dir)
        return count > 0
    else:
        print(f"⚠️ 知识库目录不存在: {kb_dir}")
        print(f"📝 请在项目根目录创建 '{kb_dir}' 目录并放入 .txt 或 .md 文件")
        return False


def try_load_persisted_kb() -> bool:
    """
    尝试加载已持久化的知识库
    
    返回:
        True: 成功加载
        False: 加载失败
    """
    return rag_manager.load_persisted_kb()


def get_kb_info() -> dict:
    """
    获取知识库信息
    
    返回:
        {
            "available": True/False,
            "status": "状态描述",
            "message": "详细信息"
        }
    """
    
    status = get_kb_status()
    
    info = {
        "available": status["available"],
        "status": status["message"],
        "message": ""
    }
    
    if status["available"]:
        info["message"] = "✅ 知识库已就绪，可以进行检索"
    else:
        info["message"] = """
⚠️ 知识库未初始化

初始化步骤：
1. 在项目根目录创建 'knowledge_base' 文件夹
2. 在文件夹中放入 .txt 或 .md 文件
3. 调用 load_kb_from_directory('knowledge_base') 来初始化知识库

或者：
- 调用 load_kb_from_files(path, extensions) 从指定路径加载
- 调用 try_load_persisted_kb() 尝试加载已持久化的知识库
"""
    
    return info


def initialize_kb_on_startup() -> bool:
    """
    在应用启动时初始化知识库
    
    尝试按以下顺序加载知识库：
    1. 加载已持久化的知识库
    2. 从 knowledge_base 目录加载
    3. 从 docs 目录加载
    4. 如果都失败，返回 False
    
    返回:
        True: 成功加载知识库
        False: 知识库加载失败或不可用
    """
    
    print("\n🔄 正在初始化知识库...")
    
    # 尝试1: 加载已持久化的知识库
    print("  1️⃣ 尝试加载已持久化的知识库...")
    if try_load_persisted_kb():
        print("     ✅ 从持久化存储加载成功\n")
        return True
    
    # 尝试2: 从 knowledge_base 目录加载
    print("  2️⃣ 尝试从 knowledge_base 目录加载...")
    if load_kb_from_directory("knowledge_base"):
        print("     ✅ 从 knowledge_base 目录加载成功\n")
        return True
    
    # 尝试3: 从 docs 目录加载
    print("  3️⃣ 尝试从 docs 目录加载...")
    if load_kb_from_directory("docs"):
        print("     ✅ 从 docs 目录加载成功\n")
        return True
    
    # 都失败了
    print("     ⚠️ 所有加载方式都失败了")
    print("\n📝 知识库初始化提示:")
    print("  - 知识库可选，系统会自动跳过知识库检索")
    print("  - 如需使用知识库功能，请参考上面的提示初始化\n")
    
    return False
