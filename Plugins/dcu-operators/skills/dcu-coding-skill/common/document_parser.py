"""
DCU Coding Skill Document Parser

文档解析工具，用于解析不同格式的文档并转换为统一格式
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class CommonDoc:
    """统一文档模型"""
    doc_id: str
    title: str
    role: str  # authoritative, guide, playbook
    domain: str
    content: str
    headings: List[Dict[str, Any]]
    source_uri: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class DocumentParser:
    """文档解析器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化文档解析器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = self._get_logger()

    def _get_logger(self):
        """获取日志记录器"""
        from .utils import Logger
        return Logger(__name__)

    def parse_document(self, source_type: str, source_path: Union[str, Path]) -> CommonDoc:
        """
        解析文档并转换为 CommonDoc 模型

        Args:
            source_type: 文档源类型 (markdown, pdf, html, etc.)
            source_path: 文档路径

        Returns:
            CommonDoc 对象
        """
        source_path = Path(source_path)
        self.logger.info(f"解析文档: {source_type} - {source_path}")

        if source_type == "markdown":
            return self._parse_markdown(source_path)
        elif source_type == "pdf":
            return self._parse_pdf(source_path)
        elif source_type == "html":
            return self._parse_html(source_path)
        else:
            raise ValueError(f"不支持的文档类型: {source_type}")

    def _parse_markdown(self, file_path: Path) -> CommonDoc:
        """
        解析 Markdown 文档

        Args:
            file_path: Markdown 文件路径

        Returns:
            CommonDoc 对象
        """
        content = file_path.read_text(encoding="utf-8")

        # 解析 frontmatter
        frontmatter, content = self._parse_frontmatter(content)

        # 提取标题
        title = frontmatter.get("title", file_path.stem)

        # 识别 role
        role = self._identify_role(file_path, content, frontmatter)

        # 识别 domain
        domain = self._identify_domain(content, frontmatter)

        # 提取标题列表
        headings = self._extract_headings(content)

        # 移除 Source 和 Warnings 前言
        content = self._remove_prefixes(content)

        # 生成 doc_id
        doc_id = self._generate_doc_id(file_path)

        return CommonDoc(
            doc_id=doc_id,
            title=title,
            role=role,
            domain=domain,
            content=content,
            headings=headings,
            source_uri=str(file_path)
        )

    def _parse_pdf(self, file_path: Path) -> CommonDoc:
        """
        解析 PDF 文档

        Args:
            file_path: PDF 文件路径

        Returns:
            CommonDoc 对象
        """
        # TODO: 实现实际的 PDF 解析
        # 这里暂时返回示例数据
        self.logger.warning(f"PDF 解析功能尚未实现: {file_path}")

        return CommonDoc(
            doc_id=f"pdf_{file_path.stem}_{hash(str(file_path))}",
            title=file_path.stem,
            role="guide",
            domain="general",
            content="[PDF 内容待解析]",
            headings=[],
            source_uri=str(file_path)
        )

    def _parse_html(self, file_path: Path) -> CommonDoc:
        """
        解析 HTML 文档

        Args:
            file_path: HTML 文件路径

        Returns:
            CommonDoc 对象
        """
        # TODO: 实现实际的 HTML 解析
        # 这里暂时返回示例数据
        self.logger.warning(f"HTML 解析功能尚未实现: {file_path}")

        return CommonDoc(
            doc_id=f"html_{file_path.stem}_{hash(str(file_path))}",
            title=file_path.stem,
            role="guide",
            domain="general",
            content="[HTML 内容待解析]",
            headings=[],
            source_uri=str(file_path)
        )

    def _parse_frontmatter(self, content: str) -> tuple:
        """
        解析 YAML frontmatter

        Args:
            content: 文档内容

        Returns:
            (frontmatter, content) 元组
        """
        # 匹配 --- 开头的 frontmatter
        pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
        match = re.match(pattern, content, re.DOTALL)

        if match:
            frontmatter_content = match.group(1)
            content = match.group(2)

            # 解析 YAML frontmatter
            import yaml
            try:
                frontmatter = yaml.safe_load(frontmatter_content)
                return frontmatter or {}, content
            except yaml.YAMLError:
                self.logger.warning(f"无法解析 frontmatter: {frontmatter_content[:100]}...")
                return {}, content
        else:
            return {}, content

    def _identify_role(self, file_path: Path, content: str, frontmatter: Dict) -> str:
        """
        识别文档 role

        Args:
            file_path: 文件路径
            content: 文档内容
            frontmatter: frontmatter 内容

        Returns:
            role (authoritative/guide/playbook)
        """
        # 优先从 frontmatter 读取
        if "role" in frontmatter:
            return frontmatter["role"]

        # 从文件名识别
        filename = file_path.name.lower()
        if "api" in filename or "reference" in filename:
            return "authoritative"
        elif "guide" in filename or "tutorial" in filename:
            return "guide"
        elif "playbook" in filename or "best-practice" in filename:
            return "playbook"

        # 从内容识别
        if "## API" in content or "## Function" in content:
            return "authoritative"
        elif "## Tutorial" in content or "## Getting Started" in content:
            return "guide"
        elif "## Best Practices" in content or "## Examples" in content:
            return "playbook"

        # 默认为 guide
        return "guide"

    def _identify_domain(self, content: str, frontmatter: Dict) -> str:
        """
        识别文档 domain

        Args:
            content: 文档内容
            frontmatter: frontmatter 内容

        Returns:
            domain
        """
        # 优先从 frontmatter 读取
        if "domain" in frontmatter:
            return frontmatter["domain"]

        # 从内容识别 domain 关键词
        domain_keywords = {
            "runtime": ["hipMalloc", "hipMemcpy", "hipFree"],
            "fft": ["hipFFT", "fft"],
            "blas": ["rocBLAS", "blas", "gemm"],
            "linear-algebra": ["hipBlas", "gemm", "gemv"],
            "solver": ["hipsolver", "linear-solver"],
            "compress": ["hipCompression"],
        }

        content_lower = content.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword.lower() in content_lower for keyword in keywords):
                return domain

        return "general"

    def _extract_headings(self, content: str) -> List[Dict[str, Any]]:
        """
        提取文档中的所有标题

        Args:
            content: 文档内容

        Returns:
            标题列表
        """
        headings = []

        # 匹配 Markdown 标题
        pattern = r'^(#{1,6})\s+(.+)$'
        for line in content.split('\n'):
            match = re.match(pattern, line)
            if match:
                level = len(match.group(1))
                title = match.group(2)
                headings.append({
                    "heading": title,
                    "level": level
                })

        return headings

    def _remove_prefixes(self, content: str) -> str:
        """
        移除文档开头的 Source 和 Warnings 前言

        Args:
            content: 文档内容

        Returns:
            处理后的内容
        """
        # 移除 ## Source 前言
        if content.startswith("## Source"):
            content = content.split('\n\n', 1)[1] if '\n\n' in content else ""

        # 移除 ## Warnings 前言
        if content.startswith("## Warnings"):
            content = content.split('\n\n', 1)[1] if '\n\n' in content else ""

        return content

    def _generate_doc_id(self, file_path: Path) -> str:
        """
        生成文档唯一 ID

        Args:
            file_path: 文件路径

        Returns:
            doc_id
        """
        import hashlib

        # 使用文件名和内容生成 hash
        content = file_path.read_text(encoding="utf-8")
        hash_obj = hashlib.md5(f"{file_path}{content}".encode("utf-8"))
        doc_hash = hash_obj.hexdigest()[:8]

        return f"{file_path.stem}_{doc_hash}"
