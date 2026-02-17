"""
Knowledge Base — Tier 2 curated markdown persistence.

Long-lived knowledge organized by domain (product, customers, market, operations).
Agents read and write markdown files; humans review diffs weekly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from utils.logging import get_logger

logger = get_logger(__name__)

KB_DIR = Path(__file__).parent.parent / "knowledge_base"


class KnowledgeBase:
    """
    Markdown-backed knowledge base for curated agent knowledge.

    Structure:
        knowledge_base/
        ├── product/       (architecture, limitations, roadmap)
        ├── customers/     (personas, feedback, churn)
        ├── market/        (competitors, trends, pricing)
        └── operations/    (runbooks, escalation policy, SLAs)
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or KB_DIR
        self._ensure_structure()

    def _ensure_structure(self) -> None:
        """Create the knowledge base directory structure if it doesn't exist."""
        for subdir in ["product", "customers", "market", "operations"]:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

    def read_doc(self, relative_path: str) -> Optional[str]:
        """
        Read a knowledge base document.

        Args:
            relative_path: Path relative to knowledge_base/, e.g. "product/roadmap.md"

        Returns:
            File contents as string, or None if not found.
        """
        filepath = self.base_dir / relative_path
        if not filepath.exists():
            return None
        try:
            return filepath.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to read KB doc %s: %s", relative_path, exc)
            return None

    def write_doc(self, relative_path: str, content: str) -> bool:
        """
        Write or overwrite a knowledge base document.

        Args:
            relative_path: Path relative to knowledge_base/, e.g. "product/vision_gap_analysis.md"
            content: Markdown content to write.

        Returns:
            True if successful.
        """
        filepath = self.base_dir / relative_path
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            filepath.write_text(content, encoding="utf-8")
            logger.info("KB doc written: %s (%d chars)", relative_path, len(content))
            return True
        except Exception as exc:
            logger.error("Failed to write KB doc %s: %s", relative_path, exc)
            return False

    def append_doc(self, relative_path: str, section: str) -> bool:
        """Append a section to an existing document (or create it)."""
        existing = self.read_doc(relative_path) or ""
        return self.write_doc(relative_path, existing + "\n" + section)

    def list_docs(self, subdirectory: str = "") -> List[str]:
        """List all markdown files in a subdirectory (or all of knowledge_base)."""
        search_dir = self.base_dir / subdirectory if subdirectory else self.base_dir
        if not search_dir.exists():
            return []
        return [
            str(p.relative_to(self.base_dir))
            for p in search_dir.rglob("*.md")
        ]

    def search_docs(self, query: str) -> List[Dict[str, str]]:
        """
        Simple keyword search across all knowledge base documents.

        Returns list of {"path": ..., "snippet": ...} for matching docs.
        """
        results = []
        query_lower = query.lower()
        for md_file in self.base_dir.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                if query_lower in content.lower():
                    # Extract a snippet around the first match
                    idx = content.lower().index(query_lower)
                    start = max(0, idx - 100)
                    end = min(len(content), idx + len(query) + 100)
                    snippet = content[start:end].strip()
                    results.append({
                        "path": str(md_file.relative_to(self.base_dir)),
                        "snippet": snippet,
                    })
            except Exception:
                continue
        return results
