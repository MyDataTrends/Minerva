"""
GitHub API client for agent operations.

Thin wrapper around the GitHub REST API using httpx.
Handles: issues, comments, labels, pull requests, repo metadata.
Falls back gracefully if httpx is not installed or token is missing.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from utils.logging import get_logger

logger = get_logger(__name__)

API_BASE = "https://api.github.com"


class GitHubClient:
    """
    Minimal GitHub REST API client for agent operations.

    Uses httpx for HTTP requests. Falls back to no-op if httpx
    is not installed or GITHUB_TOKEN is not set.
    """

    def __init__(self, token: Optional[str] = None, repo: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.repo = repo or os.getenv("GITHUB_REPO", "MyDataTrends/Assay")
        self._client = None
        self._available = False
        self._init_client()

    def _init_client(self) -> None:
        """Initialize httpx client if available."""
        if not self.token:
            logger.warning("GITHUB_TOKEN not set — GitHub client disabled")
            return

        try:
            import httpx
            self._client = httpx.Client(
                base_url=API_BASE,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=30.0,
            )
            self._available = True
            logger.info("GitHub client initialized for repo: %s", self.repo)
        except ImportError:
            logger.warning("httpx not installed — run: pip install httpx")

    @property
    def is_available(self) -> bool:
        return self._available

    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make an API request, return parsed JSON or None on failure."""
        if not self._available:
            return None
        try:
            url = endpoint.format(repo=self.repo)
            response = self._client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
        except Exception as exc:
            logger.error("GitHub API %s %s failed: %s", method, endpoint, exc)
            return None

    # ── Issues ───────────────────────────────────────────────────────

    def list_issues(self, state: str = "open", since: Optional[str] = None,
                    labels: Optional[str] = None, per_page: int = 30) -> List[Dict]:
        """List repository issues."""
        params: Dict[str, Any] = {"state": state, "per_page": per_page}
        if since:
            params["since"] = since
        if labels:
            params["labels"] = labels

        result = self._request("GET", "/repos/{repo}/issues", params=params)
        return result if isinstance(result, list) else []

    def get_issue(self, issue_number: int) -> Optional[Dict]:
        """Get a single issue by number."""
        return self._request("GET", f"/repos/{{repo}}/issues/{issue_number}")

    def create_comment(self, issue_number: int, body: str) -> Optional[Dict]:
        """Add a comment to an issue or PR."""
        return self._request(
            "POST",
            f"/repos/{{repo}}/issues/{issue_number}/comments",
            json={"body": body},
        )

    def add_labels(self, issue_number: int, labels: List[str]) -> Optional[Dict]:
        """Add labels to an issue."""
        return self._request(
            "POST",
            f"/repos/{{repo}}/issues/{issue_number}/labels",
            json={"labels": labels},
        )

    # ── Pull Requests ────────────────────────────────────────────────

    def list_pull_requests(self, state: str = "open") -> List[Dict]:
        """List pull requests."""
        result = self._request("GET", "/repos/{repo}/pulls", params={"state": state})
        return result if isinstance(result, list) else []

    def get_pull_request(self, pr_number: int) -> Optional[Dict]:
        """Get a single PR by number."""
        return self._request("GET", f"/repos/{{repo}}/pulls/{pr_number}")

    def get_pr_files(self, pr_number: int) -> List[Dict]:
        """Get files changed in a PR."""
        result = self._request("GET", f"/repos/{{repo}}/pulls/{pr_number}/files")
        return result if isinstance(result, list) else []

    # ── Repository ───────────────────────────────────────────────────

    def get_repo_info(self) -> Optional[Dict]:
        """Get repository metadata (stars, forks, open issues count)."""
        return self._request("GET", "/repos/{repo}")

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()

    def __del__(self):
        self.close()
