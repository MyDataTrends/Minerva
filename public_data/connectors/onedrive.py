"""
OneDrive/SharePoint Connector - Microsoft Graph API integration.

Provides OAuth 2.0 device code flow for secure file access without
storing credentials.
"""

import os
import logging
import json
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import requests

from public_data.connectors.base import DataConnector, DataSeries

logger = logging.getLogger(__name__)


# Microsoft Graph API endpoints
GRAPH_BASE = "https://graph.microsoft.com/v1.0"
AUTH_ENDPOINT = "https://login.microsoftonline.com/common/oauth2/v2.0"


@dataclass
class OneDriveFile:
    """Represents a file in OneDrive/SharePoint."""
    id: str
    name: str
    path: str
    size: int
    mime_type: str
    modified: datetime
    download_url: Optional[str] = None
    
    @property
    def is_data_file(self) -> bool:
        """Check if this is a supported data file."""
        exts = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
        return any(self.name.lower().endswith(ext) for ext in exts)


class OneDriveConnector(DataConnector):
    """
    Connector for Microsoft OneDrive and SharePoint.
    
    Uses OAuth 2.0 device code flow for authentication, which:
    - Doesn't require a redirect URI or web server
    - User authenticates via browser at microsoft.com/devicelogin
    - Works for both personal and business accounts
    """
    
    id = "onedrive"
    name = "Microsoft OneDrive"
    description = "Access files from OneDrive or SharePoint"
    auth_type = "oauth2_device_code"
    
    # Default scopes for file access
    SCOPES = [
        "Files.Read",
        "Files.Read.All", 
        "User.Read",
        "offline_access"
    ]
    
    def __init__(self, client_id: Optional[str] = None):
        """
        Initialize the OneDrive connector.
        
        Args:
            client_id: Azure AD app client ID. If not provided,
                      reads from ONEDRIVE_CLIENT_ID env var.
        """
        super().__init__()
        self.client_id = client_id or os.getenv("ONEDRIVE_CLIENT_ID")
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires: Optional[datetime] = None
        self._user_info: Optional[Dict] = None
        
        # Token storage path
        self._token_path = os.path.join(
            os.path.expanduser("~"), ".assay", "onedrive_token.json"
        )
    
    def _load_cached_token(self) -> bool:
        """Load cached token if available and not expired."""
        try:
            if os.path.exists(self._token_path):
                with open(self._token_path, 'r') as f:
                    data = json.load(f)
                
                expires = datetime.fromisoformat(data['expires'])
                if expires > datetime.now() + timedelta(minutes=5):
                    self.access_token = data['access_token']
                    self.refresh_token = data.get('refresh_token')
                    self.token_expires = expires
                    return True
                elif data.get('refresh_token'):
                    # Try to refresh
                    return self._refresh_access_token(data['refresh_token'])
        except Exception as e:
            logger.debug(f"No cached token: {e}")
        return False
    
    def _save_token(self):
        """Save token to cache."""
        try:
            os.makedirs(os.path.dirname(self._token_path), exist_ok=True)
            with open(self._token_path, 'w') as f:
                json.dump({
                    'access_token': self.access_token,
                    'refresh_token': self.refresh_token,
                    'expires': self.token_expires.isoformat() if self.token_expires else None
                }, f)
        except Exception as e:
            logger.warning(f"Could not cache token: {e}")
    
    def _refresh_access_token(self, refresh_token: str) -> bool:
        """Use refresh token to get new access token."""
        if not self.client_id:
            return False
        
        try:
            resp = requests.post(
                f"{AUTH_ENDPOINT}/token",
                data={
                    'client_id': self.client_id,
                    'refresh_token': refresh_token,
                    'grant_type': 'refresh_token',
                    'scope': ' '.join(self.SCOPES)
                }
            )
            
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data['access_token']
                self.refresh_token = data.get('refresh_token', refresh_token)
                self.token_expires = datetime.now() + timedelta(seconds=data.get('expires_in', 3600))
                self._save_token()
                return True
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
        return False
    
    def start_device_auth(self) -> Dict[str, str]:
        """
        Start device code authentication flow.
        
        Returns:
            Dict with 'user_code', 'verification_uri', and 'device_code'
            
        User should:
        1. Go to verification_uri (usually microsoft.com/devicelogin)
        2. Enter user_code
        3. Sign in with their Microsoft account
        """
        if not self.client_id:
            raise ValueError(
                "OneDrive client_id not configured. "
                "Set ONEDRIVE_CLIENT_ID environment variable or "
                "register an app at https://portal.azure.com"
            )
        
        resp = requests.post(
            f"{AUTH_ENDPOINT}/devicecode",
            data={
                'client_id': self.client_id,
                'scope': ' '.join(self.SCOPES)
            }
        )
        
        if resp.status_code != 200:
            raise RuntimeError(f"Device auth failed: {resp.text}")
        
        data = resp.json()
        return {
            'user_code': data['user_code'],
            'verification_uri': data['verification_uri'],
            'device_code': data['device_code'],
            'expires_in': data.get('expires_in', 900),
            'interval': data.get('interval', 5),
            'message': data.get('message', '')
        }
    
    def poll_for_token(self, device_code: str, interval: int = 5, timeout: int = 300) -> bool:
        """
        Poll for token after user completes device login.
        
        Args:
            device_code: From start_device_auth()
            interval: Seconds between polls
            timeout: Max seconds to wait
            
        Returns:
            True if authenticated successfully
        """
        start = time.time()
        
        while time.time() - start < timeout:
            resp = requests.post(
                f"{AUTH_ENDPOINT}/token",
                data={
                    'client_id': self.client_id,
                    'device_code': device_code,
                    'grant_type': 'urn:ietf:params:oauth:grant-type:device_code'
                }
            )
            
            data = resp.json()
            
            if resp.status_code == 200:
                self.access_token = data['access_token']
                self.refresh_token = data.get('refresh_token')
                self.token_expires = datetime.now() + timedelta(seconds=data.get('expires_in', 3600))
                self._save_token()
                logger.info("OneDrive authentication successful")
                return True
            
            error = data.get('error')
            if error == 'authorization_pending':
                time.sleep(interval)
                continue
            elif error == 'slow_down':
                time.sleep(interval + 5)
                continue
            elif error in ('authorization_declined', 'expired_token'):
                logger.error(f"Authentication failed: {error}")
                return False
            else:
                logger.error(f"Unknown auth error: {data}")
                return False
        
        logger.error("Authentication timed out")
        return False
    
    def is_authenticated(self) -> bool:
        """Check if we have a valid token."""
        if self._load_cached_token():
            return True
        return bool(self.access_token and self.token_expires and self.token_expires > datetime.now())
    
    def _headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        if not self.access_token:
            raise RuntimeError("Not authenticated. Call start_device_auth() first.")
        return {'Authorization': f'Bearer {self.access_token}'}
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get current user information."""
        if self._user_info:
            return self._user_info
        
        resp = requests.get(f"{GRAPH_BASE}/me", headers=self._headers())
        if resp.status_code == 200:
            self._user_info = resp.json()
            return self._user_info
        raise RuntimeError(f"Failed to get user info: {resp.text}")
    
    def list_files(
        self, 
        folder: str = "root",
        filter_data: bool = True
    ) -> List[OneDriveFile]:
        """
        List files in a OneDrive folder.
        
        Args:
            folder: Folder path or 'root' for root folder
            filter_data: If True, only return supported data files
            
        Returns:
            List of OneDriveFile objects
        """
        if folder == "root":
            url = f"{GRAPH_BASE}/me/drive/root/children"
        else:
            url = f"{GRAPH_BASE}/me/drive/root:/{folder}:/children"
        
        resp = requests.get(url, headers=self._headers())
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to list files: {resp.text}")
        
        files = []
        for item in resp.json().get('value', []):
            if 'file' in item:
                f = OneDriveFile(
                    id=item['id'],
                    name=item['name'],
                    path=item.get('parentReference', {}).get('path', '') + '/' + item['name'],
                    size=item.get('size', 0),
                    mime_type=item.get('file', {}).get('mimeType', ''),
                    modified=datetime.fromisoformat(item['lastModifiedDateTime'].replace('Z', '+00:00')),
                    download_url=item.get('@microsoft.graph.downloadUrl')
                )
                if not filter_data or f.is_data_file:
                    files.append(f)
        
        return files
    
    def fetch_data(
        self,
        file_id: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Download and parse a file as DataFrame.
        
        Args:
            file_id: OneDrive file ID
            **kwargs: Additional args passed to pd.read_csv/read_excel
            
        Returns:
            DataFrame with file contents
        """
        # Get file metadata with download URL
        url = f"{GRAPH_BASE}/me/drive/items/{file_id}"
        resp = requests.get(url, headers=self._headers())
        
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to get file: {resp.text}")
        
        item = resp.json()
        download_url = item.get('@microsoft.graph.downloadUrl')
        name = item.get('name', '')
        
        if not download_url:
            # Need to request download URL
            url = f"{GRAPH_BASE}/me/drive/items/{file_id}/content"
            resp = requests.get(url, headers=self._headers(), allow_redirects=True)
            content = resp.content
        else:
            resp = requests.get(download_url)
            content = resp.content
        
        # Parse based on extension
        import io
        
        if name.endswith('.csv'):
            return pd.read_csv(io.BytesIO(content), **kwargs)
        elif name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(io.BytesIO(content), **kwargs)
        elif name.endswith('.json'):
            return pd.read_json(io.BytesIO(content), **kwargs)
        elif name.endswith('.parquet'):
            return pd.read_parquet(io.BytesIO(content), **kwargs)
        else:
            # Try CSV as fallback
            return pd.read_csv(io.BytesIO(content), **kwargs)
    
    def get_available_series(self) -> List[DataSeries]:
        """Return empty - OneDrive is file-based not series-based."""
        return []
