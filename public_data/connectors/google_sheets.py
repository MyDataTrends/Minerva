"""
Google Sheets Connector - Read spreadsheets via Google Sheets API.

Provides OAuth 2.0 authentication for secure access to Google Sheets
without storing credentials.
"""

import os
import logging
import json
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from public_data.connectors.base import DataConnector, DataSeries

logger = logging.getLogger(__name__)


# Google OAuth endpoints
GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"
DRIVE_API = "https://www.googleapis.com/drive/v3/files"


@dataclass
class GoogleSheet:
    """Represents a Google Sheet."""
    id: str
    name: str
    modified: Optional[datetime] = None
    owner: Optional[str] = None
    
    @property
    def url(self) -> str:
        return f"https://docs.google.com/spreadsheets/d/{self.id}"


class GoogleSheetsConnector(DataConnector):
    """
    Connector for Google Sheets.
    
    Uses OAuth 2.0 with local redirect for authentication.
    """
    
    id = "google_sheets"
    name = "Google Sheets"
    description = "Access data from Google Sheets"
    auth_type = "oauth2"
    
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly"
    ]
    
    def __init__(
        self, 
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None
    ):
        """
        Initialize Google Sheets connector.
        
        Args:
            client_id: Google OAuth client ID
            client_secret: Google OAuth client secret
        """
        super().__init__()
        self.client_id = client_id or os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("GOOGLE_CLIENT_SECRET")
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires: Optional[datetime] = None
        
        # Token storage
        self._token_path = os.path.join(
            os.path.expanduser("~"), ".assay", "google_token.json"
        )
    
    def _load_cached_token(self) -> bool:
        """Load cached token if available."""
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
        if not self.client_id or not self.client_secret:
            return False
        
        try:
            resp = requests.post(
                GOOGLE_TOKEN,
                data={
                    'client_id': self.client_id,
                    'client_secret': self.client_secret,
                    'refresh_token': refresh_token,
                    'grant_type': 'refresh_token'
                }
            )
            
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data['access_token']
                self.refresh_token = refresh_token
                self.token_expires = datetime.now() + timedelta(seconds=data.get('expires_in', 3600))
                self._save_token()
                return True
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
        return False
    
    def get_auth_url(self, redirect_uri: str = "http://localhost:8503/callback") -> str:
        """
        Get the OAuth authorization URL.
        
        Args:
            redirect_uri: Where to redirect after auth
            
        Returns:
            URL for user to visit
        """
        if not self.client_id:
            raise ValueError(
                "Google client_id not configured. "
                "Set GOOGLE_CLIENT_ID environment variable."
            )
        
        import urllib.parse
        
        params = {
            'client_id': self.client_id,
            'redirect_uri': redirect_uri,
            'response_type': 'code',
            'scope': ' '.join(self.SCOPES),
            'access_type': 'offline',
            'prompt': 'consent'
        }
        
        return f"{GOOGLE_AUTH}?{urllib.parse.urlencode(params)}"
    
    def exchange_code(
        self, 
        code: str, 
        redirect_uri: str = "http://localhost:8503/callback"
    ) -> bool:
        """
        Exchange authorization code for tokens.
        
        Args:
            code: Authorization code from OAuth callback
            redirect_uri: Same redirect_uri used in get_auth_url
            
        Returns:
            True if successful
        """
        resp = requests.post(
            GOOGLE_TOKEN,
            data={
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'code': code,
                'redirect_uri': redirect_uri,
                'grant_type': 'authorization_code'
            }
        )
        
        if resp.status_code == 200:
            data = resp.json()
            self.access_token = data['access_token']
            self.refresh_token = data.get('refresh_token')
            self.token_expires = datetime.now() + timedelta(seconds=data.get('expires_in', 3600))
            self._save_token()
            return True
        
        logger.error(f"Token exchange failed: {resp.text}")
        return False
    
    def is_authenticated(self) -> bool:
        """Check if we have a valid token."""
        if self._load_cached_token():
            return True
        return bool(self.access_token and self.token_expires and self.token_expires > datetime.now())
    
    def _headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        if not self.access_token:
            raise RuntimeError("Not authenticated.")
        return {'Authorization': f'Bearer {self.access_token}'}
    
    def list_sheets(self, limit: int = 50) -> List[GoogleSheet]:
        """
        List Google Sheets accessible to the user.
        
        Args:
            limit: Maximum sheets to return
            
        Returns:
            List of GoogleSheet objects
        """
        params = {
            'q': "mimeType='application/vnd.google-apps.spreadsheet'",
            'pageSize': min(limit, 100),
            'fields': 'files(id,name,modifiedTime,owners)'
        }
        
        resp = requests.get(DRIVE_API, headers=self._headers(), params=params)
        
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to list sheets: {resp.text}")
        
        sheets = []
        for item in resp.json().get('files', []):
            sheets.append(GoogleSheet(
                id=item['id'],
                name=item['name'],
                modified=datetime.fromisoformat(item['modifiedTime'].replace('Z', '+00:00')) if item.get('modifiedTime') else None,
                owner=item.get('owners', [{}])[0].get('displayName')
            ))
        
        return sheets
    
    def get_sheet_metadata(self, spreadsheet_id: str) -> Dict[str, Any]:
        """Get metadata about a spreadsheet including sheet names."""
        url = f"{SHEETS_API}/{spreadsheet_id}"
        resp = requests.get(url, headers=self._headers(), params={'fields': 'sheets.properties'})
        
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to get sheet metadata: {resp.text}")
        
        data = resp.json()
        return {
            'sheets': [s['properties']['title'] for s in data.get('sheets', [])]
        }
    
    def fetch_data(
        self,
        spreadsheet_id: str,
        sheet_name: Optional[str] = None,
        range_notation: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch data from a Google Sheet.
        
        Args:
            spreadsheet_id: The spreadsheet ID (from URL)
            sheet_name: Name of specific sheet (uses first sheet if None)
            range_notation: A1 notation range (e.g., 'A1:D100')
            
        Returns:
            DataFrame with sheet contents
        """
        # Build range string
        if sheet_name and range_notation:
            range_str = f"'{sheet_name}'!{range_notation}"
        elif sheet_name:
            range_str = f"'{sheet_name}'"
        elif range_notation:
            range_str = range_notation
        else:
            # Get first sheet name
            meta = self.get_sheet_metadata(spreadsheet_id)
            if meta['sheets']:
                range_str = f"'{meta['sheets'][0]}'"
            else:
                range_str = "A:ZZ"
        
        import urllib.parse
        encoded_range = urllib.parse.quote(range_str, safe='')
        
        url = f"{SHEETS_API}/{spreadsheet_id}/values/{encoded_range}"
        resp = requests.get(
            url, 
            headers=self._headers(),
            params={'valueRenderOption': 'UNFORMATTED_VALUE'}
        )
        
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch sheet data: {resp.text}")
        
        data = resp.json()
        values = data.get('values', [])
        
        if not values:
            return pd.DataFrame()
        
        # First row as headers
        headers = values[0]
        rows = values[1:]
        
        # Pad rows to match header length
        max_len = len(headers)
        rows = [row + [None] * (max_len - len(row)) for row in rows]
        
        return pd.DataFrame(rows, columns=headers)
    
    @staticmethod
    def extract_spreadsheet_id(url_or_id: str) -> str:
        """
        Extract spreadsheet ID from URL or return as-is if already an ID.
        
        Args:
            url_or_id: Either a full Google Sheets URL or just the ID
            
        Returns:
            The spreadsheet ID
        """
        import re
        
        # Already an ID (no slashes)
        if '/' not in url_or_id:
            return url_or_id
        
        # Extract from URL
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url_or_id)
        if match:
            return match.group(1)
        
        raise ValueError(f"Could not extract spreadsheet ID from: {url_or_id}")
    
    def get_available_series(self) -> List[DataSeries]:
        """Return empty - Sheets is file-based not series-based."""
        return []
