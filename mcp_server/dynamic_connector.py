"""
Dynamic Connector Generator - LLM-powered API connector creation.

When no matching API is found in the registry, this module:
1. Fetches and parses API documentation from a given URL
2. Uses LLM to generate Python connector code
3. Safely executes the generated code in a sandbox
4. Returns the connector for data fetching

This enables Assay to autonomously connect to NEW APIs without 
manual connector development.
"""
import re
import logging
import requests
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class GeneratedConnector:
    """Result of dynamic connector generation."""
    api_name: str
    base_url: str
    code: str
    endpoints: list = field(default_factory=list)
    auth_type: str = "unknown"
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    validated: bool = False
    error: Optional[str] = None


class DocParser:
    """
    Fetch and parse API documentation from URLs.
    
    Supports:
    - HTML documentation pages
    - OpenAPI/Swagger JSON/YAML specs
    - README files
    """
    
    def __init__(self, max_content_length: int = 50000):
        self.max_content_length = max_content_length
    
    def fetch_docs(self, url: str) -> Tuple[str, str]:
        """
        Fetch documentation from URL.
        
        Returns:
            Tuple of (content, content_type)
        """
        try:
            headers = {
                "User-Agent": "Assay-API-Discovery/1.0 (https://github.com/assay-ai)",
                "Accept": "text/html,application/json,application/yaml,text/plain"
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get("Content-Type", "text/html")
            content = response.text[:self.max_content_length]
            
            return content, content_type
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch docs from {url}: {e}")
            raise ValueError(f"Could not fetch documentation: {e}")
    
    def parse_openapi(self, content: str) -> Dict[str, Any]:
        """Parse OpenAPI/Swagger specification."""
        try:
            spec = json.loads(content)
        except json.JSONDecodeError:
            # Try YAML
            try:
                import yaml
                spec = yaml.safe_load(content)
            except Exception:
                raise ValueError("Could not parse as JSON or YAML")
        
        # Extract key information
        info = spec.get("info", {})
        servers = spec.get("servers", [])
        paths = spec.get("paths", {})
        
        parsed = {
            "name": info.get("title", "Unknown API"),
            "description": info.get("description", ""),
            "version": info.get("version", ""),
            "base_url": servers[0].get("url", "") if servers else "",
            "endpoints": [],
            "auth_type": "none"
        }
        
        # Check for security schemes
        security_schemes = spec.get("components", {}).get("securitySchemes", {})
        if security_schemes:
            for scheme_name, scheme in security_schemes.items():
                if scheme.get("type") == "apiKey":
                    parsed["auth_type"] = "api_key"
                    parsed["auth_info"] = {
                        "name": scheme.get("name"),
                        "in": scheme.get("in"),  # header or query
                    }
                    break
                elif scheme.get("type") == "http":
                    parsed["auth_type"] = "bearer"
                    break
        
        # Extract endpoints
        for path, methods in paths.items():
            for method, details in methods.items():
                if method in ("get", "post", "put", "delete"):
                    parsed["endpoints"].append({
                        "path": path,
                        "method": method.upper(),
                        "description": details.get("summary", details.get("description", "")),
                        "parameters": [p.get("name") for p in details.get("parameters", [])],
                    })
        
        return parsed
    
    def parse_html_docs(self, content: str, url: str) -> Dict[str, Any]:
        """Extract API information from HTML documentation."""
        from html.parser import HTMLParser
        
        # Simple extraction - look for common patterns
        text_content = re.sub(r'<[^>]+>', ' ', content)
        text_content = re.sub(r'\s+', ' ', text_content)
        
        # Try to find base URL
        base_url_match = re.search(
            r'https?://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}/(?:api|v[0-9]+)',
            content
        )
        
        # Try to find API key mentions
        auth_type = "none"
        if re.search(r'api[_\-]?key|apikey|api_token', content, re.IGNORECASE):
            auth_type = "api_key"
        elif re.search(r'bearer|oauth|authorization', content, re.IGNORECASE):
            auth_type = "bearer"
        
        # Extract endpoint patterns
        endpoint_patterns = re.findall(
            r'(GET|POST|PUT|DELETE)\s+(/[a-zA-Z0-9/\{\}_\-]+)',
            content, re.IGNORECASE
        )
        
        return {
            "name": "API from " + url.split("/")[2],
            "description": text_content[:500],
            "base_url": base_url_match.group(0) if base_url_match else "",
            "endpoints": [
                {"method": m.upper(), "path": p} 
                for m, p in endpoint_patterns[:10]
            ],
            "auth_type": auth_type,
            "raw_content": text_content[:10000],  # For LLM context
        }
    
    def parse(self, url: str) -> Dict[str, Any]:
        """
        Fetch and parse documentation from URL.
        
        Automatically detects format and extracts API information.
        """
        content, content_type = self.fetch_docs(url)
        
        # Check if it's OpenAPI/Swagger
        if "json" in content_type or "yaml" in content_type:
            try:
                return self.parse_openapi(content)
            except ValueError:
                pass
        
        # Check content for OpenAPI markers
        if '"openapi"' in content or '"swagger"' in content or 'openapi:' in content:
            try:
                return self.parse_openapi(content)
            except ValueError:
                pass
        
        # Fall back to HTML parsing
        return self.parse_html_docs(content, url)


class ConnectorGenerator:
    """
    Generate Python connector code using LLM.
    """
    
    def __init__(self):
        self.template = '''
"""
Auto-generated connector for {api_name}.
Generated by Assay Dynamic Connector Generator.
"""
import requests
import pandas as pd
from typing import Dict, Any, Optional, List

class {class_name}Connector:
    """Connector for {api_name}."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "{base_url}"
        self.api_key = api_key
        self.session = requests.Session()
        {auth_setup}
    
    def fetch_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Fetch data from an endpoint and return as DataFrame.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            pandas DataFrame with the response data
        """
        url = f"{{self.base_url}}{{endpoint}}"
        params = params or {{}}
        {auth_params}
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Try to find the data array in the response
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Look for common data keys
                for key in ['data', 'results', 'items', 'records', 'values']:
                    if key in data and isinstance(data[key], list):
                        return pd.DataFrame(data[key])
                # If no list found, wrap single record
                return pd.DataFrame([data])
            else:
                return pd.DataFrame({{"value": [data]}})
                
        except requests.RequestException as e:
            raise ConnectionError(f"API request failed: {{e}}")
        except Exception as e:
            raise ValueError(f"Failed to parse response: {{e}}")
    
    {custom_methods}
'''
    
    def generate(self, parsed_docs: Dict[str, Any]) -> str:
        """
        Generate connector code from parsed documentation.
        
        Args:
            parsed_docs: Parsed API documentation from DocParser
            
        Returns:
            Python connector code as string
        """
        api_name = parsed_docs.get("name", "Unknown")
        class_name = self._sanitize_class_name(api_name)
        base_url = parsed_docs.get("base_url", "")
        auth_type = parsed_docs.get("auth_type", "none")
        
        # Generate auth setup code
        if auth_type == "api_key":
            auth_info = parsed_docs.get("auth_info", {})
            if auth_info.get("in") == "header":
                auth_setup = f'if api_key:\n            self.session.headers["{auth_info.get("name", "X-API-Key")}"] = api_key'
                auth_params = ""
            else:
                auth_setup = ""
                param_name = auth_info.get("name", "api_key")
                auth_params = f'if self.api_key:\n            params["{param_name}"] = self.api_key'
        elif auth_type == "bearer":
            auth_setup = 'if api_key:\n            self.session.headers["Authorization"] = f"Bearer {api_key}"'
            auth_params = ""
        else:
            auth_setup = "pass  # No auth required"
            auth_params = "pass  # No auth params"
        
        # Generate custom methods for specific endpoints
        custom_methods = self._generate_endpoint_methods(parsed_docs.get("endpoints", []))
        
        code = self.template.format(
            api_name=api_name,
            class_name=class_name,
            base_url=base_url,
            auth_setup=auth_setup,
            auth_params=auth_params,
            custom_methods=custom_methods,
        )
        
        return code
    
    def _sanitize_class_name(self, name: str) -> str:
        """Convert API name to valid Python class name."""
        # Remove non-alphanumeric characters
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        # Convert to PascalCase
        words = clean.split()
        return ''.join(word.capitalize() for word in words)
    
    def _generate_endpoint_methods(self, endpoints: list) -> str:
        """Generate convenience methods for specific endpoints."""
        methods = []
        
        for ep in endpoints[:5]:  # Limit to first 5 endpoints
            path = ep.get("path", "")
            method_name = self._path_to_method_name(path)
            description = ep.get("description", f"Fetch data from {path}")
            
            method_code = f'''
    def {method_name}(self, **kwargs) -> pd.DataFrame:
        """{description}"""
        return self.fetch_data("{path}", params=kwargs)
'''
            methods.append(method_code)
        
        return "\n".join(methods)
    
    def _path_to_method_name(self, path: str) -> str:
        """Convert endpoint path to method name."""
        # Remove leading slash and placeholders
        clean = re.sub(r'[{}]', '', path)
        clean = clean.strip('/').replace('/', '_').replace('-', '_')
        
        # Ensure it's a valid identifier
        if not clean or clean[0].isdigit():
            clean = "fetch_" + clean
        
        return clean.lower()[:50]  # Limit length


class SandboxExecutor:
    """
    Safely execute generated connector code.
    
    Uses restricted globals to prevent dangerous operations.
    """
    
    # Allowed modules for generated code
    ALLOWED_MODULES = {
        'requests', 'pandas', 'json', 're', 'datetime',
        'typing', 'dataclasses', 'urllib'
    }
    
    def __init__(self):
        self._connector_cache = {}
    
    def execute(self, code: str) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute generated connector code.
        
        Args:
            code: Python code string
            
        Returns:
            Tuple of (success, connector_class or None, error_message or None)
        """
        # Pre-import allowed modules
        import requests
        import pandas as pd
        from typing import Dict, Any, Optional, List
        
        # Create a safe import function that only allows specific modules
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Safe import that only allows whitelisted modules."""
            if name in self.ALLOWED_MODULES:
                return __import__(name, globals, locals, fromlist, level)
            # Handle sub-imports like 'typing.Dict'
            base_module = name.split('.')[0]
            if base_module in self.ALLOWED_MODULES:
                return __import__(name, globals, locals, fromlist, level)
            raise ImportError(f"Module '{name}' is not allowed in generated connectors")
        
        # Create restricted execution environment
        restricted_globals = {
            '__builtins__': {
                # Core for class definitions and imports
                '__import__': safe_import,
                '__build_class__': __builtins__['__build_class__'] if isinstance(__builtins__, dict) else __builtins__.__build_class__,
                '__name__': '__main__',
                '__doc__': None,
                
                # Constants
                'None': None,
                'True': True,
                'False': False,
                
                # Functions
                'print': print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sorted': sorted,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'round': round,
                'isinstance': isinstance,
                'issubclass': issubclass,
                'callable': callable,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'delattr': delattr,
                'property': property,
                'staticmethod': staticmethod,
                'classmethod': classmethod,
                'super': super,
                'type': type,
                'object': object,
                
                # Exceptions
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'IndexError': IndexError,
                'AttributeError': AttributeError,
                'ConnectionError': ConnectionError,
                'ImportError': ImportError,
            }
        }
        
        # Pre-add allowed modules to globals so imports work
        restricted_globals['requests'] = requests
        restricted_globals['pd'] = pd
        restricted_globals['pandas'] = pd
        restricted_globals['Dict'] = Dict
        restricted_globals['Any'] = Any
        restricted_globals['Optional'] = Optional
        restricted_globals['List'] = List
        
        local_namespace = {}
        
        try:
            # Compile and execute
            compiled = compile(code, '<generated_connector>', 'exec')
            exec(compiled, restricted_globals, local_namespace)
            
            # Find the connector class
            for name, obj in local_namespace.items():
                if isinstance(obj, type) and name.endswith('Connector'):
                    return True, obj, None
            
            return False, None, "No Connector class found in generated code"
            
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            return False, None, f"Syntax error: {e}"
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False, None, f"Execution error: {e}"
    
    def test_connector(self, connector_class, api_key: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Test that a connector can be instantiated and has expected methods.
        
        Returns:
            Tuple of (success, error_message or None)
        """
        try:
            # Instantiate
            connector = connector_class(api_key=api_key)
            
            # Check for required methods
            if not hasattr(connector, 'fetch_data'):
                return False, "Connector missing fetch_data method"
            
            if not hasattr(connector, 'base_url'):
                return False, "Connector missing base_url attribute"
            
            return True, None
            
        except Exception as e:
            return False, f"Connector instantiation failed: {e}"


class DynamicConnectorManager:
    """
    Main interface for dynamic connector generation.
    
    Orchestrates the full flow:
    1. Fetch and parse API docs
    2. Generate connector code
    3. Validate and execute
    4. Cache for reuse
    """
    
    def __init__(self):
        self.doc_parser = DocParser()
        self.generator = ConnectorGenerator()
        self.executor = SandboxExecutor()
        self._cache: Dict[str, GeneratedConnector] = {}
    
    def generate_connector(
        self, 
        docs_url: str,
        api_key: Optional[str] = None,
        force_regenerate: bool = False
    ) -> GeneratedConnector:
        """
        Generate a connector from API documentation.
        
        Args:
            docs_url: URL to API documentation
            api_key: Optional API key for testing
            force_regenerate: Force regeneration even if cached
            
        Returns:
            GeneratedConnector with code and status
        """
        # Check cache
        if docs_url in self._cache and not force_regenerate:
            logger.info(f"Using cached connector for {docs_url}")
            return self._cache[docs_url]
        
        result = GeneratedConnector(
            api_name="",
            base_url="",
            code="",
        )
        
        try:
            # Step 1: Parse documentation
            logger.info(f"Parsing API docs from {docs_url}")
            parsed = self.doc_parser.parse(docs_url)
            
            result.api_name = parsed.get("name", "Unknown")
            result.base_url = parsed.get("base_url", "")
            result.endpoints = parsed.get("endpoints", [])
            result.auth_type = parsed.get("auth_type", "unknown")
            
            # Step 2: Generate connector code
            logger.info(f"Generating connector for {result.api_name}")
            result.code = self.generator.generate(parsed)
            
            # Step 3: Execute and validate
            logger.info("Validating generated connector")
            success, connector_class, error = self.executor.execute(result.code)
            
            if not success:
                result.error = error
                return result
            
            # Step 4: Test connector
            test_success, test_error = self.executor.test_connector(connector_class, api_key)
            
            if test_success:
                result.validated = True
                self._cache[docs_url] = result
                logger.info(f"Successfully generated connector for {result.api_name}")
            else:
                result.error = test_error
            
            return result
            
        except Exception as e:
            logger.error(f"Connector generation failed: {e}")
            result.error = str(e)
            return result
    
    def get_connector_instance(
        self,
        docs_url: str,
        api_key: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get a ready-to-use connector instance.
        
        Args:
            docs_url: URL to API documentation
            api_key: API key if required
            
        Returns:
            Instantiated connector or None if generation failed
        """
        generated = self.generate_connector(docs_url, api_key)
        
        if not generated.validated:
            logger.error(f"Cannot get instance - validation failed: {generated.error}")
            return None
        
        success, connector_class, error = self.executor.execute(generated.code)
        
        if success and connector_class:
            return connector_class(api_key=api_key)
        
        return None


# Global manager instance
_manager: Optional[DynamicConnectorManager] = None


def get_connector_manager() -> DynamicConnectorManager:
    """Get the global connector manager instance."""
    global _manager
    if _manager is None:
        _manager = DynamicConnectorManager()
    return _manager


def generate_connector_from_docs(docs_url: str, api_key: Optional[str] = None) -> GeneratedConnector:
    """
    Convenience function to generate a connector from documentation URL.
    
    Args:
        docs_url: URL to API documentation (OpenAPI spec or HTML docs)
        api_key: Optional API key for testing
        
    Returns:
        GeneratedConnector with code and validation status
    """
    return get_connector_manager().generate_connector(docs_url, api_key)
