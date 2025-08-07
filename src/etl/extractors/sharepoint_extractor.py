"""
SharePoint extractor
"""

import tempfile
from typing import Iterator, Dict, Any
import pandas as pd
from pathlib import Path
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File
from .base import BaseExtractor
from .pdf_extractor import PDFExtractor
from .docx_extractor import DOCXExtractor


class SharePointExtractor(BaseExtractor):
    """Extractor for SharePoint documents"""
    
    def validate_config(self) -> bool:
        """Validate SharePoint extractor configuration"""
        required_keys = ['site_url', 'username', 'password', 'folder_path']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        return True
    
    def extract(self) -> Iterator[pd.DataFrame]:
        """
        Extract documents from SharePoint
        
        Yields:
            DataFrame with extracted document content
        """
        self.validate_config()
        
        site_url = self.config['site_url']
        username = self.config['username']
        password = self.config['password']
        folder_path = self.config['folder_path']
        file_patterns = self.config.get('file_patterns', ['*'])
        
        self.logger.info(f"Connecting to SharePoint: {site_url}")
        
        try:
            # Authenticate with SharePoint
            ctx_auth = AuthenticationContext(site_url)
            if ctx_auth.acquire_token_for_user(username, password):
                ctx = ClientContext(site_url, ctx_auth)
                
                # Get files from folder
                folder = ctx.web.get_folder_by_server_relative_url(folder_path)
                files = folder.files
                ctx.load(files)
                ctx.execute_query()
                
                for file in files:
                    file_name = file.properties['Name']
                    
                    # Check if file matches patterns
                    if self._matches_patterns(file_name, file_patterns):
                        self.logger.info(f"Processing SharePoint file: {file_name}")
                        
                        try:
                            # Download file to temporary location
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as temp_file:
                                file_content = File.open_binary(ctx, file.serverRelativeUrl)
                                temp_file.write(file_content.content)
                                temp_file_path = temp_file.name
                            
                            # Extract content based on file type
                            df = self._extract_file_content(temp_file_path, file_name)
                            if df is not None:
                                # Add SharePoint metadata
                                df['_sharepoint_url'] = file.serverRelativeUrl
                                df['_sharepoint_site'] = site_url
                                yield df
                            
                            # Clean up temporary file
                            Path(temp_file_path).unlink()
                            
                        except Exception as e:
                            self.logger.error(f"Error processing SharePoint file {file_name}: {e}")
                            continue
            else:
                raise ValueError("SharePoint authentication failed")
                
        except Exception as e:
            self.logger.error(f"Error connecting to SharePoint: {e}")
            raise
    
    def _matches_patterns(self, file_name: str, patterns: list) -> bool:
        """Check if file name matches any of the specified patterns"""
        import fnmatch
        return any(fnmatch.fnmatch(file_name.lower(), pattern.lower()) for pattern in patterns)
    
    def _extract_file_content(self, file_path: str, original_name: str) -> pd.DataFrame:
        """
        Extract content from file based on its type
        
        Args:
            file_path: Path to temporary file
            original_name: Original file name
            
        Returns:
            DataFrame with extracted content or None
        """
        file_ext = Path(original_name).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                # Use PDF extractor
                pdf_config = {'paths': [file_path]}
                pdf_extractor = PDFExtractor(pdf_config)
                for df in pdf_extractor.extract():
                    df['file_name'] = original_name
                    return df
                    
            elif file_ext == '.docx':
                # Use DOCX extractor
                docx_config = {'paths': [file_path]}
                docx_extractor = DOCXExtractor(docx_config)
                for df in docx_extractor.extract():
                    df['file_name'] = original_name
                    return df
                    
            elif file_ext in ['.xlsx', '.xls']:
                # Handle Excel files
                df = pd.read_excel(file_path)
                df['file_name'] = original_name
                df['_source_type'] = 'excel'
                return df
                
            else:
                self.logger.warning(f"Unsupported file type: {file_ext}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting content from {original_name}: {e}")
            return None