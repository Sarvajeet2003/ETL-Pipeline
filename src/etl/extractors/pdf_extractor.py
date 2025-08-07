"""
PDF document extractor
"""

import glob
from typing import Iterator, Dict, Any
import pandas as pd
from pathlib import Path
import PyPDF2
from .base import BaseExtractor


class PDFExtractor(BaseExtractor):
    """Extractor for PDF documents"""
    
    def validate_config(self) -> bool:
        """Validate PDF extractor configuration"""
        required_keys = ['paths']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        return True
    
    def extract(self) -> Iterator[pd.DataFrame]:
        """
        Extract text content from PDF files
        
        Yields:
            DataFrame with extracted text content
        """
        self.validate_config()
        
        paths = self.config['paths']
        
        for path_pattern in paths:
            files = glob.glob(path_pattern)
            
            if not files:
                self.logger.warning(f"No PDF files found matching pattern: {path_pattern}")
                continue
            
            for file_path in files:
                self.logger.info(f"Extracting text from PDF: {file_path}")
                
                try:
                    text_content = self._extract_pdf_text(file_path)
                    
                    if text_content:
                        # Create DataFrame with extracted content
                        df = pd.DataFrame({
                            'content': [text_content],
                            'file_name': [Path(file_path).name],
                            'file_path': [file_path],
                            '_source_type': ['pdf']
                        })
                        yield df
                    else:
                        self.logger.warning(f"No text extracted from: {file_path}")
                        
                except Exception as e:
                    self.logger.error(f"Error extracting from PDF {file_path}: {e}")
                    continue
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        text_content = ""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        self.logger.warning(f"Error extracting page {page_num + 1} from {file_path}: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error opening PDF file {file_path}: {e}")
            raise
        
        return text_content.strip()