"""
Unit tests for ETL pipeline
"""
import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from src.etl.extractor import DataExtractor
from src.etl.validator import DataValidator
from src.etl.cleaner import DataCleaner

class TestDataExtractor:
    """Test DataExtractor class"""
    
    def setup_method(self):
        """Setup test data"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create sample CSV file
        sample_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        sample_data.to_csv(self.temp_path / "sample.csv", index=False)
        
        self.extractor = DataExtractor(str(self.temp_path))
    
    def teardown_method(self):
        """Cleanup test data"""
        shutil.rmtree(self.temp_dir)
    
    def test_extract_csv_files(self):
        """Test CSV file extraction"""
        df = self.extractor.extract_csv_files()
        
        assert len(df) == 3
        assert 'source_file' in df.columns
        assert df['source_file'].iloc[0] == 'sample.csv'
    
    def test_get_data_info(self):
        """Test data info generation"""
        df = self.extractor.extract_csv_files()
        info = self.extractor.get_data_info(df)
        
        assert info['shape'] == (3, 5)  # 4 original + 1 source_file
        assert 'columns' in info
        assert 'missing_values' in info

class TestDataValidator:
    """Test DataValidator class"""
    
    def setup_method(self):
        """Setup test data"""
        self.config = {
            'min_rows': 2,
            'max_missing_percentage': 0.3
        }
        self.validator = DataValidator(self.config)
        
        self.valid_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
        
        self.invalid_df = pd.DataFrame({
            'id': [1],
            'name': ['Alice']
        })
    
    def test_validate_structure_valid(self):
        """Test structure validation with valid data"""
        result = self.validator.validate_structure(self.valid_df)
        assert result is True
    
    def test_validate_structure_invalid(self):
        """Test structure validation with invalid data"""
        with pytest.raises(Exception):
            self.validator.validate_structure(self.invalid_df)
    
    def test_validate_data_quality(self):
        """Test data quality validation"""
        result = self.validator.validate_data_quality(self.valid_df)
        assert result is True

class TestDataCleaner:
    """Test DataCleaner class"""
    
    def setup_method(self):
        """Setup test data"""
        self.config = {
            'drop_duplicates': True,
            'handle_missing': 'drop',
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5
        }
        self.cleaner = DataCleaner(self.config)
        
        # Create data with duplicates and missing values
        self.dirty_df = pd.DataFrame({
            'id': [1, 2, 2, 3, 4],  # Duplicate row
            'name': ['Alice', 'Bob', 'Bob', 'Charlie', None],  # Missing value
            'age': [25, 30, 30, 35, 40],
            'salary': [50000, 60000, 60000, 70000, 100000]  # Potential outlier
        })
    
    def test_remove_duplicates(self):
        """Test duplicate removal"""
        cleaned_df = self.cleaner.remove_duplicates(self.dirty_df)
        assert len(cleaned_df) == 4  # One duplicate removed
    
    def test_handle_missing_values_drop(self):
        """Test missing value handling with drop method"""
        cleaned_df = self.cleaner.handle_missing_values(self.dirty_df)
        assert cleaned_df.isnull().sum().sum() == 0
    
    def test_handle_missing_values_fill_mean(self):
        """Test missing value handling with mean fill"""
        self.cleaner.config['handle_missing'] = 'fill_mean'
        cleaned_df = self.cleaner.handle_missing_values(self.dirty_df)
        assert cleaned_df.isnull().sum().sum() == 0
    
    def test_remove_outliers(self):
        """Test outlier removal"""
        cleaned_df = self.cleaner.remove_outliers(self.dirty_df)
        # Should remove or keep data based on IQR method
        assert len(cleaned_df) <= len(self.dirty_df)
    
    def test_standardize_data_types(self):
        """Test data type standardization"""
        cleaned_df = self.cleaner.standardize_data_types(self.dirty_df)
        assert cleaned_df is not None