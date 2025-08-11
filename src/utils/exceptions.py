"""
Custom exceptions for the data science pipeline
"""

class PipelineException(Exception):
    """Base exception for pipeline errors"""
    pass

class ETLException(PipelineException):
    """Exception raised during ETL operations"""
    pass

class DataValidationException(ETLException):
    """Exception raised during data validation"""
    pass

class MLException(PipelineException):
    """Exception raised during ML operations"""
    pass

class ModelTrainingException(MLException):
    """Exception raised during model training"""
    pass

class ModelLoadException(MLException):
    """Exception raised when loading models"""
    pass

class APIException(PipelineException):
    """Exception raised in API operations"""
    pass

class AuthenticationException(APIException):
    """Exception raised during authentication"""
    pass