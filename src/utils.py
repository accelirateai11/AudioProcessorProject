# src/utils.py
import logging
import sys
import torch
from typing import Dict, Any, Optional, List
from contextvars import ContextVar

# Request ID context variable for tracing
request_id_var = ContextVar('request_id', default=None)

class RequestIDFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get() or 'no_request_id'
        return True

def setup_logging(name='audio_processor', level=logging.INFO):
    """Configure structured logging"""
    # Check if logger already has handlers to avoid duplicates
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        '%(asctime)s [%(request_id)s] [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Set level and add filter and handler
    logger.setLevel(level)
    
    # Create and add the request ID filter
    request_filter = RequestIDFilter()
    logger.addFilter(request_filter)
    
    # Add handler
    logger.addHandler(handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    # Reduce noise from other libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    return logger

# Leave only one instance of logger
logger = setup_logging()

def generate_request_id():
    """Generate a unique request ID"""
    import uuid
    return str(uuid.uuid4())

def set_request_context(request_id=None):
    """Set request context for current thread"""
    if request_id is None:
        request_id = generate_request_id()
    request_id_var.set(request_id)
    return request_id

def get_device_info() -> Dict[str, Any]:
    """Get device information for compute"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'device_count': 0,
        'device_name': 'CPU'
    }
    
    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        info['device_name'] = torch.cuda.get_device_name(0)
    
    return info

def get_device():
    """Detect and return best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

# Model caching
MODEL_CACHE = {}

def get_model(model_name, device=None):
    """Lazy-load model with caching"""
    if model_name not in MODEL_CACHE:
        # Implementation details here
        pass
    return MODEL_CACHE[model_name]

class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        logger.info(f"{self.name} completed in {self.elapsed_ms:.2f}ms")

# Import at the end to avoid circular imports
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import time

class TranscriptionConfig(BaseModel):
    """Configuration for transcription request"""
    language_hint: Optional[str] = Field(None, description="Language code (e.g., 'en')")
    enable_separation: bool = Field(True, description="Enable vocal separation")
    diarize: bool = Field(False, description="Enable speaker diarization")
    model_size: str = Field('small', description="Whisper model size")
    target_sr: int = Field(16000, description="Target sample rate")
    apply_vad: bool = Field(True, description="Apply Voice Activity Detection")
    vad_threshold: float = Field(0.01, description="VAD energy threshold (0-1)")

class AudioSegment(BaseModel):
    """Transcription segment"""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text")
    speaker: Optional[str] = Field(None, description="Speaker label if diarization enabled")

class TranscriptionResponse(BaseModel):
    """Response from transcription endpoint"""
    request_id: str = Field(..., description="Unique request ID")
    duration_sec: float = Field(..., description="Audio duration in seconds")
    sample_rate: int = Field(..., description="Sample rate used")
    pipeline: Dict[str, Any] = Field(..., description="Pipeline configuration used")
    segments: List[AudioSegment] = Field(..., description="Transcription segments")
    text: str = Field(..., description="Full transcription text")
    language: str = Field(..., description="Detected language")
    timings_ms: Dict[str, int] = Field(..., description="Processing timings in milliseconds")

class ErrorResponse(BaseModel):
    """Error response"""
    request_id: str = Field(..., description="Request ID")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")