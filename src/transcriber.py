# src/transcriber.py
import logging
import warnings
import torch
import numpy as np
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class Transcriber:
    """Handle audio transcription using Whisper"""
    
    MODEL_SIZES = {
        'tiny': 'openai/whisper-tiny',
        'base': 'openai/whisper-base', 
        'small': 'openai/whisper-small',
        'medium': 'openai/whisper-medium',
        'large': 'openai/whisper-large-v3'
    }
    
    def __init__(self, 
                 device: str = 'cpu',
                 default_model: str = 'small',
                 enable_diarization: bool = False):
        """
        Initialize transcriber
        
        Args:
            device: Device to use (cpu/cuda)
            default_model: Default model size
            enable_diarization: Enable speaker diarization
        """
        self.device = device
        self.default_model = default_model
        self.enable_diarization = enable_diarization
        self.models = {}
        self.diarization_pipeline = None
        
        # Preload default model
        self._load_model(default_model)
        
        if enable_diarization:
            self._init_diarization()
    
    def _load_model(self, model_size: str) -> Any:
        """Load Whisper model"""
        if model_size in self.models:
            return self.models[model_size]
        
        try:
            import whisper
            
            logger.info(f"Loading Whisper {model_size} model...")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = whisper.load_model(model_size, device=self.device)
            
            self.models[model_size] = model
            logger.info(f"Whisper {model_size} loaded on {self.device}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            
            # Try transformers fallback
            try:
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                
                model_name = self.MODEL_SIZES.get(model_size, 'openai/whisper-small')
                
                processor = WhisperProcessor.from_pretrained(model_name)
                model = WhisperForConditionalGeneration.from_pretrained(model_name)
                model.to(self.device)
                
                self.models[model_size] = {'model': model, 'processor': processor, 'transformers': True}
                logger.info(f"Loaded Whisper via transformers: {model_name}")
                
                return self.models[model_size]
                
            except Exception as e2:
                logger.error(f"Failed to load via transformers: {str(e2)}")
                raise RuntimeError("Could not load transcription model")
    
    def _init_diarization(self):
        """Initialize speaker diarization"""
        try:
            from pyannote.audio import Pipeline
            
            # This would require authentication with Hugging Face
            # For demo purposes, we'll use a mock implementation
            logger.warning("Diarization requires pyannote authentication - using mock")
            self.diarization_pipeline = None
            
        except ImportError:
            logger.warning("pyannote not available for diarization")
            self.diarization_pipeline = None
    
    def transcribe(self,
                   audio: np.ndarray,
                   sample_rate: int,
                   language: Optional[str] = None,
                   model_size: str = None,
                   chunk_length: int = 30,
                   apply_vad: bool = True,
                   vad_threshold: float = 0.01) -> Dict[str, Any]:
        """
        Transcribe audio
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            language: Language code (e.g. 'en')
            model_size: Model size (tiny, base, small, medium, large)
            chunk_length: Maximum chunk length in seconds
            apply_vad: Whether to apply VAD
            vad_threshold: VAD energy threshold
            
        Returns:
            Transcription result dictionary
        """
        model_size = model_size or self.default_model
        
        # Apply VAD if requested
        if apply_vad:
            from src.audio_processor import AudioProcessor
            processor = AudioProcessor()
            audio = processor.apply_vad(audio, sample_rate, threshold=vad_threshold)
            logger.info(f"Applied VAD, new audio length: {len(audio)/sample_rate:.2f}s")
        
        # Handle long audio files
        audio_duration = len(audio) / sample_rate
        if audio_duration > chunk_length:
            logger.info(f"Audio duration ({audio_duration:.2f}s) exceeds chunk length ({chunk_length}s), using chunked processing")
            return self._transcribe_long_audio(audio, sample_rate, language, model_size)
        
        # Load model
        model_dict = self._load_model(model_size)
        
        # Run transcription
        result = self._transcribe_transformers(model_dict, audio, sample_rate, language)
        
        # Apply diarization if enabled
        if self.enable_diarization and self.diarization_pipeline:
            try:
                result = self._apply_diarization(result, audio, sample_rate)
            except Exception as e:
                logger.error(f"Diarization failed: {str(e)}")
    
        return result

    def _transcribe_long_audio(self, 
                              audio: np.ndarray, 
                              sample_rate: int, 
                              language: Optional[str],
                              model_size: str) -> Dict[str, Any]:
        """Process long audio using chunking and stitching"""
        from src.audio_processor import AudioProcessor
        processor = AudioProcessor()
        
        # Process with chunking
        return processor.process_long_audio(
            audio, 
            sample_rate, 
            self,
            max_duration=30.0,
            overlap=2.0,
            language=language,
            model_size=model_size
        )
    
    def _transcribe_transformers(self, model, audio, sample_rate, language):
        """
        Transcribe using Transformers implementation
        
        Args:
            model: Whisper model object (not a dictionary)
            audio: Audio array
            sample_rate: Sample rate
            language: Language code
            
        Returns:
            Transcription result
        """
        try:
            # If model is a dictionary (old implementation), extract model
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
            
            # Prepare options
            options = {}
            if language:
                options["language"] = language
            
            # Run transcription
            result = model.transcribe(audio, **options)
            
            # Format result
            return {
                "text": result["text"],
                "segments": result["segments"],
                "language": result.get("language", "en")
            }
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise
    
    def _add_diarization(self,
                        segments: List[Dict],
                        audio: np.ndarray,
                        sample_rate: int) -> List[Dict]:
        """Add speaker labels to segments"""
        # Mock implementation - real implementation would use pyannote
        for i, segment in enumerate(segments):
            # Alternate between speakers for demo
            segment['speaker'] = f"SPEAKER_{i % 2}"
        
        return segments
    
    def cleanup(self):
        """Clean up resources"""
        for model in self.models.values():
            if isinstance(model, dict) and 'model' in model:
                del model['model']
            else:
                del model
        
        self.models.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
def format_response(request_id, audio_info, result, timings):
    """Format standardized API response"""
    return {
        "request_id": request_id,
        "duration_sec": audio_info["duration"],
        "sample_rate": audio_info["sample_rate"],
        "pipeline": {
            "separation": {
                "enabled": audio_info["separation_enabled"],
                "method": audio_info["separation_method"]
            },
            "transcription": {
                "model": audio_info["model_name"]
            }
        },
        "segments": result["segments"],
        "text": result["text"],
        "language": result["language"],
        "timings_ms": {
            "load": timings["load"],
            "separation": timings["separation"],
            "transcription": timings["transcription"],
            "total": timings["total"]
        }
    }