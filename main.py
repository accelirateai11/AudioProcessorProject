# main.py
import os
import uuid
import time
import json
import tempfile
import traceback
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.audio_processor import AudioProcessor
from src.vocal_separator import VocalSeparator
from src.transcriber import Transcriber
from src.utils import setup_logging, get_device_info
from src.utils import TranscriptionConfig, TranscriptionResponse, AudioSegment

# Setup logging
logger = setup_logging()

# Global instances for model management
audio_processor = None
separator = None
transcriber = None

# Update the lifespan function to create separator with output_dir
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle - load on startup, cleanup on shutdown"""
    global audio_processor, separator, transcriber
    
    logger.info("Starting up - initializing models...")
    start_time = time.time()
    
    # Initialize components
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")
    
    # Create default output directory for stems
    stems_dir = os.path.join(os.getcwd(), "stems")
    os.makedirs(stems_dir, exist_ok=True)
    
    audio_processor = AudioProcessor()
    separator = VocalSeparator(device=device_info['device'], output_dir=stems_dir)
    transcriber = Transcriber(device=device_info['device'])
    
    load_time = (time.time() - start_time) * 1000
    logger.info(f"Models loaded in {load_time:.2f}ms")
    
    yield
    
    # Cleanup
    logger.info("Shutting down - cleaning up models...")
    if separator:
        separator.cleanup()
    if transcriber:
        transcriber.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Audio Transcription Service",
    description="Microservice for audio transcription with vocal separation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Audio Transcription Service is running."}

# Update the transcribe endpoint to use file_name
@app.post("/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    config: Optional[str] = Form(None)
):
    """
    Transcribe audio file with optional vocal separation
    
    Args:
        file: Audio file (wav/mp3/m4a/flac/ogg)
        config: Optional JSON configuration
    
    Returns:
        Transcription result with metadata
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    timings = {}
    
    logger.info(f"Request {request_id}: Starting transcription for {file.filename}")
    
    try:
        # Validate file format
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in audio_processor.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_ext}. Supported formats: {', '.join(audio_processor.SUPPORTED_FORMATS)}"
            )
        
        # Parse configuration
        if config:
            try:
                config_dict = json.loads(config)
                config_obj = TranscriptionConfig(**config_dict)
            except (json.JSONDecodeError, Exception) as e:
                raise HTTPException(status_code=400, detail=f"Invalid config JSON: {str(e)}")
        else:
            config_obj = TranscriptionConfig()
        
        # Validate file size (100MB limit)
        file_size = 0
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            raise HTTPException(status_code=413, detail="File too large (max 100MB)")
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Create request-specific stems directory
        stems_dir = os.path.join(os.getcwd(), "stems", request_id)
        os.makedirs(stems_dir, exist_ok=True)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Load and process audio
            load_start = time.time()
            try:
                audio_data, sample_rate, duration = audio_processor.load_audio(
                    tmp_path, 
                    target_sr=config_obj.target_sr
                )
            except ValueError as format_error:
                raise HTTPException(status_code=400, detail=str(format_error))
            except RuntimeError as load_error:
                raise HTTPException(status_code=422, detail=f"Failed to decode audio: {str(load_error)}")
                
            timings['load'] = int((time.time() - load_start) * 1000)
            
            logger.info(f"Request {request_id}: Loaded audio - duration: {duration:.2f}s, sr: {sample_rate}")
            
            # Apply VAD if configured (default enabled)
            vad_enabled = config_obj.dict().get("apply_vad", True)
            if vad_enabled:
                vad_start = time.time()
                try:
                    audio_data = audio_processor.apply_vad(
                        audio_data, 
                        sample_rate,
                        threshold=config_obj.dict().get("vad_threshold", 0.01)
                    )
                    timings['vad'] = int((time.time() - vad_start) * 1000)
                except Exception as e:
                    logger.warning(f"Request {request_id}: VAD failed: {str(e)}, using original audio")
            
            # Vocal separation
            separation_enabled = False
            separation_method = None
            stems_location = None
            
            if config_obj.enable_separation and separator:
                sep_start = time.time()
                try:
                    separated_audio = separator.separate(
                        audio_data, 
                        sample_rate,
                        file_name=file.filename,
                        output_dir=stems_dir
                    )
                    if separated_audio is not None:
                        audio_data = separated_audio
                        separation_enabled = True
                        separation_method = separator.method_name
                        stems_location = stems_dir
                        logger.info(f"Request {request_id}: Vocal separation successful, stems saved to {stems_dir}")
                    else:
                        logger.warning(f"Request {request_id}: Separation returned None, using original audio")
                except Exception as e:
                    logger.warning(f"Request {request_id}: Separation failed: {str(e)}, falling back to original")
                
                timings['separation'] = int((time.time() - sep_start) * 1000)
            
            # Transcription
            trans_start = time.time()
            result = transcriber.transcribe(
                audio_data,
                sample_rate,
                language=config_obj.language_hint,
                model_size=config_obj.model_size,
                apply_vad=False  # VAD already applied above if configured
            )
            timings['transcription'] = int((time.time() - trans_start) * 1000)
            
            # Format segments
            segments = []
            for seg in result.get('segments', []):
                segments.append(AudioSegment(
                    start=float(seg.get('start', 0)),
                    end=float(seg.get('end', 0)),
                    text=seg.get('text', '').strip(),
                    speaker=seg.get('speaker') if config_obj.diarize else None
                ))
            
            # Calculate total time
            timings['total'] = int((time.time() - start_time) * 1000)
            
            response = TranscriptionResponse(
                request_id=request_id,
                duration_sec=duration,
                sample_rate=sample_rate,
                pipeline={
                    "vad": {
                        "enabled": vad_enabled
                    },
                    "separation": {
                        "enabled": separation_enabled,
                        "method": separation_method or "none",
                        "stems_dir": stems_location
                    },
                    "transcription": {
                        "model": f"whisper-{config_obj.model_size}"
                    }
                },
                segments=segments,
                text=result.get('text', ''),
                language=result.get('language', 'unknown'),
                timings_ms=timings
            )
            
            logger.info(f"Request {request_id}: Completed successfully in {timings['total']}ms")
            return response
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request {request_id}: Error - {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "request_id": request_id,
                "error": str(e),
                "type": type(e).__name__
            }
        )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    request_id = str(uuid.uuid4())
    logger.error(f"Unhandled exception for request {request_id}: {str(exc)}\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )