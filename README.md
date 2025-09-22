# Audio Transcription Microservice

A containerized microservice that accepts audio files, separates vocals from background noise, and returns accurate transcriptions. Built with FastAPI, Whisper, and Demucs.

## Features

- 🎯 **Audio Transcription**: High-quality speech-to-text using OpenAI Whisper
- 🎵 **Vocal Separation**: Isolate speech from background noise using Demucs
- 🌍 **Multi-language Support**: Automatic language detection and transcription
- 🚀 **GPU Acceleration**: Automatic GPU detection and utilization when available
- 📊 **Detailed Metrics**: Request tracking, timing analytics, and structured logging
- 🐳 **Fully Containerized**: Docker and Docker Compose support
- 📝 **Multiple Output Formats**: JSON, plain text, SRT, and WebVTT
- 🔧 **REST API**: Well-documented OpenAPI/Swagger interface
- 🔊 **Voice Activity Detection**: Removes silent segments for improved accuracy
- ⚡ **Long File Support**: Processes long audio files via chunking and stitching

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/accelirateai11/AudioProcessorProject.git
cd audio-transcription-service

# Build and run with Docker Compose
docker-compose up --build

# The service will be available at http://localhost:8000
```

### Local Installation

```bash

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

## API Usage

### Basic Transcription

```bash
curl -X POST "http://localhost:8000/v1/transcribe" \
  -F "file=@sample.mp3" \
  -F 'config={"language_hint":"en","enable_separation":true}'
```

### Using the CLI Tool

```bash
# Basic transcription
python cli.py audio.mp3

# Specify output format (text, json, or srt)
python cli.py path/to/audio_file.mp3 --format json
python cli.py path/to/audio_file.mp3 --format srt

# Save output to a file
python cli.py path/to/audio_file.mp3 --output transcript.txt

# Change model size
python cli.py path/to/audio_file.mp3 --model medium

# Specify language hint
python cli.py path/to/audio_file.mp3 --language en

# Disable vocal separation
python cli.py path/to/audio_file.mp3 --no-separation

# Enable speaker diarization
python cli.py path/to/audio_file.mp3 --diarize

# Disable Voice Activity Detection
python cli.py path/to/audio_file.mp3 --no-vad

# Show detailed progress and stats
python cli.py path/to/audio_file.mp3 --verbose

# Point to a different API server
python cli.py path/to/audio_file.mp3 --url http://other-server:8000

# Get help
python cli.py --help
```

### Python Client Example

```python
import requests
import json

# Prepare the request
url = "http://localhost:8000/v1/transcribe"
files = {'file': open('audio.mp3', 'rb')}
config = {
    "language_hint": "en",
    "enable_separation": True,
    "model_size": "small"
}
data = {'config': json.dumps(config)}

# Send request
response = requests.post(url, files=files, data=data)
result = response.json()

# Print transcription
print(result['text'])
```

## API Documentation

### Endpoint: `POST /v1/transcribe`

**Request:**
- `file`: Audio file (multipart/form-data) - Supported formats: wav, mp3, m4a, flac, ogg
- `config`: Optional JSON configuration

**Configuration Options:**
```json
{
  "language_hint": "en",      // ISO language code
  "enable_separation": true,   // Enable vocal separation
  "diarize": false,           // Enable speaker diarization
  "model_size": "small",      // tiny, base, small, medium, large
  "target_sr": 16000,         // Target sample rate
  "apply_vad": true,          // Apply Voice Activity Detection
  "vad_threshold": 0.01       // Energy threshold for VAD (0-1)
}
```

**Response:**
```json
{
  "request_id": "uuid",
  "duration_sec": 31.2,
  "sample_rate": 16000,
  "pipeline": {
    "vad": {
      "enabled": true
    },
    "separation": {
      "enabled": true, 
      "method": "demucs",
      "stems_dir": "/path/to/stems"
    },
    "transcription": {
      "model": "whisper-small"
    }
  },
  "segments": [
    {"start": 0.0, "end": 3.1, "text": "Hello world", "speaker": null}
  ],
  "text": "Hello world",
  "language": "en",
  "timings_ms": {
    "load": 420,
    "vad": 100,
    "separation": 1800,
    "transcription": 4100,
    "total": 6400
  }
}
```

**Error Responses:**
- `400`: Invalid file or format
- `413`: File too large (>100MB)
- `422`: Audio decode failure
- `500`: Internal server error

## Architecture

### System Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Client    │────▶│   FastAPI App   │────▶│ Audio Loader │
└─────────────┘     └─────────────────┘     └──────────────┘
                             │                      │
                             ▼                      ▼
                    ┌─────────────────┐     ┌──────────────┐
                    │  Request Queue  │     │   FFmpeg     │
                    └─────────────────┘     └──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Voice Activity  │
                    │   Detection     │
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐     ┌──────────────┐
                    │ Vocal Separator │────▶│  Saved Stems │
                    │    (Demucs)     │     │  Directory   │
                    └─────────────────┘     └──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Transcriber    │
                    │   (Whisper)     │
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Post-processor  │
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Response     │
                    └─────────────────┘
```

### Pipeline Stages

1. **Audio Decoding**: Load and decode audio using torchaudio/soundfile/ffmpeg
2. **Normalization**: Convert to standard format (16kHz, mono, float32)
3. **Voice Activity Detection**: Remove silent segments
4. **Vocal Separation**: Isolate speech using Demucs (optional)
- Saves stems (vocals, drums, bass, other) to a request-specific directory
5. **VAD**: Remove silence and split long audio
6. **Chunking**: Split long audio files into manageable segments with overlap
7. **Transcription**: Convert speech to text using Whisper
8. **Diarization**: Add speaker labels to transcription (optional)
9. **Post-processing**: Format response, merge segments, resolve overlaps.

### Performance Optimization

- **Model Caching**: Models are loaded once and reused across requests
- **Lazy Loading**: Models are loaded only when needed
- **GPU Detection**: Automatically uses GPU when available
- **Chunking**: Processes long audio files in smaller pieces for memory efficiency
- **Request Tracing**: Each request has a unique ID for tracking through the pipeline

## Configuration

Configure the service using environment variables (see `.env.example`):

- `PORT`: Server port (default: 8000)
- `WORKERS`: Number of worker processes (default: 1)
- `MODEL_CACHE_DIR`: Directory for cached models
- `DEFAULT_MODEL_SIZE`: Default Whisper model size
- `MAX_FILE_SIZE_MB`: Maximum file size in MB (default: 100)
- `MAX_DURATION_SEC`: Maximum audio duration in seconds (default: 600)

## Limitations and Trade-offs

- **Model Size**: Larger models provide better accuracy but require more memory and processing time
- **Without GPU**: Processing times are significantly longer on CPU-only environments
- **Multi-speaker Audio**: Transcription quality may decrease with overlapping speech
- **Very Long Files**: Files longer than 10 minutes are processed in chunks, which may affect coherence

## Development

### Prerequisites

- Python 3.10+
- Poetry
- Docker & Docker Compose (for containerized development)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/audio-transcription-service.git
cd audio-transcription-service

# Install dependencies
poetry install

# Configure environment variables
cp .env.example .env
nano .env  # Update settings as needed

# Run the development server
poetry run uvicorn main:app --reload
```

### Testing

```bash
# Run tests
pytest

# Run specific test
pytest tests/test_api.py::test_transcribe_basic
```

## Troubleshooting

- **Common Issues**:
  - If you encounter issues, contact us.
  - Ensure all environment variables are set correctly in the `.env` file.
  - Check file permissions for reading audio files and writing outputs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech-to-text model
- [Demucs](https://github.com/facebookresearch/demucs) for the vocal separation model
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Docker](https://www.docker.com/) for containerization