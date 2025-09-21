import pytest
import httpx
import json
import io
import numpy as np
import soundfile as sf
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

API_URL = "http://localhost:8000"

@pytest.fixture
def client():
    """Create test client"""
    return httpx.Client(base_url=API_URL, timeout=120.0)

@pytest.fixture
def sample_audio():
    """Generate sample audio for testing"""
    sample_rate = 16000
    duration = 3  # seconds
    frequency = 440  # Hz (A4 note)
    
    t = np.linspace(0, duration, sample_rate * duration)
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    
    # Add some noise
    noise = np.random.normal(0, 0.01, audio.shape)
    audio = audio + noise
    
    # Create in-memory WAV file
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format='WAV')
    buffer.seek(0)
    
    return buffer

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "message" in data  # <-- update here

def test_transcribe_basic(client, sample_audio):
    """Test basic transcription"""
    files = {"file": ("test.wav", sample_audio, "audio/wav")}
    response = client.post("/v1/transcribe", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "request_id" in data
    assert "text" in data
    assert "segments" in data
    assert "duration_sec" in data
    assert data["duration_sec"] > 0

def test_invalid_file(client):
    """Test with invalid file"""
    files = {"file": ("test.txt", b"not audio", "text/plain")}
    response = client.post("/v1/transcribe", files=files)
    
    assert response.status_code in [400, 422]

def test_empty_file(client):
    """Test with empty file"""
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/v1/transcribe", files=files)
    
    assert response.status_code == 400

def test_invalid_config(client, sample_audio):
    """Test with invalid configuration"""
    files = {"file": ("test.wav", sample_audio, "audio/wav")}
    data = {"config": "invalid json"}
    
    response = client.post("/v1/transcribe", files=files, data=data)
    
    assert response.status_code == 400

if __name__ == "__main__":
    pytest.main([__file__, "-v"])