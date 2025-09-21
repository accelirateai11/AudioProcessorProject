# src/audio_processor.py
import os
import logging
import subprocess
from typing import Tuple
import numpy as np
import soundfile as sf
import librosa
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handle audio loading, conversion, and preprocessing"""
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus', '.webm'}
    
    def __init__(self):
        self.ffmpeg_available = self._check_ffmpeg()
        if not self.ffmpeg_available:
            logger.warning("ffmpeg not available - some file formats may not be supported")
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available"""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
            return True
        except (FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def load_audio(self, 
                  file_path: str, 
                  target_sr: int = 16000,
                  mono: bool = True,
                  validate_format: bool = True) -> Tuple[np.ndarray, int, float]:
        """
        Load audio file and convert to target sample rate
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate
            mono: Convert to mono
            validate_format: Check if format is supported
            
        Returns:
            Tuple of (audio_array, sample_rate, duration_seconds)
            
        Raises:
            ValueError: If file format is not supported
            RuntimeError: If file cannot be loaded
        """
        # Validate file format
        if validate_format:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported file format: {ext}. Supported formats: {', '.join(self.SUPPORTED_FORMATS)}")
        
        try:
            # Try loading with librosa first (handles most formats)
            audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
            duration = librosa.get_duration(y=audio, sr=sr)
            return audio, sr, duration
            
        except Exception as e:
            logger.warning(f"Failed to load with librosa: {str(e)}")
            
            # Fall back to ffmpeg if available
            if self.ffmpeg_available:
                try:
                    audio, sr = self._load_with_ffmpeg(file_path, target_sr, mono)
                    duration = len(audio) / sr
                    return audio, sr, duration
                except Exception as ffmpeg_error:
                    raise RuntimeError(f"Failed to load audio file: {str(ffmpeg_error)}")
            else:
                raise RuntimeError(f"Failed to load audio file and ffmpeg not available: {str(e)}")
    
    def _load_with_ffmpeg(self, 
                          file_path: str, 
                          target_sr: int, 
                          mono: bool) -> Tuple[np.ndarray, int]:
        """Load audio using ffmpeg"""
        import subprocess
        import tempfile
        
        # Create temp WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
        
        try:
            # Build ffmpeg command
            cmd = ["ffmpeg", "-i", file_path, "-ar", str(target_sr)]
            
            if mono:
                cmd.extend(["-ac", "1"])
                
            cmd.extend(["-f", "wav", temp_wav_path, "-y", "-loglevel", "error"])
            
            # Run ffmpeg
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Load the converted WAV
            audio, sr = sf.read(temp_wav_path)
            return audio, sr
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
    
    def apply_vad(self, 
                 audio: np.ndarray, 
                 sample_rate: int,
                 frame_duration_ms: int = 30,
                 threshold: float = 0.01,
                 min_silence_duration_ms: int = 300) -> np.ndarray:
        """
        Apply Voice Activity Detection to remove silence
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            frame_duration_ms: Frame size for energy calculation in ms
            threshold: Energy threshold (0-1)
            min_silence_duration_ms: Minimum silence duration to remove in ms
            
        Returns:
            Audio with silences removed
        """
        logger.info("Applying VAD to remove silence")
        
        # Convert durations to samples
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        min_silence_samples = int(sample_rate * min_silence_duration_ms / 1000)
        
        # Calculate energy for each frame
        energies = []
        for i in range(0, len(audio), frame_size):
            frame = audio[i:i+frame_size]
            if len(frame) < frame_size:
                # Pad last frame if needed
                frame = np.pad(frame, (0, frame_size - len(frame)))
            
            # Calculate energy (RMS)
            energy = np.sqrt(np.mean(frame**2))
            energies.append(energy)
        
        # Calculate threshold as percentage of max energy if relative
        if threshold < 1.0:
            abs_threshold = threshold * max(energies)
        else:
            abs_threshold = threshold
        
        # Mark each frame as speech or silence
        is_speech = [energy > abs_threshold for energy in energies]
        
        # Only remove long silence segments
        # Convert to samples where speech is active
        active_samples = []
        current_segment = []
        in_silence = False
        silence_frames = 0
        
        for i, speech in enumerate(is_speech):
            frame_start = i * frame_size
            frame_end = min(frame_start + frame_size, len(audio))
            
            if speech:
                # If we were in silence and it wasn't long enough, add the silence frames
                if in_silence and silence_frames * frame_size < min_silence_samples:
                    active_samples.extend(current_segment)
                
                # Add current speech frame
                active_samples.extend(range(frame_start, frame_end))
                in_silence = False
                silence_frames = 0
                current_segment = []
            else:
                if not in_silence:
                    # Starting a new silence segment
                    in_silence = True
                    current_segment = list(range(frame_start, frame_end))
                    silence_frames = 1
                else:
                    # Continue the silence segment
                    current_segment.extend(range(frame_start, frame_end))
                    silence_frames += 1
        
        # Handle last silence segment if not long enough
        if in_silence and silence_frames * frame_size < min_silence_samples:
            active_samples.extend(current_segment)
        
        # Create a mask of active samples
        mask = np.zeros(len(audio), dtype=bool)
        mask[active_samples] = True
        
        # Create output audio
        output_audio = audio[mask]
        
        removed_duration = (len(audio) - len(output_audio)) / sample_rate
        logger.info(f"VAD removed {removed_duration:.2f}s of silence ({removed_duration/len(audio)*sample_rate*100:.1f}%)")
        
        return output_audio
    
    def chunk_audio(self,
                   audio: np.ndarray,
                   sample_rate: int,
                   chunk_duration: float = 30.0,
                   overlap: float = 2.0) -> List[Tuple[np.ndarray, float]]:
        """
        Split long audio into overlapping chunks
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            chunk_duration: Maximum chunk duration in seconds
            overlap: Overlap between chunks in seconds
            
        Returns:
            List of (chunk_audio, start_time) tuples
        """
        logger.info(f"Chunking audio of {len(audio)/sample_rate:.2f}s into {chunk_duration}s segments with {overlap}s overlap")
        
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        stride = chunk_samples - overlap_samples
        
        # Handle edge case of very short audio
        if len(audio) <= chunk_samples:
            return [(audio, 0.0)]
        
        chunks = []
        for start in range(0, len(audio), stride):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            chunks.append((chunk, start / sample_rate))
            
            if end == len(audio):
                break
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def process_long_audio(self, audio: np.ndarray, sample_rate: int, 
                          transcriber, max_duration: float = 30.0, 
                          overlap: float = 2.0, **transcribe_kwargs) -> Dict[str, Any]:
        """
        Process long audio by chunking and stitching results
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            transcriber: Transcriber instance
            max_duration: Maximum chunk duration
            overlap: Overlap between chunks
            **transcribe_kwargs: Additional arguments for transcriber
            
        Returns:
            Combined transcription result
        """
        duration = len(audio) / sample_rate
        logger.info(f"Processing long audio ({duration:.2f}s) using chunking")
        
        # If audio is shorter than max_duration, process directly
        if duration <= max_duration:
            return transcriber.transcribe(audio, sample_rate, **transcribe_kwargs)
        
        # Split into chunks
        chunks = self.chunk_audio(audio, sample_rate, max_duration, overlap)
        
        # Process each chunk
        results = []
        for i, (chunk, start_time) in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} (start: {start_time:.2f}s)")
            
            # Transcribe chunk
            result = transcriber.transcribe(chunk, sample_rate, **transcribe_kwargs)
            
            # Adjust segment timestamps
            for segment in result.get('segments', []):
                segment['start'] += start_time
                segment['end'] += start_time
            
            results.append(result)
        
        # Merge results
        return self._merge_chunk_results(results)
    
    def _merge_chunk_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from multiple chunks
        
        Args:
            results: List of transcription results
            
        Returns:
            Merged result
        """
        if not results:
            return {"text": "", "segments": [], "language": "unknown"}
        
        # Determine most common language
        languages = [r.get('language', 'unknown') for r in results]
        language = max(set(languages), key=languages.count)
        
        # Combine texts with spaces
        texts = [r.get('text', '').strip() for r in results]
        combined_text = ' '.join(texts)
        
        # Combine segments
        segments = []
        for result in results:
            segments.extend(result.get('segments', []))
        
        # Sort segments by start time
        segments.sort(key=lambda x: x.get('start', 0))
        
        # Resolve overlaps and merge adjacent segments
        processed_segments = self._resolve_segment_overlaps(segments)
        
        return {
            "text": combined_text,
            "segments": processed_segments,
            "language": language
        }
    
    def _resolve_segment_overlaps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve overlaps between segments
        
        Args:
            segments: List of segments with potential overlaps
            
        Returns:
            List of segments with resolved overlaps
        """
        if not segments:
            return []
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x.get('start', 0))
        
        processed = [sorted_segments[0]]
        for current in sorted_segments[1:]:
            previous = processed[-1]
            
            # Check for overlap
            if current['start'] <= previous['end']:
                # If significant overlap, merge segments
                overlap_size = previous['end'] - current['start']
                segment_size = current['end'] - current['start']
                
                if overlap_size > segment_size * 0.5:
                    # Merge into previous segment
                    previous['end'] = max(previous['end'], current['end'])
                    previous['text'] = f"{previous['text']} {current['text']}"
                    
                    # Merge speaker if available
                    if 'speaker' in previous and 'speaker' in current:
                        if previous['speaker'] != current['speaker']:
                            previous['speaker'] = f"{previous['speaker']}/{current['speaker']}"
                else:
                    # Adjust boundary to remove small overlap
                    boundary = (previous['end'] + current['start']) / 2
                    previous['end'] = boundary
                    current['start'] = boundary
                    processed.append(current)
            else:
                processed.append(current)
        
        return processed