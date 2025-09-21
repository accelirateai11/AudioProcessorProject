# src/separator.py
import logging
import torch
import numpy as np
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

class VocalSeparator:
    """Handle vocal/noise separation using Demucs or fallback methods"""
    
    def __init__(self, device: str = 'cpu', method: str = 'auto', output_dir: str = None):
        """
        Initialize separator
        
        Args:
            device: Device to use (cpu/cuda)
            method: Separation method (auto/demucs/spectral)
            output_dir: Directory to save separated stems (if None, won't save)
        """
        self.device = device
        self.method = method
        self.model = None
        self.method_name = None
        self.output_dir = output_dir
        
        if method == 'auto' or method == 'demucs':
            self._init_demucs()
        
        if self.model is None:
            self._init_fallback()
    
    def _init_demucs(self):
        """Initialize Demucs model"""
        try:
            from demucs import pretrained
            from demucs.apply import apply_model
            from demucs.audio import convert_audio
            
            logger.info("Loading Demucs model...")
            
            # Load a smaller model for efficiency
            self.model = pretrained.get_model('htdemucs')
            self.model.to(self.device)
            self.model.eval()
            
            self.apply_model = apply_model
            self.convert_audio = convert_audio
            self.method_name = 'demucs'
            
            logger.info(f"Demucs model loaded on {self.device}")
            
        except Exception as e:
            logger.warning(f"Failed to load Demucs: {str(e)}")
            self.model = None
    
    def _init_fallback(self):
        """Initialize fallback separation method"""
        self.method_name = 'spectral_subtraction'
        logger.info("Using spectral subtraction as fallback")
    
    def separate(self, 
                 audio: np.ndarray, 
                 sample_rate: int,
                 file_name: str = None,
                 output_dir: str = None) -> Optional[np.ndarray]:
        """
        Separate vocals from audio
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
            file_name: Original file name for saving stems
            output_dir: Directory to save stems (overrides self.output_dir)
            
        Returns:
            Separated vocal audio or None if failed
        """
        save_dir = output_dir or self.output_dir
        
        try:
            if self.model is not None and self.method_name == 'demucs':
                return self._separate_demucs(audio, sample_rate, file_name, save_dir)
            else:
                return self._separate_spectral(audio, sample_rate, file_name, save_dir)
                
        except Exception as e:
            logger.error(f"Separation failed: {str(e)}")
            return None
    
    def _separate_demucs(self, 
                          audio: np.ndarray, 
                          sample_rate: int, 
                          file_name: str = None,
                          output_dir: str = None) -> np.ndarray:
        """Separate using Demucs and save stems"""
        import os
        import soundfile as sf
        import torch
        from pathlib import Path
        
        with torch.no_grad():
            # Convert to torch tensor
            if len(audio.shape) == 1:
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            else:
                audio_tensor = torch.from_numpy(audio).float()
            
            # Add batch dimension if needed
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Convert audio to model's expected sample rate (44100)
            audio_tensor = self.convert_audio(
                audio_tensor,
                sample_rate,
                self.model.samplerate,
                self.model.audio_channels
            )
            
            # Move to device
            audio_tensor = audio_tensor.to(self.device)
            
            # Apply model
            sources = self.apply_model(
                self.model,
                audio_tensor,
                device=self.device,
                shifts=1,  # Reduced shifts for speed
                overlap=0.25,
                progress=False
            )[0]
            
            # Save stems if output directory is provided
            if output_dir and file_name:
                stem_names = ["drums", "bass", "other", "vocals"]
                
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Get base filename without extension
                base_name = Path(file_name).stem
                
                # Save each stem
                for i, name in enumerate(stem_names):
                    stem_audio = sources[i].cpu().numpy()
                    
                    # Convert to original sample rate if needed
                    if sample_rate != self.model.samplerate:
                        import torchaudio
                        stem_tensor = torch.from_numpy(stem_audio)
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=self.model.samplerate,
                            new_freq=sample_rate
                        )
                        stem_audio = resampler(stem_tensor).numpy()
                    
                    # Convert to mono if input was mono
                    if len(audio.shape) == 1 and len(stem_audio.shape) > 1:
                        stem_audio = np.mean(stem_audio, axis=0)
                    
                    # Determine file extension
                    file_ext = Path(file_name).suffix or ".wav"
                    
                    # Save the stem
                    stem_path = os.path.join(output_dir, f"{base_name}_{name}{file_ext}")
                    logger.info(f"Saving {name} stem to {stem_path}")
                    
                    sf.write(stem_path, stem_audio.squeeze(), sample_rate)
            
            # Extract vocals (usually index 3 for vocals in htdemucs)
            vocals = sources[3].cpu().numpy()
            
            # Convert back to original sample rate if needed
            if sample_rate != self.model.samplerate:
                import torchaudio
                vocals_tensor = torch.from_numpy(vocals)
                resampler = torchaudio.transforms.Resample(
                    orig_freq=self.model.samplerate,
                    new_freq=sample_rate
                )
                vocals = resampler(vocals_tensor).numpy()
            
            # Convert to mono if input was mono
            if len(audio.shape) == 1 and len(vocals.shape) > 1:
                vocals = np.mean(vocals, axis=0)
            
            return vocals.squeeze()
    
    def _separate_spectral(self, 
                           audio: np.ndarray, 
                           sample_rate: int,
                           file_name: str = None,
                           output_dir: str = None) -> np.ndarray:
        """Fallback spectral subtraction for noise reduction"""
        try:
            from scipy import signal
            from scipy.fft import fft, ifft
            import os
            import soundfile as sf
            from pathlib import Path
            
            # Estimate noise from first 0.5 seconds
            noise_duration = min(int(0.5 * sample_rate), len(audio) // 4)
            noise_sample = audio[:noise_duration]
            
            # Compute STFT
            nperseg = 2048
            noverlap = nperseg // 2
            
            f, t, Zxx = signal.stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
            f_noise, t_noise, Zxx_noise = signal.stft(noise_sample, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
            
            # Estimate noise spectrum
            noise_spectrum = np.mean(np.abs(Zxx_noise), axis=1, keepdims=True)
            
            # Spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            floor_factor = 0.1  # Noise floor
            
            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)
            
            # Subtract noise
            clean_magnitude = magnitude - alpha * noise_spectrum
            clean_magnitude = np.maximum(clean_magnitude, floor_factor * magnitude)
            
            # Reconstruct signal
            clean_Zxx = clean_magnitude * np.exp(1j * phase)
            _, clean_audio = signal.istft(clean_Zxx, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
            
            # Ensure same length as input
            if len(clean_audio) > len(audio):
                clean_audio = clean_audio[:len(audio)]
            elif len(clean_audio) < len(audio):
                clean_audio = np.pad(clean_audio, (0, len(audio) - len(clean_audio)))
            
            # Save stems if output directory is provided
            if output_dir and file_name:
                # Create output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Get base filename without extension
                base_name = Path(file_name).stem
                file_ext = Path(file_name).suffix or ".wav"
                
                # Save vocals (cleaned audio)
                vocals_path = os.path.join(output_dir, f"{base_name}_vocals{file_ext}")
                logger.info(f"Saving vocals to {vocals_path}")
                sf.write(vocals_path, clean_audio, sample_rate)
                
                # Save noise (difference between original and vocals)
                noise_audio = audio - clean_audio
                noise_path = os.path.join(output_dir, f"{base_name}_noise{file_ext}")
                logger.info(f"Saving noise to {noise_path}")
                sf.write(noise_path, noise_audio, sample_rate)
            
            return clean_audio
            
        except ImportError:
            logger.warning("scipy not available for spectral subtraction")
            return audio
    
    def cleanup(self):
        """Clean up resources"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()