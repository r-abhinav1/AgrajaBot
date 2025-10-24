import os
import librosa
import numpy as np
import noisereduce as nr
from typing import Optional
import io
import warnings

# --- Configuration ---
SEGMENT_DURATION = 10       # seconds
OVERLAP = 5                 # seconds
TARGET_SR = 16000           # Hz

class AudioPreprocessor:
    def __init__(self, target_sr=TARGET_SR, segment_duration=SEGMENT_DURATION, overlap=OVERLAP, debug: bool = False):
        self.target_sr = target_sr
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.debug = debug

    def _load_audio(self, filepath):
        """
        Load audio from m4a or wav and resample to target_sr.
        Returns waveform (numpy array) and sample rate.
        """
        if filepath.lower().endswith(".m4a"):
            # Convert m4a to wav in memory (lazy import pydub to avoid ffmpeg warning otherwise)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    from pydub import AudioSegment, utils as pdu

                    # Configure ffmpeg/ffprobe paths if not available on PATH
                    ffmpeg_path = os.environ.get("FFMPEG_BIN") or pdu.which("ffmpeg")
                    ffprobe_path = os.environ.get("FFPROBE_BIN") or pdu.which("ffprobe")

                    if not ffmpeg_path or not ffprobe_path:
                        # Try local bundled binaries under project_root/ffmpeg/{ffmpeg,ffprobe}[.exe]
                        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
                        ff_dir = os.path.join(project_root, "ffmpeg")
                        exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
                        pexe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
                        local_ffmpeg = os.path.join(ff_dir, exe)
                        local_ffprobe = os.path.join(ff_dir, pexe)
                        if os.path.isfile(local_ffmpeg) and os.path.isfile(local_ffprobe):
                            ffmpeg_path, ffprobe_path = local_ffmpeg, local_ffprobe

                    if ffmpeg_path:
                        # pydub uses these attributes to locate binaries
                        AudioSegment.converter = ffmpeg_path
                        AudioSegment.ffmpeg = ffmpeg_path
                    if ffprobe_path:
                        AudioSegment.ffprobe = ffprobe_path

                    # Attempt decode (try 'm4a' then 'mp4' for broader compatibility)
                    try:
                        audio = AudioSegment.from_file(filepath, format="m4a")
                    except Exception as e1:
                        try:
                            audio = AudioSegment.from_file(filepath, format="mp4")
                        except Exception as e2:
                            raise RuntimeError(
                                f"Reading .m4a with ffmpeg failed. Tried formats m4a/mp4. ffmpeg={ffmpeg_path}, ffprobe={ffprobe_path}. "
                                f"Errors: m4a->{str(e1)} | mp4->{str(e2)}"
                            ) from e2
            except Exception as e:
                raise RuntimeError(
                    "Reading .m4a requires ffmpeg. Install ffmpeg and ensure it's on PATH, or place ffmpeg/ffprobe binaries in 'ffmpeg/' folder at the project root (Dementai/ffmpeg) and retry. "
                    f"Underlying error: {str(e)}"
                ) from e
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)
            y, sr = librosa.load(wav_io, sr=self.target_sr, mono=True)
        else:
            y, sr = librosa.load(filepath, sr=self.target_sr, mono=True)
        return y, sr

    def _reduce_noise(self, y, sr):
        """Spectral gating noise reduction."""
        return nr.reduce_noise(y=y, sr=sr)

    def _trim_silence(self, y, sr, top_db=30):
        """Trim leading and trailing silence."""
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        return yt

    def _rms_normalize(self, y, target_db=-20):
        """Normalize audio to consistent loudness."""
        rms = np.sqrt(np.mean(y ** 2))
        scalar = 10 ** (target_db / 20) / (rms + 1e-9)
        return y * scalar

    def _pre_emphasis(self, y, coeff=0.97):
        """Apply pre-emphasis filter."""
        if len(y) == 0:
            return y
        return np.append(y[0], y[1:] - coeff * y[:-1])

    def _segment_audio(self, y, sr):
        """Split into 10s overlapping segments."""
        step = self.segment_duration - self.overlap
        samples_per_segment = int(self.segment_duration * sr)
        step_size = int(step * sr)
        segments = []

        for start in range(0, len(y), step_size):
            end = start + samples_per_segment
            seg = y[start:end]
            if len(seg) < samples_per_segment:
                seg = np.pad(seg, (0, samples_per_segment - len(seg)))
            segments.append(seg)
            if end >= len(y):
                break
        return segments

    def preprocess(self, filepath):
        """
        Full preprocessing pipeline.
        Returns: List of numpy arrays (segments)
        """
        # Step 1: Load
        y, sr = self._load_audio(filepath)
        
        print(f"[Pre] Loaded: sr={sr}, len={len(y)}, dur={len(y)/sr:.2f}s, mean={np.mean(y):.4f}, std={np.std(y):.4f}")

        # Step 2: Noise reduction
        y = self._reduce_noise(y, sr)
        
        print(f"[Pre] After noise reduce: mean={np.mean(y):.4f}, std={np.std(y):.4f}")

        # Step 3: Trim silence
        y = self._trim_silence(y, sr)
        if len(y) == 0:
            raise ValueError("No speech detected after silence trimming.")
        
        print(f"[Pre] After trim: len={len(y)}, dur={len(y)/sr:.2f}s")

        # Step 4: RMS normalize
        y = self._rms_normalize(y)
        
        print(f"[Pre] After RMS norm: mean={np.mean(y):.4f}, std={np.std(y):.4f}")

        # Step 5: Pre-emphasis
        y = self._pre_emphasis(y)
        
        print(f"[Pre] After pre-emphasis: mean={np.mean(y):.4f}, std={np.std(y):.4f}")

        # Step 6: Segment
        segments = self._segment_audio(y, sr)
        
        lens = [len(s) for s in segments]
        print(f"[Pre] Segments: {len(segments)} | min={min(lens)} max={max(lens)} mean={int(np.mean(lens))} samples")

        return segments
