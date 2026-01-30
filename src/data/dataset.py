import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import random

import numpy as np
import torch
import soundfile as sf

from .transforms import AudioPreprocessor

logger = logging.getLogger('mae_ast')


class AudioDataset(torch.utils.data.Dataset):
    """
    Dataset for loading audio from TSV manifests.

    This dataset:
    1. Reads manifests TSV to get file paths
    2. Loads raw audio waveforms
    3. Applies transforms (mel spectrogram, normalization)
    4. Returns processed features ready for the model

    Args:
        manifest_path: Path to TSV manifests file
        sample_rate: Target sample rate
        max_duration: Maximum audio duration in seconds
        min_duration: Minimum audio duration in seconds
        feature_type: Type of features ('fbank', 'mfcc', 'spectrogram')
        n_mels: Number of mel filterbanks
        normalize: Whether to normalize features
        random_crop: If True, randomly crop audio; else crop from start
    """

    def __init__(
            self,
            manifest_path: str,
            sample_rate: int = 16000,
            max_duration: Optional[float] = 10.0,
            min_duration: Optional[float] = 0.5,
            feature_type: str = 'fbank',
            n_mels: int = 128,
            normalize: bool = True,
            random_crop: bool = True,
            return_labels: bool = False,
    ):
        self.manifest_path = manifest_path
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.feature_type = feature_type
        self.n_mels = n_mels
        self.normalize = normalize
        self.random_crop = random_crop
        self.return_labels = return_labels

        # Convert duration to samples
        self.max_samples = int(max_duration * sample_rate) if max_duration else None
        self.min_samples = int(min_duration * sample_rate) if min_duration else None

        # Load manifests
        self.root_path, self.audio_paths, self.audio_sizes, self.labels = self._load_manifest()

        # Initialize audio preprocessor
        # This handles conversion to mel spectrogram as described in paper
        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            feature_type=feature_type,
            n_mels=n_mels,
            normalize=normalize
        )

        logger.info(f"Loaded {len(self)} audio files from {manifest_path}")
        logger.info(f"Feature type: {feature_type}, n_mels: {n_mels}")
        logger.info(f"Max duration: {max_duration}s, Random crop: {random_crop}")
        logger.info(f"Return labels: {return_labels}")
        if return_labels and self.labels is not None:
            logger.info(f"Number of unique labels: {len(set(self.labels))}")

    def _load_manifest(self) -> Tuple[Path, List[str], List[int], Optional[List[int]]]:
        """
        Load TSV manifests file.

        Returns:
            root_path: Root directory for audio files
            audio_paths: List of relative audio file paths
            audio_sizes: List of audio lengths in samples
            labels: List of labels (None if unlabeled)
        """
        audio_paths = []
        audio_sizes = []
        labels = [] if self.return_labels else None

        with open(self.manifest_path, 'r') as f:
            lines = f.readlines()

        # First line is root directory
        root_path = Path(lines[0].strip())

        # Parse remaining lines
        for i, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')

            if len(parts) < 2:
                logger.warning(f"Line {i}: Expected 2 columns (filename, frames), got {len(parts)}. Skipping.")
                continue

            label = None
            if self.return_labels:
                # Expect 3 columns: filename, num_samples, label
                if len(parts) < 3:
                    logger.warning(
                        f"Line {i}: Expected 3 columns (filename, frames, label), got {len(parts)}. Skipping.")
                    continue

                label = parts[2]

            filename, num_samples = parts[0], parts[1]
            num_samples = int(num_samples)

            # Filter by duration
            if self.min_samples and num_samples < self.min_samples:
                continue
            if self.max_samples and num_samples > self.max_samples:
                # We'll crop long audios, so don't skip them
                pass

            audio_paths.append(filename)
            audio_sizes.append(num_samples)
            if self.return_labels:
                labels.append(int(label))

        logger.info(f"Root path: {root_path}")
        logger.info(f"Loaded {len(audio_paths)} valid audio files")

        return root_path, audio_paths, audio_sizes, labels

    def _load_audio(self, index: int) -> torch.Tensor:
        """
        Load audio waveform from disk.

        Args:
            index: Dataset index

        Returns:
            waveform: Audio tensor [num_samples]
        """
        # Construct full path
        audio_path = self.root_path / self.audio_paths[index]

        # Load audio
        try:
            waveform, sr = sf.read(str(audio_path), dtype='float32')
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            # Return silent audio as fallback
            waveform = np.zeros(self.sample_rate, dtype=np.float32)
            sr = self.sample_rate

        # Convert to torch tensor
        waveform = torch.from_numpy(waveform)

        # Convert stereo to mono if needed
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=-1)

        # Resample if needed
        if sr != self.sample_rate:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        return waveform

    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Crop or pad audio to max_samples length.

        Args:
            waveform: Input audio [num_samples]

        Returns:
            waveform: Cropped/padded audio [max_samples]
        """
        if self.max_samples is None:
            return waveform

        current_samples = waveform.size(0)

        if current_samples == self.max_samples:
            return waveform

        elif current_samples > self.max_samples:
            # Crop
            if self.random_crop:
                # Random crop (used during training)
                start = random.randint(0, current_samples - self.max_samples)
            else:
                # Crop from beginning (used during evaluation)
                start = 0

            return waveform[start:start + self.max_samples]

        else:
            # Pad with zeros
            padding = self.max_samples - current_samples
            return torch.nn.functional.pad(waveform, (0, padding), value=0.0)

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, index: int) -> dict:
        """
        Get a single audio sample.

        Returns:
            dict with keys:
                - 'audio': Processed features [time, n_mels]
                - 'audio_length': Original length before padding
                - 'id': Sample index
                - 'path': Audio file path (for debugging)
                - 'label': Label (only if return_labels=True)
        """
        # Load raw waveform
        waveform = self._load_audio(index)

        # Store original length
        original_length = waveform.size(0)

        # Crop or pad to fixed length
        waveform = self._crop_or_pad(waveform)

        # Convert to features (mel spectrogram)
        features = self.preprocessor(waveform)

        output = {
            'audio': features,  # [time, n_mels]
            'audio_length': features.size(0),  # Time frames
            'id': index,
            'path': str(self.audio_paths[index])
        }

        if self.return_labels and self.labels is not None:
            output['label'] = self.labels[index]

        return output