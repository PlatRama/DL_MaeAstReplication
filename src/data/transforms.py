import logging
import torch
import torch.nn as nn
from typing import Optional
import torchaudio
import torchaudio.transforms as T

logger = logging.getLogger('mae_ast')

class MelSpectrogramTransform(nn.Module):
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 400,
                 hop_length: int = 160,
                 n_mels: int = 128,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Default f_max to Nyquist frequency
        if (f_max is None):
            f_max = sample_rate / 2.0

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            norm='slaney',
            mel_scale='slaney'
        )

        logger.info(f"MelSpectrogram: sr={sample_rate}, n_fft={n_fft}, "
                    f"hop={hop_length}, n_mels={n_mels}")


    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
               Convert waveform to log mel spectrogram.

               Args:
                   waveform: [num_samples] or [1, num_samples]

               Returns:
                   log_mel: [time, n_mels] log mel spectrogram
        """
        # Ensure correct shape

        # Ensure correct shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, num_samples]

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)  # [1, n_mels, time]

        # Convert to log scale (with small epsilon for numerical stability)
        # Paper uses log mel filterbank
        log_mel = torch.log(mel_spec + 1e-6)

        # Transpose to [time, n_mels] format
        log_mel = log_mel.unsqueeze(0).transpose(0, 1)  # [time, n_mels]
        return log_mel


class KaldiFbankTransform(nn.Module):
    def __init__(self,
                 sample_rate: int = 16000,
                 n_mels: int = 128,
                 frame_length: float = 25.0,
                 frame_shift: float = 10.):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift

        logger.info(f"KaldiFbank: sr={sample_rate}, n_mels={n_mels}, "
                    f"frame={frame_length}ms, shift={frame_shift}ms")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
               Extract Kaldi fbank features.

               Args:
                   waveform: [num_samples]

               Returns:
                   fbank: [time, n_mels]
               """
        # Kaldi compliance requires [1, num_samples]

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Extract fbank using Kaldi implementation
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=self.sample_rate,
            num_mel_bins=self.n_mels,
            frame_length=self.frame_length,
            frame_shift=self.frame_shift,
            use_energy=False
        )

        return fbank  # [time, n_mels]


class NormalizationTransform(nn.Module):
    """
        Normalize features to specified mean and standard deviation.

        This normalization is applied to the spectrogram features
        BEFORE patchification.

        Args:
            target_mean: Target mean (default: 0.0 as in paper)
            target_std: Target standard deviation (default: 0.5 as in paper)
            eps: Small epsilon for numerical stability
        """

    def __init__(self,
                 target_mean: float = 0.0,
                 target_std: float = 0.5,
                 eps: float = 1e-6):
        super().__init__()

        self.target_mean = target_mean
        self.target_std = target_std
        self.eps = eps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
                Normalize features.

                Args:
                    features: [time, n_mels] or [time, freq]

                Returns:
                    normalized: [time, n_mels] normalized features
                """
        # Compute current statistics
        mean = features.mean()
        std = features.std() + self.eps

        # Normalize to zero mean, unit variance
        normalized = (features - mean) / std
        # Scale to target mean and std
        normalized = normalized * self.target_std + self.target_mean
        return normalized


class AudioPreprocessor(nn.Module):
    """
        Complete audio preprocessing pipeline.

        Pipeline:
        1. Waveform → Log Mel Spectrogram
        2. Normalize to mean=0, std=0.5

        This matches the preprocessing described in paper Section 2.1.

        Args:
            sample_rate: Audio sample rate
            feature_type: 'fbank' or 'melspec'
            n_mels: Number of mel bins
            normalize: Whether to normalize
            use_kaldi: If True, use Kaldi fbank; else use MelSpectrogram
        """

    def __init__(self,
                 sample_rate: int = 16000,
                 feature_type: str = 'fbank',
                 n_mels: int = 128,
                 normalize: bool = True,
                 use_kaldi: bool = True):

        super().__init__()

        self.sample_rate = sample_rate
        self.feature_type = feature_type
        self.n_mels = n_mels
        self.normalize = normalize
        self.use_kaldi = use_kaldi

        if use_kaldi:
            self.feature_extractor = KaldiFbankTransform(
                sample_rate=sample_rate,
                n_mels=n_mels,
                frame_length=25.0,
                frame_shift=10.0,
            )
        else:
            self.feature_extractor = MelSpectrogramTransform(
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=400,
                hop_length=160,
            )

        if normalize:
            self.normalizer = NormalizationTransform(
                target_mean=0.0,
                target_std=0.5)
        else:
            self.normalizer = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
                Preprocess audio waveform.

                Args:
                    waveform: [num_samples] raw audio

                Returns:
                    features: [time, n_mels] preprocessed features
                """
        # Extract features
        features = self.feature_extractor(waveform)

        if self.normalizer is not None:
            features = self.normalizer(features)

        return features
