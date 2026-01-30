import logging
from typing import List, Dict, Optional

import torch

logger = logging.getLogger('mae_ast')

class AudioCollator:
    """
    Collate audio samples into batches.

    This collator:
    1. Pads all samples to the same length (longest in batch)
    2. Creates padding masks (True = padding, False = valid)
    3. Stacks into batch tensors

    Paper Context:
    - Input spectrograms are variable length in time dimension
    - Padding masks are used in Transformer attention to ignore padding
    - Positional embeddings are only applied to non-padded positions

    Args:
        sample_rate: Audio sample rate (for logging)
        max_length: Maximum sequence length (in feature frames)
        pad_to_multiple: Pad to multiple of this value (useful for efficient computation)
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 max_length: Optional[int] = None,
                 pad_to_multiple: int = 1):
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.pad_to_multiple = pad_to_multiple

    def __call__(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples into a batch.

        Args:
            samples: List of dicts from AudioDataset.__getitem__()
                Each dict has keys: 'audio', 'audio_length', 'id', 'path'

        Returns:
            Batch dict with keys:
                - 'audio': [batch, max_time, n_mels] padded features
                - 'padding_mask': [batch, max_time] True=padding, False=valid
                - 'audio_lengths': [batch] original lengths before padding
                - 'ids': [batch] sample indices
                - 'labels': [batch] labels (if present in samples)
        """

        if len(samples) == 0:
            return {}

        # Extract components
        audios = [s['audio'] for s in samples] # list of [time, n_mels]
        audio_lengths = [s['audio_length'] for s in samples]
        ids = [s['id'] for s in samples]

        # Check if labels are present
        has_labels = 'label' in samples[0]
        if has_labels:
            labels = [s['label'] for s in samples]

        # Determine batch max length
        max_len = max(audio_lengths)

        # Apply max_length constraint if specified
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)

        # Pad to multiple if specified
        # This can improve computational efficiency on some hardware
        if self.pad_to_multiple > 1:
            max_len = ((max_len + self.pad_to_multiple - 1)
                       // self.pad_to_multiple * self.pad_to_multiple)

        batch_size = len(samples)
        n_mels = audios[0].size(1)

        # Allocate batch tensors
        batch_audio = torch.zeros(batch_size, max_len, n_mels, dtype=torch.float32)
        padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)

        # fill batch
        for i, (audio, length) in enumerate(zip(audios, audio_lengths)):
            length = min(length, max_len)
            # Copy audio data
            batch_audio[i, :length] = audio[:length]
            # Mark valid positions in padding mask
            # Paper: Padding mask is used in Transformer attention
            # True = padding (ignore), False = valid (attend)
            padding_mask[i, :length] = False

        # Prepare output
        output = {
            'audio': batch_audio,
            'padding_mask': padding_mask,
            'audio_lengths': torch.LongTensor(audio_lengths),
            'ids': torch.LongTensor(ids),
        }

        # Add labels if present
        if has_labels:
            output['labels'] = torch.LongTensor(labels)

        return output
