
import logging
import random
from typing import Tuple, Optional

import torch

logger = logging.getLogger('mae_ast')

class MaskGenerator:
    """
    Base class for masking strategies.
    """
    def __init__(self, mask_ratio: float = 0.75):
        self.mask_ratio = mask_ratio
        logger.info(f"MaskGenerator initialized with mask_ratio={mask_ratio}")

    def generate_mask(self,
                      batch_size: int,
                      num_tokens: int,
                      device: torch.device
                      ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Generate mask indices.

        Args:
            batch_size: Batch size
            num_tokens: Total number of tokens per sample
            device: Device for tensors

        Returns:
            retained_indices: List of arrays with indices to keep
            masked_indices: List of arrays with indices to mask
            num_masked: Number of masked tokens
        """
        raise NotImplementedError("Subclasses must implement generate_mask()")


class RandomMasking(MaskGenerator):
    """
       Fully random masking.

       This is the simplest masking strategy:
       - Shuffle all token indices randomly
       - Keep first (1 - mask_ratio) tokens
       - Mask remaining tokens

       Args:
           mask_ratio: Proportion to mask (default: 0.75)
           batched: If True, use same mask for entire batch
       """

    def __init__(self, mask_ratio: float = 0.75, batched: bool = False):
        super().__init__(mask_ratio)
        self.batched = batched
        logger.info(f"RandomMasking: batched={batched}")

    def generate_mask(self,
                      batch_size: int,
                      num_tokens: int,
                      device: torch.device
                      ) -> Tuple[torch.Tensor, torch.Tensor, int]:

        num_retained = max(1, int((1 - self.mask_ratio) * num_tokens))
        num_masked = num_tokens - num_retained

        if self.batched:
            # Same mask for entire batch
            perm = torch.randperm(num_tokens, device=device)
            retained_idx = perm[:num_retained].sort()[0]
            masked_idx = perm[num_retained:].sort()[0]

            retained_indices = retained_idx.unsqueeze(0).expand(batch_size, -1)
            masked_indices = masked_idx.unsqueeze(0).expand(batch_size, -1)
        else:
            # Different mask for each sample
            retained_indices = torch.zeros(batch_size, num_retained, dtype=torch.long, device=device)
            masked_indices = torch.zeros(batch_size, num_masked, dtype=torch.long, device=device)

            for i in range(batch_size):
                perm = torch.randperm(num_tokens, device=device)
                retained_indices[i] = perm[:num_retained].sort()[0]
                masked_indices[i] = perm[num_retained:].sort()[0]

        return retained_indices, masked_indices, num_masked


class ChunkedMasking(MaskGenerator):
    """
    Chunked masking

    This creates contiguous rectangular regions of masked tokens,
    making the task harder (can't simply interpolate from neighbors).

    Args:
        mask_ratio: Proportion to mask
        chunk_size_range: Tuple of (min, max) chunk sizes
        num_freq_patches: Number of patches in frequency dimension
        batched: If True, same mask for entire batch
    """
    def __init__(self,
                 mask_ratio: float = 0.75,
                 chunk_size_range: Tuple[int, int] = (3, 5),
                 num_freq_patches: int = 8,
                 batched: bool = False):
        super().__init__(mask_ratio)
        self.chunk_size_range = chunk_size_range
        self.num_freq_patches = num_freq_patches
        self.batched = batched
        logger.info(f"ChunkedMasking: chunk_range={chunk_size_range}, "
                    f"freq_patches={num_freq_patches}, batched={batched}")


    def generate_mask(self,
                      batch_size: int,
                      num_tokens: int,
                      device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Generate chunked mask.

        Algorithm:
        1. Randomly select chunk size C ∈ [3, 5]
        2. Randomly select top-left corner
        3. Mask C×C region (handling 2D structure of patches)
        4. Repeat until mask_ratio is reached

        Paper uses 2D patch layout:
            Frequency dim: 8 patches (128 mels / 16)
            Time dim: variable (depends on audio length)
        """

        num_retained = max(1, int((1 - self.mask_ratio) * num_tokens))
        num_masked_target = num_tokens - num_retained

        num_time_patches = num_tokens // self.num_freq_patches

        num_masks_to_generate = 1 if self.batched else batch_size
        all_masked_indices = []
        all_retained_indices = []
        for _ in range(num_masks_to_generate):
            masked_set = set()

            while len(masked_set) < num_masked_target:
                chunk_size = random.randint(*self.chunk_size_range)
                start_time = random.randint(0, num_time_patches - 1)
                start_freq = random.randint(0, self.num_freq_patches - 1)

                for t_offset in range(chunk_size):
                    for f_offset in range(chunk_size):
                        time_idx = start_time + t_offset
                        freq_idx = start_freq + f_offset

                        if time_idx < num_time_patches and freq_idx < self.num_freq_patches:
                            token_idx = freq_idx * num_time_patches + time_idx

                            if token_idx < num_tokens:
                                masked_set.add(token_idx)

                if len(masked_set) >= num_masked_target:
                    break

            masked_idx = sorted(list(masked_set))[:num_masked_target]

            all_indices = set(range(num_tokens))
            retained_idx = sorted(all_indices - set(masked_idx))

            all_masked_indices.append(masked_idx)
            all_retained_indices.append(retained_idx)


        if self.batched:
            # Same mask for all samples
            retained_tensor = torch.tensor(all_retained_indices[0], dtype=torch.long, device=device)
            masked_tensor = torch.tensor(all_masked_indices[0], dtype=torch.long, device=device)

            retained_indices = retained_tensor.unsqueeze(0).expand(batch_size, -1)
            masked_indices = masked_tensor.unsqueeze(0).expand(batch_size, -1)
        else:
            # Different mask per sample
            retained_indices = torch.zeros(batch_size, len(all_retained_indices[0]),
                                           dtype=torch.long, device=device)
            masked_indices = torch.zeros(batch_size, len(all_masked_indices[0]),
                                         dtype=torch.long, device=device)

            for i in range(batch_size):
                retained_indices[i] = torch.tensor(all_retained_indices[i], dtype=torch.long, device=device)
                masked_indices[i] = torch.tensor(all_masked_indices[i], dtype=torch.long, device=device)

        return retained_indices, masked_indices, num_masked_target