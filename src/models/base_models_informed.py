# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import torch
from asteroid.utils.torch_utils import pad_x_to_y, jitable_shape
from asteroid.models.base_models import _shape_reconstructed, _unsqueeze_to_3d
from asteroid.models.base_models import BaseEncoderMaskerDecoder

class BaseEncoderMaskerDecoderInformed(BaseEncoderMaskerDecoder):
    """Base class for informed encoder-masker-decoder extraction models.
    Adapted from Asteroid calss BaseEncoderMaskerDecoder
    https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/base_models.py

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masked network.
        decoder (Decoder): Decoder instance.
        auxiliary (nn.Module): auxiliary network processing enrollment.
        encoder_activation (optional[str], optional): activation to apply after encoder.
            see ``asteroid.masknn.activations`` for valid values.
    """
    def __init__(self, encoder, masker, decoder, auxiliary,
                 encoder_activation=None):
        super().__init__(encoder, masker, decoder, encoder_activation)
        self.auxiliary = auxiliary

    def forward(self, wav, enrollment):
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.
            enrollment (torch.Tensor): enrollment information tensor. 1D, 2D or 3D tensor.

        Returns:
            torch.Tensor, of shape (batch, 1, time) or (1, time)
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        tf_rep = self.forward_encoder(wav)
        enroll_emb = self.auxiliary(enrollment)
        est_masks = self.forward_masker(tf_rep, enroll_emb)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)


    def forward_masker(self, tf_rep: torch.Tensor, enroll: torch.Tensor) -> torch.Tensor:
        """Estimates masks from time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq).
            enroll (torch.Tensor): Time-frequency representation in (batch,
                feat, seq).

        Returns:
            torch.Tensor: Estimated masks
        """
        return self.masker(tf_rep, enroll)
