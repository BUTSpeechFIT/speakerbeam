# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import torch
import torch.nn as nn
from asteroid_filterbanks import make_enc_dec
from asteroid.masknn.convolutional import TDConvNet
from models.base_models_informed import BaseEncoderMaskerDecoderInformed
from models.adapt_layers import make_adapt_layer

class Lambda(nn.Module):
    """
    https://stackoverflow.com/a/64064088
    Input: A Function
    Returns : A Module that can be used
        inside nn.Sequential
    """
    def __init__(self, func, **kwargs):
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def forward(self, x): return self.func(x, **self.kwargs)

class TimeDomainSpeakerBeam(BaseEncoderMaskerDecoderInformed):
    """TimeDomain SpeakerBeam target speech extraction model.
    Adapted from Asteroid class ConvTasnet 
    https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/conv_tasnet.py

    Args:
        i_adapt_layer (int): Index of adaptation layer.
        adapt_layer_type (str): Type of adaptation layer, see adapt_layers.py for options.
        adapt_enroll_dim (int): Dimensionality of the speaker embedding.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
            If 0 or None, TDConvNet won't have any skip connections and the
            masks will be computed from the residual output.
            Corresponds to the ConvTasnet architecture in v1 or the paper.
        conv_kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        causal (bool, optional) : Whether or not the convolutions are causal.
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sampling rate of the model.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.
    """

    def __init__(
        self,
        i_adapt_layer,
        adapt_layer_type,
        adapt_enroll_dim,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="sigmoid",
        in_chan=None,
        causal=False,
        fb_name="free",
        kernel_size=16,
        n_filters=512,
        stride=8,
        encoder_activation=None,
        sample_rate=8000,
        **fb_kwargs,
    ):
        encoder, decoder = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )

        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        masker = TDConvNetInformed(
            n_feats,
            i_adapt_layer,
            adapt_layer_type,
            adapt_enroll_dim,
            out_chan=out_chan,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            skip_chan=skip_chan,
            conv_kernel_size=conv_kernel_size,
            norm_type=norm_type,
            mask_act=mask_act,
            causal=causal,
        )

        # Encoder for auxiliary network
        encoder_aux, _ = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            **fb_kwargs,
        )
        # Auxiliary network
        auxiliary = nn.Sequential(
            Lambda(torch.unsqueeze, dim=1),
            encoder_aux,
            TDConvNet(
                n_feats,
                n_src=1,
                out_chan=adapt_enroll_dim*2 if skip_chan else adapt_enroll_dim,
                n_blocks=n_blocks,
                n_repeats=1,
                bn_chan=bn_chan,
                hid_chan=hid_chan,
                skip_chan=skip_chan,
                conv_kernel_size=conv_kernel_size,
                norm_type=norm_type,
                mask_act='linear',
                causal=False
            ),
            Lambda(torch.mean, dim=-1),
            Lambda(torch.squeeze, dim=1)
        )

        super().__init__(encoder, masker, decoder, auxiliary,
                         encoder_activation=encoder_activation)

class TDConvNetInformed(TDConvNet):
    """
    Adapted from Asteroid class TDConvNet 
    https://github.com/asteroid-team/asteroid/blob/master/asteroid/masknn/convolutional.py
    """
    def __init__(
        self,
        in_chan,
        i_adapt_layer,
        adapt_layer_type,
        adapt_enroll_dim,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="relu",
        causal=False,
        **adapt_layer_kwargs
    ):
        super(TDConvNetInformed, self).__init__(
                in_chan, 1, out_chan, n_blocks, n_repeats, 
                bn_chan, hid_chan, skip_chan, conv_kernel_size,
                norm_type, mask_act, causal)
        self.i_adapt_layer = i_adapt_layer
        self.adapt_enroll_dim = adapt_enroll_dim
        self.adapt_layer_type = adapt_layer_type
        self.adapt_layer = make_adapt_layer(adapt_layer_type, 
                                            indim=bn_chan,
                                            enrolldim=adapt_enroll_dim,
                                            ninputs=2 if self.skip_chan else 1,
                                            **adapt_layer_kwargs)

    def forward(self, mixture_w, enroll_emb):
        r"""Forward with auxiliary enrollment information
        
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$
            enroll_emb (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$
                                                or $(batch, nfilters)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, 1, nfilters, nframes)$
        """
        batch, _, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = torch.tensor([0.0], device=output.device)
        for i, layer in enumerate(self.TCN):
            # Common to w. skip and w.o skip architectures
            tcn_out = layer(output)
            if self.skip_chan:
                residual, skip = tcn_out
                if i == self.i_adapt_layer:
                    residual, skip = self.adapt_layer((residual, skip), 
                                            torch.chunk(enroll_emb,2,dim=1))
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
                if i == self.i_adapt_layer:
                    residual = self.adapt_layer(residual, enroll_emb)
            output = output + residual
        # Use residual output when no skip connection
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, 1, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask

    def get_config(self):
        config = super().get_config()
        config.update({
            'i_adapt_layer': self.i_adapt_layer,
            'adapt_layer_type': self.adapt_layer_type,
            'adapt_enroll_dim': self.adapt_enroll_dim
            })
        return config
