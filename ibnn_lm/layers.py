# Copyright 2026.
# Apache License 2.0 (matching the upstream vmg-io-csic/ibnn repository this is derived from).
#
# A transformer-oriented re-implementation of the IBNN neuron model from
# Mohedano et al., "Updating the standard neuron model in artificial neural networks" (2026).
#
# The upstream reference code implements the neuron for 2D image convolutions, with the
# lateral interaction realized as a spatial cross-difference convolution. Here we strip the
# model down to its essential math and re-cast it as a fully-connected layer whose lateral
# coupling runs across the HIDDEN CHANNELS of a single position. Applied per token position
# independently, this is automatically causal and parallel across the sequence, which is what
# we want for a language model.
#
# Standard neuron (SM):     u_i = phi( (W x)_i - b_i )
# IBNN neuron (this file):  z_i = (W x)_i - b_i  -  lambda * sum_k w_ik * sigma( z_k - z_i )
#                           v_i = phi( z_i )
# with w_ik uniform (1/D) so the new term adds NO trainable weights, lambda a single
# (optionally trainable) scalar per layer, and sigma(z) = tanh(p z).
#
# z appears on both sides => implicit. We solve it by a damped fixed-point iteration started
# from the SM pre-activation y = Wx - b. num_iters=1 reproduces the cheap "lite" layer
# (purely forward, no solver); num_iters>1 unrolls the solve and is differentiated directly
# by autograd (simple and robust; swap in IFT/torchdeq later if the forward cost matters).

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lateral_term(z: torch.Tensor, p: float) -> torch.Tensor:
    """Mean-field lateral interaction (1/D) * sum_k tanh( p * (z_k - z_i) ).

    z: (..., D). Returns a tensor of the same shape. This is the term that, scaled by
    -lambda, is added to the pre-activation. It is O(D^2) per position; with tanh (an odd
    function) it cannot be factored, so keep D moderate or approximate for very wide layers.
    """
    # pairwise differences d[..., i, k] = z_k - z_i
    diff = z.unsqueeze(-2) - z.unsqueeze(-1)          # (..., D, D), entry [i,k] = z_i - z_k
    coupling = torch.tanh(p * (-diff))                # tanh(p (z_k - z_i))
    return coupling.mean(dim=-1)                      # average over k -> (..., D)


class IBNNLinear(nn.Module):
    """Linear -> IBNN lateral coupling -> activation.

    Drop-in for the "up-projection + activation" half of a transformer FFN. The lateral
    coupling acts over the `out_features` (hidden) dimension, independently per position,
    so causality is preserved with no masking.

    Args:
        in_features, out_features: as in nn.Linear.
        lam: initial lambda. The paper uses small negative values (e.g. -0.01..-0.05);
             lambda <= 0 also guarantees the fixed point exists/is unique and is contractive.
        lam_trainable: if True, lambda is a learned scalar (the paper recommends this).
        p: slope of sigma = tanh(p * .). The paper uses p=10.
        num_iters: fixed-point iterations. 1 == "lite" (forward only). >1 unrolls the solve.
        tau: damping for the fixed-point update in (0, 1].
        activation: phi, applied after the coupling. 'gelu' | 'relu' | 'identity'.
        bias: whether to learn the bias b.
    """

    def __init__(self, in_features, out_features, lam=-0.05, lam_trainable=True,
                 p=10.0, num_iters=1, tau=1.0, activation="gelu", bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = float(p)
        self.num_iters = int(num_iters)
        self.tau = float(tau)
        self.activation = activation

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # single scalar lambda; the new neuron term adds exactly this one parameter per layer
        lam_t = torch.tensor(float(lam))
        self.lam = nn.Parameter(lam_t) if lam_trainable else self.register_buffer("lam", lam_t) or lam_t

    def _phi(self, z):
        if self.activation == "gelu":
            return F.gelu(z)
        if self.activation == "relu":
            return F.relu(z)
        if self.activation == "identity":
            return z
        raise ValueError(f"unknown activation {self.activation}")

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)   # SM pre-activation (..., D)
        z = y
        lam = self.lam
        for _ in range(self.num_iters):
            z = (1.0 - self.tau) * z + self.tau * (y - lam * _lateral_term(z, self.p))
        return self._phi(z)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"num_iters={self.num_iters}, p={self.p}, act={self.activation}")


class IBNNMLP(nn.Module):
    """Transformer FFN block with the IBNN neuron on the hidden layer.

    x -> IBNNLinear(d_model -> d_ff) -> Linear(d_ff -> d_model) -> dropout
    The down-projection is an ordinary linear, exactly as in a standard transformer; only the
    hidden units carry the implicit-bias lateral coupling.
    """

    def __init__(self, d_model, d_ff=None, dropout=0.0, **ibnn_kwargs):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.up = IBNNLinear(d_model, d_ff, **ibnn_kwargs)
        self.down = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(self.up(x)))
