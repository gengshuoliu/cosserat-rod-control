"""
1D Fourier Neural Operator (FNO) for the Cosserat rod dynamics surrogate.

Architecture (following Li et al. 2021 "Fourier Neural Operator for PDEs"):

  Input:  (B, C_in,  N)  — N spatial points (rod elements), C_in channels
  Output: (B, C_out, N)

Each FNO layer = SpectralConv1d  (global)  +  pointwise Linear  (local)  + activation.

                 ┌─────────────────────┐
                 │    FNO1d            │
                 │  input projection   │   1×1 Conv: C_in  → hidden
                 │  FNO layers × L     │   each: SpectralConv + local linear + GeLU
                 │  output projection  │   MLP: hidden → hidden → C_out
                 └─────────────────────┘

SpectralConv1d:
  - FFT along spatial dim → complex spectrum  (B, C, N//2+1)
  - Keep first `modes` frequency components; multiply by learnable complex weights
  - iFFT back to spatial domain  (B, C, N)
  - Combined with a local linear branch (1×1 conv, no FFT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Spectral Convolution 1D                                                     #
# --------------------------------------------------------------------------- #

class SpectralConv1d(nn.Module):
    """
    Fourier integral operator on a 1D domain.

    Computes:  (K u)(x) = IFFT[ R(k) * FFT[u](k) ]   for k = 0..modes-1

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    modes        : int  — number of Fourier modes to keep  (<= N//2 + 1)
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes        = modes

        # Learnable complex weights: (out_channels, in_channels, modes)
        # Stored as real and imaginary parts separately for compatibility
        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            scale * torch.randn(out_channels, in_channels, modes, dtype=torch.float32)
        )
        self.weights_imag = nn.Parameter(
            scale * torch.randn(out_channels, in_channels, modes, dtype=torch.float32)
        )

    @property
    def weights(self) -> torch.Tensor:
        """Complex weight tensor: (out_channels, in_channels, modes)."""
        return torch.complex(self.weights_real, self.weights_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C_in, N)

        Returns
        -------
        (B, C_out, N)
        """
        B, C, N = x.shape

        # FFT along spatial dimension
        x_ft = torch.fft.rfft(x, dim=-1)   # (B, C_in, N//2+1)  complex

        # Multiply selected Fourier modes by learnable weights
        # x_ft_modes : (B, C_in, modes)
        # weights    : (C_out, C_in, modes)
        # out_ft     : (B, C_out, modes)
        modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1],
                             dtype=torch.complex64, device=x.device)
        out_ft[:, :, :modes] = torch.einsum(
            "bim,oim->bom", x_ft[:, :, :modes], self.weights[:, :, :modes]
        )

        # iFFT back to spatial domain
        out = torch.fft.irfft(out_ft, n=N, dim=-1)   # (B, C_out, N)
        return out


# --------------------------------------------------------------------------- #
#  Single FNO Block (one layer)                                                #
# --------------------------------------------------------------------------- #

class FNOBlock1d(nn.Module):
    """
    One FNO residual layer:
        h = activation( SpectralConv(x) + W(x) )

    where W is a pointwise linear transform (1×1 conv).
    """

    def __init__(self, channels: int, modes: int, dropout: float = 0.0):
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, modes)
        self.local    = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm     = nn.InstanceNorm1d(channels, affine=True)
        self.act      = nn.GELU()
        self.drop     = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, N)

        Returns
        -------
        (B, C, N)
        """
        return x + self.drop(self.act(self.norm(self.spectral(x) + self.local(x))))


# --------------------------------------------------------------------------- #
#  Full 1D FNO                                                                 #
# --------------------------------------------------------------------------- #

class FNO1d(nn.Module):
    """
    Complete 1D Fourier Neural Operator.

    Architecture:
      1. Input projection:  C_in  -> hidden_channels      (Conv1d 1×1)
      2. FNO layers × n_layers  (FNOBlock1d)
      3. Output head:       hidden -> hidden -> C_out      (MLP pointwise)

    Parameters
    ----------
    in_channels     : int  — input channels  (e.g. 24)
    out_channels    : int  — output channels (e.g. 6)
    modes           : int  — Fourier modes to retain (e.g. 12)
    hidden_channels : int  — hidden dimension in FNO layers (e.g. 64)
    n_layers        : int  — number of FNO blocks (e.g. 4)
    """

    def __init__(self,
                 in_channels:     int = 24,
                 out_channels:    int = 6,
                 modes:           int = 12,
                 hidden_channels: int = 64,
                 n_layers:        int = 4,
                 dropout:         float = 0.0):
        super().__init__()

        self.in_channels     = in_channels
        self.out_channels    = out_channels
        self.modes           = modes
        self.hidden_channels = hidden_channels
        self.n_layers        = n_layers

        # 1. Input projection
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # 2. FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(hidden_channels, modes, dropout=dropout) for _ in range(n_layers)
        ])

        # 3. Output head: pointwise MLP (two linear layers with activation)
        self.output_head = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C_in, N)  — normalised FNO input

        Returns
        -------
        out : (B, C_out, N)  — predicted N_theta (normalised)
        """
        # Input projection
        x = self.input_proj(x)         # (B, hidden, N)

        # FNO blocks
        for block in self.fno_blocks:
            x = block(x)               # (B, hidden, N)

        # Output head
        out = self.output_head(x)      # (B, C_out, N)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"FNO1d(in={self.in_channels}, out={self.out_channels}, "
            f"modes={self.modes}, hidden={self.hidden_channels}, "
            f"layers={self.n_layers}, params={self.count_parameters():,})"
        )
