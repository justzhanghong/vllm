"""Covariance helpers for shared-latent OSCAR calibration."""

import math

import torch


class CovarianceAccumulator:
    """Accumulate an uncentered second moment in FP64 on CPU."""

    def __init__(self, latent_rank: int) -> None:
        if latent_rank <= 0:
            raise ValueError(f"latent_rank must be positive, got {latent_rank}")
        self.latent_rank = latent_rank
        self.samples = 0
        self._second_moment = torch.zeros(
            latent_rank,
            latent_rank,
            dtype=torch.float64,
        )

    def update(self, values: torch.Tensor) -> None:
        """Add all leading-dimension rows from a tensor."""
        if values.shape[-1] != self.latent_rank:
            raise ValueError(
                f"expected latent rank {self.latent_rank}, got {values.shape[-1]}"
            )
        rows = values.detach().reshape(-1, self.latent_rank).double().cpu()
        self._second_moment.addmm_(rows.T, rows)
        self.samples += rows.shape[0]

    def covariance(self) -> torch.Tensor:
        """Return the symmetric uncentered covariance."""
        if self.samples == 0:
            raise RuntimeError("cannot compute covariance without samples")
        covariance = self._second_moment / self.samples
        return (covariance + covariance.T) / 2


def normalize_covariance(covariance: torch.Tensor) -> torch.Tensor:
    """Normalize a positive covariance by trace."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"covariance must be square, got {covariance.shape}")
    symmetric = (covariance.double() + covariance.double().T) / 2
    trace = symmetric.trace()
    if not bool(torch.isfinite(trace)) or trace <= 0:
        raise ValueError(f"covariance trace must be finite and positive, got {trace}")
    return symmetric / trace


def build_shared_covariance(
    score_covariance: torch.Tensor,
    value_covariance: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Combine trace-normalized score and value covariances."""
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    score = normalize_covariance(score_covariance)
    value = normalize_covariance(value_covariance)
    if score.shape != value.shape:
        raise ValueError(
            f"score/value covariance shapes differ: {score.shape} != {value.shape}"
        )
    shared = alpha * score + (1 - alpha) * value
    return (shared + shared.T) / 2


def covariance_rotation(covariance: torch.Tensor) -> torch.Tensor:
    """Return a descending-eigenvalue orthogonal basis for a covariance."""
    symmetric = normalize_covariance(covariance)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvectors[:, order]


def normalized_hadamard(rank: int) -> torch.Tensor:
    """Build a normalized Sylvester Hadamard matrix in FP64."""
    if rank <= 0 or rank & (rank - 1):
        raise ValueError(f"Hadamard rank must be a power of two, got {rank}")
    matrix = torch.ones(1, 1, dtype=torch.float64)
    while matrix.shape[0] < rank:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(rank)


def bit_reversal_permutation(rank: int) -> torch.Tensor:
    """Return the bit-reversed indices for a power-of-two latent rank."""
    if rank <= 0 or rank & (rank - 1):
        raise ValueError(f"bit-reversal rank must be a power of two, got {rank}")
    bits = int(math.log2(rank))
    return torch.tensor(
        [int(f"{index:0{bits}b}"[::-1], 2) for index in range(rank)],
        dtype=torch.int64,
    )


def oscar_covariance_rotation(covariance: torch.Tensor) -> torch.Tensor:
    """Compose the shared eigenbasis with normalized Hadamard and PBR."""
    eigenbasis = covariance_rotation(covariance)
    rank = eigenbasis.shape[0]
    hadamard = normalized_hadamard(rank)
    permutation = torch.eye(rank, dtype=torch.float64)[
        :, bit_reversal_permutation(rank)
    ]
    return eigenbasis @ hadamard @ permutation
