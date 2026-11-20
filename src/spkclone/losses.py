"""Supervised contrastive loss (Khosla et al., 2020, the "L_out" form)."""
import torch
import torch.nn.functional as F


def supcon_loss(emb, labels, T=0.1):
    """emb: (N, D) embeddings, labels: (N,) speaker ids, T: temperature.

    For each anchor i, the loss is the mean over its positives p (same label, p != i) of
    -log softmax_j(sim(i, j) / T)[p], where j runs over all samples except i.
    Anchors without any positive contribute 0 (they cannot be avoided if a batch has K=1).
    """
    z = F.normalize(emb.float(), dim=-1)
    sim = z @ z.T / T
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)  # exclude self-similarity from the softmax
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None]).float().masked_fill(eye, 0)
    return (-(pos * logp).sum(1) / pos.sum(1).clamp(min=1)).mean()
