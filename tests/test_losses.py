import torch

from spkclone.losses import supcon_loss


def clustered(n_spk=3, k=2, dim=8):
    labels = torch.arange(n_spk).repeat_interleave(k)
    return torch.eye(dim)[labels], labels


def test_perfect_clusters_give_near_zero_loss():
    emb, labels = clustered()
    assert supcon_loss(emb, labels, T=0.05).item() < 1e-3


def test_random_embeddings_have_higher_loss():
    torch.manual_seed(0)
    emb, labels = clustered()
    assert supcon_loss(torch.randn(6, 8), labels).item() > supcon_loss(emb, labels).item()


def test_no_positives_gives_zero():
    assert supcon_loss(torch.randn(4, 8), torch.arange(4)).item() == 0.0


def test_scale_invariant():
    torch.manual_seed(1)
    emb, labels = torch.randn(8, 16), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    assert torch.allclose(supcon_loss(emb, labels), supcon_loss(5 * emb, labels), atol=1e-5)


def test_gradient_is_finite():
    emb = torch.randn(8, 16, requires_grad=True)
    supcon_loss(emb, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])).backward()
    assert torch.isfinite(emb.grad).all()
