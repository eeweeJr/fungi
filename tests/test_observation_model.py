"""Tests for the observation model: backbone wrapper, fusion, heads, losses, invariances."""

import pytest
import torch
from torch import nn

from fungai.model.backbone import TokenBackbone, layer_decay_scale
from fungai.model.hierarchy import (
    FlatHead,
    HierarchicalHead,
    build_species_genus_index,
    hierarchical_logp,
)
from fungai.model.losses import observation_loss
from fungai.model.observation import ObservationClassifier

SPECIES_GENUS = torch.tensor([0, 0, 0, 1, 1, 2])  # 6 species in 3 genera
N_GENUS = 3


class StubBackbone(nn.Module):
    """Photo-wise toy backbone: a conv patch embedding plus one token mixer.
    Same interface as TokenBackbone."""

    def __init__(self, dim=24, patch=8):
        super().__init__()
        self.embed_dim, self.patch_size, self.num_prefix_tokens, self.n_blocks = dim, patch, 1, 2
        self.embed = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(dim, 4, 48, batch_first=True,
                                                                dropout=0.0) for _ in range(2)])

    def forward(self, x):
        t = self.embed(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], dim=1)
        for blk in self.blocks:
            t = blk(t)
        return t[:, 0], t[:, 1:]


def _model(fusion="perceiver", head="hierarchical", seed=0):
    torch.manual_seed(seed)
    return ObservationClassifier(StubBackbone(), SPECIES_GENUS, N_GENUS, fusion=fusion, head=head,
                                 fusion_dim=32, n_queries=4, fusion_blocks=2, fusion_heads=4,
                                 fusion_dropout=0.1, photo_dropout=0.1).eval()


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(3, 4, 3, 32, 32, generator=g)
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    images[~mask] = 0
    return images, mask


# --------------------------------------------------------------------------- backbone

@pytest.mark.parametrize("name,prefix,dim", [
    ("vit_small_patch16_dinov3.lvd1689m", 5, 384),        # class + 4 registers, RoPE
    ("vit_small_patch14_reg4_dinov2.lvd142m", 5, 384),    # class + 4 registers, patch 14
    ("vit_base_patch16_224", 1, 768),                     # plain ViT, fixed pos embedding
])
def test_token_backbone_shapes(name, prefix, dim):
    size = 112
    bb = TokenBackbone(name, pretrained=False, image_size=size).eval()
    assert bb.num_prefix_tokens == prefix and bb.embed_dim == dim
    with torch.no_grad():
        cls, patches = bb(torch.randn(2, 3, size, size))
    assert cls.shape == (2, dim)
    assert patches.shape == (2, (size // bb.patch_size) ** 2, dim)
    groups = dict(bb.layer_groups())
    assert list(groups)[0] == "embed" and list(groups)[-1] == "top"
    n_grouped = sum(len(v) for v in groups.values())
    assert n_grouped == len(list(bb.parameters()))


def test_layer_decay_scale():
    assert layer_decay_scale("top", 12, 0.75) == 1.0
    assert layer_decay_scale("block_11", 12, 0.75) == 1.0
    assert layer_decay_scale("block_10", 12, 0.75) == pytest.approx(0.75)
    assert layer_decay_scale("embed", 12, 0.75) == pytest.approx(0.75 ** 12)


# --------------------------------------------------------------------------- heads

def test_species_genus_index_and_empty_genus():
    idx = build_species_genus_index({"a1": 0, "a2": 1, "b1": 2}, {"a1": "A", "a2": "A", "b1": "B"},
                                    {"A": 0, "B": 1})
    assert idx.tolist() == [0, 0, 1]
    with pytest.raises(ValueError, match="no known species"):
        build_species_genus_index({"a1": 0}, {"a1": "A"}, {"A": 0, "B": 1})


@pytest.mark.parametrize("head_cls", [HierarchicalHead, FlatHead])
def test_head_probabilities_are_consistent(head_cls):
    torch.manual_seed(0)
    head = head_cls(16, SPECIES_GENUS, N_GENUS)
    out = head(torch.randn(5, 16))
    p_species = out["species_logp"].exp()
    assert torch.allclose(p_species.sum(-1), torch.ones(5), atol=1e-5)
    for g in range(N_GENUS):
        members = SPECIES_GENUS == g
        assert torch.allclose(p_species[:, members].sum(-1), out["genus_logp"][:, g].exp(), atol=1e-5)
        assert torch.allclose(out["species_cond_logp"][:, members].exp().sum(-1), torch.ones(5),
                              atol=1e-5)


def test_hierarchical_temperatures_act_on_their_own_level():
    torch.manual_seed(0)
    g, s = torch.randn(4, N_GENUS), torch.randn(4, 6)
    base = hierarchical_logp(g, s, SPECIES_GENUS, N_GENUS)
    hot = hierarchical_logp(g, s, SPECIES_GENUS, N_GENUS, t_genus=3.0)
    assert torch.allclose(base["species_cond_logp"], hot["species_cond_logp"])
    assert not torch.allclose(base["genus_logp"], hot["genus_logp"])


# --------------------------------------------------------------------------- model

@pytest.mark.parametrize("fusion,head", [("perceiver", "hierarchical"), ("perceiver", "flat"),
                                         ("none", "hierarchical")])
def test_output_contract(fusion, head):
    model = _model(fusion, head)
    images, mask = _batch()
    out = model(images, mask, return_attention=True)
    assert out["species_logp"].shape == (3, 6)
    assert out["genus_logp"].shape == (3, N_GENUS)
    assert out["species_cond_logp"].shape == (3, 6)
    assert out["photo_logits"].shape == (int(mask.sum()), 6)
    assert out["photo_obs_index"].tolist() == [0, 0, 0, 1, 2, 2, 2, 2]
    assert torch.allclose(out["species_logp"].exp().sum(-1), torch.ones(3), atol=1e-5)
    if fusion == "perceiver":
        assert out["obs_emb"].shape == (3, 32)
        assert len(out["attn"]) == 2
        assert out["attn"][0].shape == (3, 4, 4, 4 * 16)   # [B, heads, K, P*T]


def test_permutation_invariance():
    model = _model()
    images, mask = _batch()
    ref = model(images, mask)["species_logp"]
    perm_images, perm_mask = images.clone(), mask.clone()
    perm_images[0, :3] = images[0, [2, 0, 1]]
    perm_images[2] = images[2, [3, 1, 0, 2]]
    out = model(perm_images, perm_mask)["species_logp"]
    assert (out - ref).abs().max() < 1e-5


def test_extra_padding_slot_changes_nothing():
    model = _model()
    images, mask = _batch()
    ref = model(images, mask)["species_logp"]
    images5 = torch.cat([images, torch.randn(3, 1, 3, 32, 32)], dim=1)  # junk in the pad slot
    mask5 = torch.cat([mask, torch.zeros(3, 1, dtype=torch.bool)], dim=1)
    out = model(images5, mask5)["species_logp"]
    assert (out - ref).abs().max() < 1e-5


def test_attention_on_padding_is_exactly_zero():
    model = _model()
    images, mask = _batch()
    out = model(images, mask, return_attention=True)
    key_mask = out["key_mask"]
    for w in out["attn"]:
        assert w[~key_mask[:, None, None, :].expand_as(w)].abs().max() == 0
        assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)


def test_explicit_and_fused_attention_paths_agree():
    model = _model()
    images, mask = _batch()
    a = model(images, mask, return_attention=False)["species_logp"]
    b = model(images, mask, return_attention=True)["species_logp"]
    assert (a - b).abs().max() < 1e-5


def test_single_photo_observations():
    model = _model()
    images = torch.randn(2, 1, 3, 32, 32)
    out = model(images, torch.ones(2, 1, dtype=torch.bool))
    assert out["species_logp"].shape == (2, 6)


def test_fusion_none_matches_photo_average():
    model = _model("none")
    images, mask = _batch()
    out = model(images, mask)
    assert torch.allclose(out["species_logp"], model.photo_average(out, 3), atol=1e-5)


def test_freeze_backbone():
    model = _model().train()
    model.freeze_backbone(True)
    assert not model.backbone.training and model.training
    images, mask = _batch()
    out = model(images, mask)
    loss = observation_loss(out, torch.tensor([0, 3, 5]), torch.tensor([0, 1, 2]),
                            "hierarchical", "perceiver", 0.3)["total"]
    loss.backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.fusion.queries.grad is not None
    model.freeze_backbone(False)
    assert model.backbone.training
    assert all(p.requires_grad for p in model.backbone.parameters())


# --------------------------------------------------------------------------- losses

def test_hierarchical_loss_equals_joint_nll_for_known_species():
    model = _model()
    images, mask = _batch()
    out = model(images, mask)
    label, genus = torch.tensor([0, 3, 5]), torch.tensor([0, 1, 2])
    parts = observation_loss(out, label, genus, "hierarchical", "perceiver", 0.0)
    assert parts["total"].item() == pytest.approx(parts["species"].item(), rel=1e-5)


def test_loss_terms_and_held_out_rows():
    model = _model()
    images, mask = _batch()
    out = model(images, mask)
    label, genus = torch.tensor([0, -1, 5]), torch.tensor([0, 1, 2])  # obs 1 is a held-out species
    parts = observation_loss(out, label, genus, "hierarchical", "perceiver", 0.3)
    for k in ("total", "genus", "species_cond", "species", "photo"):
        assert torch.isfinite(parts[k])
    expected = parts["genus"] + parts["species_cond"] + 0.3 * parts["photo"]
    assert parts["total"].item() == pytest.approx(expected.item(), rel=1e-5)
