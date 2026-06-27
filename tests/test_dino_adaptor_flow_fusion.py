import torch

from models.spatial_models.frame_models.dino_adaptor_model import GatedFusion


def test_flow_fusion_prefers_rgb_initially():
    fusion = GatedFusion(dim=8)
    rgb = torch.ones(2, 8)
    flow = torch.zeros(2, 8)

    fused = fusion(rgb, flow)

    # The gate should favor RGB at initialization so the pretrained backbone
    # is not overwhelmed by the freshly initialized flow branch.
    assert torch.all(fused > 0.5 * rgb)
