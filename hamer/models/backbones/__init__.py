from .vit import vit
from Dino.hub.backbones import dinov2_vitl14, dinov2_vitb14

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        return vit(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == "dino":
        return dinov2_vitb14()
    else:
        raise NotImplementedError('Backbone type is not implemented')
