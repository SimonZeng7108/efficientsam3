# Lazy imports to avoid triggering timm registry conflicts at import time
# Import specific modules only when needed, not with star imports

# For backwards compatibility, we keep the imports available
# but they'll only load when explicitly accessed
def __getattr__(name):
    """Lazy import to avoid registry conflicts."""
    if name in ('RepViT', 'repvit_m0_9', 'repvit_m1_0', 'repvit_m1_1', 'repvit_m1_5', 'repvit_m2_3',
                'RepViTBlock', 'Conv2d_BN', '_make_divisible'):
        from . import repvit
        return getattr(repvit, name)
    elif name in ('TinyViT', 'tiny_vit_5m_224', 'tiny_vit_11m_224', 'tiny_vit_21m_224', 
                  'tiny_vit_21m_384', 'tiny_vit_21m_512'):
        from . import tiny_vit
        return getattr(tiny_vit, name)
    elif name in ('EfficientViTBackbone', 'EfficientViTLargeBackbone', 
                  'efficientvit_backbone_b0', 'efficientvit_backbone_b1', 'efficientvit_backbone_b2',
                  'efficientvit_backbone_l0', 'efficientvit_backbone_l1', 'efficientvit_backbone_l2'):
        from .efficientvit import backbone
        return getattr(backbone, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
