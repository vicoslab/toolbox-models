"""Semantic head using the published STEM FPN without upstream changes."""
from models.FPN import FPN

def build_semantic_fpn(
    num_classes,
    backbone="tu-convnext_base",
    pretrained=True,
    in_channels=3,
    encoder_depth=4,
    upsampling=4,
    decoder_segmentation_head_channels=64,
    checkpoint_encoder_features=False,
):
    if int(num_classes) < 2:
        raise ValueError("semantic FPN requires at least two classes")
    model = FPN(
        num_classes=[int(num_classes)],
        backbone=backbone,
        use_custom_fpn=True,
        add_output_exp=False,
        init_decoder_gain=0.1,
        pretrained=bool(pretrained),
        in_channels=int(in_channels),
        fpn_args={
            "encoder_depth": int(encoder_depth),
            "upsampling": int(upsampling),
            "decoder_segmentation_head_channels": int(
                decoder_segmentation_head_channels
            ),
            "classes_grouping": [tuple(range(int(num_classes)))],
            "checkpoint_encoder_features": bool(checkpoint_encoder_features),
        },
    )
    model.init_output(num_vector_fields=0)
    return model
