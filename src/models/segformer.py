import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)

from models.segmentation_heads import SegFormerMLPDecoder



### Galeio code with some comments by jv

class SegmentationModelSegformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        size_output: tuple[int, int] = (224, 224),    #jv: A reason for using 224,224? [H,W of the output label img]
        head_fuse_dropout: float = 0.1,
        device: str = "cuda",
    ):
        """
        The SegFormer base model is in MITLicense.
        """
        super().__init__()

        self.size_output = size_output
        # jv: https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512
        # jv: Try b4 or more complex model? 
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b3-finetuned-ade-512-512",
            # ignore_mismatched_sizes=True
        )
        self.processor = AutoImageProcessor.from_pretrained(
            "nvidia/segformer-b3-finetuned-ade-512-512"
        )
        self.device = device
        assert num_classes == 2
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone

        # Freeze the backbone
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Count number of parameters of the backbone
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters of the backbone: {num_params}")

        # Initialize the decoder
        self.decode_head = SegFormerMLPDecoder(
            featuremap_channels_dims=[64, 128, 320, 512],# jv: channel dims of segformer-b2 output feature maps [C1, C2, C3, C4]
            num_classes=num_classes,
            decoder_dim=256, # jv: MLP channel dimension (C parameter in the paper) They use C=256 as their fast model. An higher value likely improve, but more complex model.
            size_output=self.size_output,
            head_fuse_dropout=head_fuse_dropout,
        )
        assert self.decode_head.name_alias == "segformer_mlp_head"
        self.decode_head.to(self.device)

        self.model.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(
            images=x, return_tensors="pt", do_rescale=False
        ).to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        features = outputs.hidden_states # jv: List of feature maps from different stages of the backbone 
        logits = self.decode_head(features)
        return logits
