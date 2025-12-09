import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerMLPDecoder(nn.Module):
    """
    Given a list of feature maps or varying dimensions (B, C, H, W):
    - First upsample each of them to the highest resolution 
    - Then fuse them together via a conv layer.
    """

    # What is the embedding dimension ?
    def __init__(
        self,
        featuremap_channels_dims: list[int],
        num_classes: int = 2,
        decoder_dim: int = 256, # jv: MLP channel dimension (C parameter in the paper) They use C=256 as their fast model. An higher value likely improve, but more complex model.
        size_output: tuple[int, int] = (224, 224), # jv: Final output size (H, W of the image) Is there a reason to have 224x224 as default ?
        head_fuse_dropout: float = 0.1,
    ):
        """
        For Segformer-b2, the list of input features is of the shape:
            (B, 64, H/4, W/4),
            (B, 128, H/8, H/8),
            (B, 320, H/16, W/16),
            (B, 512, H/32, W/32)
        e.g for an input image of size 512x512:
            (B, 64, 128, 128),
            (B, 128, 64, 64),
            (B, 320, 32, 32),
            (B, 512, 16, 16)

        B: batch size
        _______________________


        So the linear_layers first give them the same number of channels (decoder_dim), each feature map
        is also upsampled to highest common resolution and then fused with self.fuse module.

        Args:
            featuremap_channels_dims: list[int]
            decoder_dim: Channel dimension at which to fuse the feature maps
            num_classes: int
            head_fuse_dropout: float = 0.1, the dropout applied in the fusing module
        """
        super().__init__()

        self.name_alias = "segformer_mlp_head"
        self.decoder_dim = decoder_dim
        self.featuremap_channels_dims = featuremap_channels_dims
        self.num_classes = num_classes
        self.size_output = size_output
        self.head_fuse_dropout = head_fuse_dropout

        # TODO: Mais est-ce qu'on ne peut pas d'emblee les convertir a la dimension 4 le nombre de channels ?
        
        ## jv: MLP layer (1x1 conv) to unify the channel dimension to decoder_dim
        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(nbr_channels, decoder_dim, kernel_size=1),
                    nn.BatchNorm2d(decoder_dim),
                    nn.ReLU(inplace=True),
                    # nn.GeLU(inplace=True),# jv: Segformer uses GeLU?
                )
                for nbr_channels in featuremap_channels_dims
            ]
        )

        # jv: Fusion MLP (1x1 conv)
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(head_fuse_dropout),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1),
        )

        self.last_upsampling = nn.Upsample(
            size=size_output, mode="bilinear", align_corners=False
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: list[torch.Tensor] list of 4 tensors [B, C_feature, H, W]
        """
        # TODO: could make more check about features.
        assert len(features) == 4 and features[0].ndim == 4
        target_size = features[0].shape[2:]  # The highest resolution (C1)
        upsampled = []
        for i, layer in enumerate(self.linear_layers):
            feat = layer(features[i])
            # TODO: this interpolate module could be integrated in the linear_layers ...
            feat = F.interpolate(
                feat, size=target_size, mode="bilinear", align_corners=False
            )
            upsampled.append(feat)
        fused = torch.cat(upsampled, dim=1)
        fused = self.fuse(fused)
        logits = self.last_upsampling(fused)
        assert logits.shape[2:] == self.size_output
        return logits


# class OceanSARMLPDecoder(nn.Module):
#     def __init__(
#         self,
#         featuremap_channels_dims: list[int],  # e.g. [384, 384, 384, 384]
#         num_classes: int = 2,
#         decoder_dim: int = 256,
#         size_output: tuple[int, int] = (256, 256),
#         head_fuse_dropout: float = 0.1,
#     ):
#         super().__init__()

#         self.name_alias = "oceansar_mlp_head"
#         self.size_output = size_output

#         # Convert each feature map to decoder_dim channels
#         self.linear_layers = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.Conv2d(ch, decoder_dim, kernel_size=1),
#                     nn.BatchNorm2d(decoder_dim),
#                     nn.ReLU(inplace=True),
#                 )
#                 for ch in featuremap_channels_dims
#             ]
#         )

#         # Fuse them all together
#         self.fuse = nn.Sequential(
#             nn.Conv2d(
#                 decoder_dim * len(featuremap_channels_dims), decoder_dim, kernel_size=1
#             ),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(head_fuse_dropout),
#             nn.Conv2d(decoder_dim, num_classes, kernel_size=1),
#         )

#         # Final resize to desired output size
#         self.last_upsampling = nn.Upsample(
#             size=size_output, mode="bilinear", align_corners=False
#         )

#     def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
#         """
#         Args:
#             features: List of tensors with shape [B, C, 32, 32]
#         """
#         assert len(features) == len(self.linear_layers)
#         assert all(f.shape[2:] == features[0].shape[2:] for f in features), (
#             "Spatial sizes must match"
#         )

#         # Apply 1x1 convs
#         projected = [layer(f) for layer, f in zip(self.linear_layers, features)]

#         # Concatenate and fuse
#         fused = torch.cat(projected, dim=1)  # shape: [B, decoder_dim * N, 32, 32]
#         out = self.fuse(fused)

#         # Optional: upsample to final size
#         out = self.last_upsampling(out)

#         return out


class ViTMLPDecoder(nn.Module):
    def __init__(
        self,
        featuremap_channels_dims: list[int],  # e.g. [786, 786, 786, 786]
        num_classes: int = 2,
        decoder_dim: int = 256,
        size_output: tuple[int, int] = (512, 512),
        head_fuse_dropout: float = 0.1,
    ):
        super().__init__()

        self.size_output = size_output

        # Convert each feature map to decoder_dim channels
        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, decoder_dim, kernel_size=1),
                    nn.BatchNorm2d(decoder_dim),
                    nn.ReLU(inplace=True),
                )
                for ch in featuremap_channels_dims
            ]
        )

        # Fuse them all together
        self.fuse = nn.Sequential(
            nn.Conv2d(
                decoder_dim * len(featuremap_channels_dims), decoder_dim, kernel_size=1
            ),
            nn.ReLU(inplace=True),
            nn.Dropout2d(head_fuse_dropout),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1),
        )

        # Final resize to desired output size
        self.last_upsampling = nn.Upsample(
            size=size_output, mode="bilinear", align_corners=False
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of tensors with each shape [B, C, 32, 32]
        """
        assert len(features) == len(self.linear_layers)
        assert all(f.shape[2:] == features[0].shape[2:] for f in features), (
            "Spatial sizes must match"
        )

        # Apply 1x1 convs
        projected = [layer(f) for layer, f in zip(self.linear_layers, features)]

        # Concatenate and fuse
        fused = torch.cat(projected, dim=1)  # shape: [B, decoder_dim * N, 32, 32]
        out = self.fuse(fused)

        # Optional: upsample to final size
        out = self.last_upsampling(out)

        return out
    

    #  torch.Size([1, 192, 96, 96])
    #  torch.Size([1, 384, 48, 48])
    #  torch.Size([1, 768, 24, 24])
    #  torch.Size([1, 1536, 12, 12])
