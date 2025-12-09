---
library_name: segmentation-models-pytorch
license: mit
pipeline_tag: image-segmentation
tags:
- model_hub_mixin
- pytorch_model_hub_mixin
- segmentation-models-pytorch
- semantic-segmentation
- pytorch
languages:
- python
---
# DeepLabV3Plus Model Card

Table of Contents:
- [Load trained model](#load-trained-model)
- [Model init parameters](#model-init-parameters)
- [Model metrics](#model-metrics)
- [Dataset](#dataset)

## Load trained model
```python
import segmentation_models_pytorch as smp

model = smp.from_pretrained("<save-directory-or-this-repo>")
```

## Model init parameters
```python
model_init_params = {
    "encoder_name": "mit_b1",
    "encoder_depth": 5,
    "encoder_weights": "imagenet",
    "encoder_output_stride": 16,
    "decoder_channels": 256,
    "decoder_atrous_rates": (12, 24, 36),
    "decoder_aspp_separable": True,
    "decoder_aspp_dropout": 0.5,
    "in_channels": 1,
    "classes": 2,
    "activation": None,
    "upsampling": 4,
    "aux_params": None
}
```

## Model metrics
```json
{
    "epoch": 97,
    "iou": 0.9233444929122925,
    "dice": 0.9585462808609009
}
```

## Dataset
Dataset name: 512_20_pruned3570

## More Information
- Library: https://github.com/qubvel/segmentation_models.pytorch
- Docs: https://smp.readthedocs.io/en/latest/

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin)