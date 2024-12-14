import ml_collections

def get_b16_config():
    """
    Returns the ViT-B/16 configuration.

    This configuration defines a Vision Transformer with a base architecture
    and patch size of 16x16. It specifies the transformer parameters such as
    hidden size, number of layers, number of heads, and other relevant settings.

    Returns:
        ml_collections.ConfigDict: Configuration dictionary for ViT-B/16.
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'  # Type of classifier: segmentation.
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)  # Decoder channel configurations.
    config.n_classes = 2  # Number of segmentation classes.
    config.activation = 'softmax'  # Activation function for output.
    return config

def get_testing():
    """
    Returns a minimal configuration for testing purposes.

    This lightweight configuration is used for debugging or validating the
    implementation of the model with minimal resources.

    Returns:
        ml_collections.ConfigDict: Minimal testing configuration.
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'  # Classification type: token-level.
    config.representation_size = None
    return config

def get_r50_b16_config():
    """
    Returns the ResNet-50 + ViT-B/16 configuration.

    Combines a ResNet-50 backbone with a Vision Transformer (ViT-B/16) for
    better feature extraction and segmentation performance.

    Returns:
        ml_collections.ConfigDict: Configuration dictionary for ResNet-50 + ViT-B/16.
    """
    config = get_b16_config()
    config.patches.grid = (16, 16)  # Define the patch grid.
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)  # ResNet-50 layer configuration.
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]  # Skip connection configurations.
    config.n_classes = 2
    config.n_skip = 3  # Number of skip connections.
    config.activation = 'softmax'

    return config

def get_b32_config():
    """
    Returns the ViT-B/32 configuration.

    This configuration specifies a Vision Transformer with a base architecture
    and patch size of 32x32.

    Returns:
        ml_collections.ConfigDict: Configuration dictionary for ViT-B/32.
    """
    config = get_b16_config()
    config.patches.size = (32, 32)  # Update patch size to 32x32.
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config

def get_l16_config():
    """
    Returns the ViT-L/16 configuration.

    Defines a Vision Transformer with a larger architecture (L/16) and
    patch size of 16x16, suitable for tasks requiring higher capacity.

    Returns:
        ml_collections.ConfigDict: Configuration dictionary for ViT-L/16.
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    config.classifier = 'seg'
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config

def get_r50_l16_config():
    """
    Returns the ResNet-50 + ViT-L/16 configuration.

    Combines a ResNet-50 backbone with a Vision Transformer (ViT-L/16)
    for enhanced segmentation capabilities.

    Returns:
        ml_collections.ConfigDict: Configuration dictionary for ResNet-50 + ViT-L/16.
    """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.resnet_pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'
    return config

def get_l32_config():
    """
    Returns the ViT-L/32 configuration.

    Defines a larger Vision Transformer architecture (L/32) with
    patch size 32x32 for specific use cases.

    Returns:
        ml_collections.ConfigDict: Configuration dictionary for ViT-L/32.
    """
    config = get_l16_config()
    config.patches.size = (32, 32)  # Update patch size to 32x32.
    return config

def get_h14_config():
    """
    Returns the ViT-H/14 configuration.

    Specifies a Vision Transformer with an even larger architecture (H/14)
    and patch size of 14x14 for high-capacity tasks.

    Returns:
        ml_collections.ConfigDict: Configuration dictionary for ViT-H/14.
    """
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'  # Token-level classifier.
    config.representation_size = None

    return config
