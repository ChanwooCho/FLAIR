# -*- coding: utf-8 -*-
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.

import tensorflow as tf
from typing import Optional, Callable, List, Tuple

from keras.layers import (
    RandomCrop, RandomFlip, Normalization, Rescaling,
    Conv2D, ZeroPadding2D, ReLU, MaxPooling2D, BatchNormalization)
import tensorflow_addons.layers.normalizations as tfa_norms
from keras.applications import imagenet_utils

import models.mobilenet_v1
import models.mobilenet_v2
import models.mobilenet_v3

import models.inception_v3
import models.squeezenet
import models.regnet
import models.efficientnet
import models.resnet
import models.nasnet
import models.mobilevit
import models.squeezenet

# ImageNet statistics from https://pytorch.org/vision/stable/models.html
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_VARIANCE = [0.229 ** 2, 0.224 ** 2, 0.225 ** 2]


class FrozenBatchNormalization(BatchNormalization):
    """
    BatchNormalization layer that freezes the moving mean and average updates.
    It is intended to be used in fine-tuning a pretrained model in federated
    learning setting, where the moving mean and average will be assigned to
    the ones in the pretrained model. Only beta and gamma are updated.
    """
    def call(self, inputs, training=None):
        inputs_dtype = inputs.dtype.base_dtype
        if inputs_dtype in (tf.float16, tf.bfloat16):
            inputs = tf.cast(inputs, tf.float32)

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]

        # Broadcasting only necessary for single-axis batch norm where the axis
        # is not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if (v is not None and len(v.shape) != ndims and
                    reduction_axes != list(range(ndims - 1))):
                return tf.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)
        # use pretrained moving_mean and moving_variance for normalization
        mean, variance = self.moving_mean, self.moving_variance
        mean = tf.cast(mean, inputs.dtype)
        variance = tf.cast(variance, inputs.dtype)
        if offset is not None:
            offset = tf.cast(offset, inputs.dtype)
        if scale is not None:
            scale = tf.cast(scale, inputs.dtype)
        outputs = tf.nn.batch_normalization(inputs, _broadcast(mean),
                                            _broadcast(variance), offset, scale,
                                            self.epsilon)
        if inputs_dtype in (tf.float16, tf.bfloat16):
            outputs = tf.cast(outputs, inputs_dtype)

        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)
        return outputs


def conv3x3(x: tf.Tensor, scope: str, out_planes: int, stride: int = 1,
            groups: int = 1, dilation: int = 1, seed: int = 0):
    """3x3 convolution with padding"""
    x = ZeroPadding2D(padding=(dilation, dilation), name=f"{scope}_padding")(x)
    return Conv2D(
        out_planes,
        kernel_size=3,
        strides=stride,
        groups=groups,
        use_bias=False,
        dilation_rate=dilation,
        name=f"{scope}_3x3",
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
    )(x)


def conv1x1(x: tf.Tensor, scope: str, out_planes: int, stride: int = 1,
            seed: int = 0):
    """1x1 convolution"""
    return Conv2D(
        out_planes,
        kernel_size=1,
        strides=stride,
        use_bias=False,
        name=f"{scope}_1x1",
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
    )(x)


def norm(x: tf.Tensor, scope: str, use_batch_norm: bool):
    """Normalization layer"""
    if use_batch_norm:
        return FrozenBatchNormalization(axis=3, epsilon=1e-5, name=scope)(x)
    else:
        return tfa_norms.GroupNormalization(epsilon=1e-5, name=scope)(x)


def relu(x: tf.Tensor, scope: str):
    """ReLU activation layer"""
    return ReLU(name=scope)(x)


def basic_block(x: tf.Tensor, scope: str, out_planes: int, use_batch_norm: bool,
                stride: int = 1, downsample: Optional[Callable] = None,
                seed: int = 0):
    """Basic ResNet block"""
    out = conv3x3(x, f"{scope}_conv1", out_planes, stride, seed=seed)
    out = norm(out, scope=f"{scope}_norm1", use_batch_norm=use_batch_norm)
    out = relu(out, f"{scope}_relu1")
    out = conv3x3(out, f"{scope}_conv2", out_planes, seed=seed)
    out = norm(out, scope=f"{scope}_norm2", use_batch_norm=use_batch_norm)
    if downsample is not None:
        x = downsample(x)
    out += x
    out = relu(out, f"{scope}_relu2")
    return out


def block_layers(
    x: tf.Tensor,
    scope: str,
    in_planes: int,
    out_planes: int,
    blocks: int,
    use_batch_norm: bool,
    stride: int = 1,
    seed: int = 0,
):
    """Layers of ResNet block"""
    downsample = None
    if stride != 1 or in_planes != out_planes:
        # Downsample is performed when stride > 1 according to Section 3.3 in
        # https://arxiv.org/pdf/1512.03385.pdf
        def downsample(h: tf.Tensor):
            h = conv1x1(h, f"{scope}_downsample_conv", out_planes, stride)
            return norm(h, f"{scope}_downsample_norm", use_batch_norm)

    x = basic_block(x, f"{scope}_block1", out_planes, use_batch_norm, stride,
                    downsample, seed=seed)
    for i in range(1, blocks):
        x = basic_block(x, f"{scope}_block{i + 1}", out_planes, use_batch_norm,
                        seed=seed)
    return x


def create_resnet(input_shape: Tuple[int, int, int],
                  num_classes: int,
                  use_batch_norm: bool,
                  repetitions: List[int] = None,
                  initial_filters: int = 64,
                  seed: int = 0):
    """
    Creates a ResNet Keras model. Implementation follows torchvision in
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """
    img_input = tf.keras.layers.Input(shape=input_shape)

    # preprocessing layer
    x = RandomCrop(height=224, width=224)(img_input)
    x = RandomFlip()(x)
    x = Rescaling(scale=1 / 255.0)(x)
    x = Normalization(
        axis=-1, mean=IMAGENET_MEAN, variance=IMAGENET_VARIANCE)(x)

    # initial conv layer
    x = ZeroPadding2D((3, 3), name="initial_padding")(x)
    x = Conv2D(
        initial_filters,
        kernel_size=7, strides=2, use_bias=False, name="initial_conv",
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed))(x)
    x = norm(x, scope="initial_norm", use_batch_norm=use_batch_norm)
    x = relu(x, scope="initial_relu")
    x = ZeroPadding2D((1, 1), name="pooling_padding")(x)
    x = MaxPooling2D(pool_size=3, strides=2, name="initial_pooling")(x)

    # residual blocks
    x = block_layers(x, "layer1", initial_filters, 64, repetitions[0],
                     use_batch_norm, seed=seed)
    x = block_layers(x, "layer2", initial_filters, 128, repetitions[1],
                     use_batch_norm, 2, seed=seed)
    x = block_layers(x, "layer3", initial_filters, 256, repetitions[2],
                     use_batch_norm, 2, seed=seed)
    x = block_layers(x, "layer4", initial_filters, 512, repetitions[3],
                     use_batch_norm, 2, seed=seed)

    # classification layers
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pooling")(x)
    x = tf.keras.layers.Dense(
        num_classes, name="classifier",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
    model = tf.keras.models.Model(img_input, x)
    return model


def resnet18(input_shape: Tuple[int, int, int],
             num_classes: int,
             pretrained: bool,
             seed: int = 0):
    """
    Creates a ResNet18 keras model.

    :param input_shape:
        Input image shape in [height, weight, channels.]
    :param num_classes:
        Number of output classes.
    :param pretrained:
        Whether the model is pretrained on ImageNet. If true, model will use
        BatchNormalization. If false, model will use GroupNormalization in order
        to train with differential privacy.
    :param seed:
        Random seed for initialize the weights.

    :return:
        A ResNet18 keras model
    """
    model = create_resnet(
        input_shape,
        num_classes,
        use_batch_norm=pretrained,
        repetitions=[2, 2, 2, 2],
        seed=seed)
    return model

resnet18([224, 224, 3], 17, False)

#############################################################################################################

def hard_sigmoid(x):
    return tf.keras.layers.ReLU(6.)(x + 3.) * (1. / 6.)

def hard_swish(x):
    return tf.keras.layers.Multiply()([x, hard_sigmoid(x)])

def create_mobilenet(input_shape: Tuple[int, int, int],
                     num_classes: int,
                     seed: int = 0):

    img_input = tf.keras.layers.Input(shape=input_shape)

    # preprocessing layer
    x = RandomCrop(height=224, width=224)(img_input)
    x = RandomFlip()(x)
    x = Rescaling(scale=1 / 255.0)(x)
    x = Normalization(
        axis=-1, mean=IMAGENET_MEAN, variance=IMAGENET_VARIANCE)(x)
    
    # bottleneck
    x = models.mobilenet_v3.MobileNetV3Small(include_top=False, weights=None)(x)

    # Mopbilenetv3large classification layers
    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
    x = Conv2D(
        1280,
        kernel_size=1,
        padding="same",
        use_bias=True,
        name="Conv_2",
    )(x)
    x = hard_swish(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # multilabel classification
    x = tf.keras.layers.Dense(
        num_classes, name="classifier",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
    model = tf.keras.models.Model(img_input, x)

    return model

#############################################################################################################
def create_inception(input_shape: Tuple[int, int, int],
                     num_classes: int,
                     seed: int = 0):

    img_input = tf.keras.layers.Input(shape=input_shape)

    # preprocessing layer
    x = RandomCrop(height=224, width=224)(img_input)
    x = RandomFlip()(x)
    x = Rescaling(scale=1 / 255.0)(x)
    x = Normalization(
        axis=-1, mean=IMAGENET_MEAN, variance=IMAGENET_VARIANCE)(x)
    
    # bottleneck
    x = inception_v3.InceptionV3(include_top=False, weights=None, classes=17)(x)

    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(x)
    x = Conv2D(
        1024,
        kernel_size=1,
        padding="same",
        use_bias=True,
        name="Conv_2",
    )(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(1e-3, name="dropout")(x)

    
    # multilabel classification
    x = tf.keras.layers.Dense(
        num_classes, name="classifier",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
    model = tf.keras.models.Model(img_input, x)

    return model

#############################################################################################################
def create_squeezenet(input_shape: Tuple[int, int, int],
                     num_classes: int,
                     seed: int = 0):

    img_input = tf.keras.layers.Input(shape=input_shape)

    # preprocessing layer
    # x = RandomCrop(height=224, width=224)(img_input)
    x = img_input
    x = RandomFlip()(x)
    x = Rescaling(scale=1 / 255.0)(x)
    x = Normalization(
        axis=-1, mean=IMAGENET_MEAN, variance=IMAGENET_VARIANCE)(x)
    
    # bottleneck
    x = models.squeezenet.SqueezeNet(input_shape=input_shape, nb_classes=num_classes)(x)

    x = tf.keras.layers.Flatten()(x)
     # multilabel classification
    x = tf.keras.layers.Dense(
        num_classes, name="classifier",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        
    model = tf.keras.models.Model(img_input, x)

    return model

#############################################################################################################
def create_regnet(input_shape: Tuple[int, int, int],
                     num_classes: int,
                     seed: int = 0):

    img_input = tf.keras.layers.Input(shape=input_shape)

    # preprocessing layer
    # x = RandomCrop(height=224, width=224)(img_input)
    x = img_input
    x = RandomFlip()(x)
    x = Rescaling(scale=1 / 255.0)(x)
    x = Normalization(
        axis=-1, mean=IMAGENET_MEAN, variance=IMAGENET_VARIANCE)(x)
    
    # bottleneck
    x = models.regnet.RegNetY016(include_top=False, weights=None, include_preprocessing=False)(x)

    x = tf.keras.layers.Flatten()(x)
     # multilabel classification
    x = tf.keras.layers.Dense(
        num_classes, name="classifier",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        
    model = tf.keras.models.Model(img_input, x)

    return model

#############################################################################################################
def create_efficientnet(input_shape: Tuple[int, int, int],
                     num_classes: int,
                     seed: int = 0):

    img_input = tf.keras.layers.Input(shape=input_shape)

    # preprocessing layer
    x = RandomCrop(height=224, width=224)(img_input)
    x = RandomFlip()(x)
    x = Rescaling(scale=1 / 255.0)(x)
    x = Normalization(
        axis=-1, mean=IMAGENET_MEAN, variance=IMAGENET_VARIANCE)(x)
    
    # bottleneck
    x = models.efficientnet.EfficientNetB0(include_top=False, weights=None)(x)

    x = tf.keras.layers.Flatten()(x)
     # multilabel classification
    x = tf.keras.layers.Dense(
        num_classes, name="classifier",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        
    model = tf.keras.models.Model(img_input, x)

    return model

    #############################################################################################################
def create_resnet50(input_shape: Tuple[int, int, int],
                     num_classes: int,
                     seed: int = 0):

    img_input = tf.keras.layers.Input(shape=input_shape)

    # preprocessing layer
    x = RandomCrop(height=224, width=224)(img_input)
    x = RandomFlip()(x)
    x = Rescaling(scale=1 / 255.0)(x)
    x = Normalization(
        axis=-1, mean=IMAGENET_MEAN, variance=IMAGENET_VARIANCE)(x)
    
    # bottleneck
    x = models.resnet.ResNet50(include_top=False, weights=None)(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pooling")(x)
     # multilabel classification
    x = tf.keras.layers.Dense(
        num_classes, name="classifier",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        
    model = tf.keras.models.Model(img_input, x)

    return model

    #############################################################################################################
def create_resnet101(input_shape: Tuple[int, int, int],
                     num_classes: int,
                     seed: int = 0):

    img_input = tf.keras.layers.Input(shape=input_shape)

    # preprocessing layer
    x = RandomCrop(height=224, width=224)(img_input)
    x = RandomFlip()(x)
    x = Rescaling(scale=1 / 255.0)(x)
    x = Normalization(
        axis=-1, mean=IMAGENET_MEAN, variance=IMAGENET_VARIANCE)(x)
    
    # bottleneck
    x = models.resnet.ResNet101(include_top=False, weights=None)(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pooling")(x)
     # multilabel classification
    x = tf.keras.layers.Dense(
        num_classes, name="classifier",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        
    model = tf.keras.models.Model(img_input, x)

    return model

#############################################################################################################
def create_nasnet(input_shape: Tuple[int, int, int],
                     num_classes: int,
                     seed: int = 0):

    img_input = tf.keras.layers.Input(shape=input_shape)

    # preprocessing layer
    x = RandomCrop(height=224, width=224)(img_input)
    x = RandomFlip()(x)
    x = Rescaling(scale=1 / 255.0)(x)
    x = Normalization(
        axis=-1, mean=IMAGENET_MEAN, variance=IMAGENET_VARIANCE)(x)
    
    # bottleneck
    x = models.nasnet.NASNetMobile(include_top=False, weights=None)(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pooling")(x)
     # multilabel classification
    x = tf.keras.layers.Dense(
        num_classes, name="classifier",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
        
    model = tf.keras.models.Model(img_input, x)

    return model

#############################################################################################################
##for mobilevit
patch_size = 4
expansion_factor = 2


def conv_block(x, filters=16, kernel_size=3, strides=2):
    conv_layer = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=strides, activation=tf.nn.swish, padding="same"
    )
    return conv_layer(x)


# Reference: https://git.io/JKgtC


def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    m = tf.keras.layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)

    m = tfa_norms.GroupNormalization(
        axis=-1, groups=24 if m.shape[-1]==48 else 32, epsilon=1e-3
    )(m)
    m = tf.nn.swish(m)

    if strides == 2:
        m = tf.keras.layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(m, 3))(m)
    m = tf.keras.layers.DepthwiseConv2D(
        3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False
    )(m)

    m = tfa_norms.GroupNormalization(
        axis=-1, groups=24 if m.shape[-1]==48 else 32, epsilon=1e-3
    )(m)
    m = tf.nn.swish(m)

    m = tf.keras.layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
    
    groups = -1

    if m.shape[-1] == 64:
        groups = 32
    elif m.shape[-1] == 80:
        groups = 20
    else:
        gorups = 24

    m = tfa_norms.GroupNormalization(
        axis=-1, groups=groups, epsilon=1e-3
    )(m)

    if tf.math.equal(x.shape[-1], output_channels) == tf.constant(True) and strides == 1:
        return tf.keras.layers.Add()([m, x])
    return m


# Reference:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.swish)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = tf.keras.layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=[x.shape[-1] * 2, x.shape[-1]], dropout_rate=0.1,)
        # Skip connection 2.
        x = tf.keras.layers.Add()([x3, x2])

    return x


def mobilevit_block(x, num_blocks, projection_dim, strides=1):
    # Local projection with convolutions.
    interpolate=False
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = conv_block(
        local_features, filters=projection_dim, kernel_size=1, strides=strides
    )

    # Unfold into patches and then pass through Transformers.
    num_patches = int((local_features.shape[1] * local_features.shape[2]) / patch_size)
#     if num_patches * patch_size != local_features.shape[1] * local_features.shape[2]:
#         local_features = tf.image.resize(local_features, [local_features.shape[1]+1,local_features.shape[2]+1])
#         num_patches = 16
#         interpolate=True
        
        
    non_overlapping_patches = tf.keras.layers.Reshape((patch_size, num_patches, projection_dim))(
        local_features
    )
    global_features = transformer_block(
        non_overlapping_patches, num_blocks, projection_dim
    )

    # Fold into conv-like feature-maps.
    folded_feature_map = tf.keras.layers.Reshape((*local_features.shape[1:-1], projection_dim))(
        global_features
    )

    
    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(
        folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides
    )
    
#     if interpolate:
#         folded_feature_map = tf.image.resize(folded_feature_map, [folded_feature_map.shape[1]-1,folded_feature_map.shape[2]-1])
    
    local_global_features = tf.keras.layers.Concatenate(axis=-1)([x, folded_feature_map])

    # Fuse the local and global features using a convoluion layer.
    local_global_features = conv_block(
        local_global_features, filters=projection_dim, strides=strides
    )

    return local_global_features


def create_mobilevit(input_shape: Tuple[int, int, int],
                  num_classes: int,
                  seed: int = 0):
    img_input = tf.keras.layers.Input(shape=input_shape)

    
    # preprocessing layer
    x = RandomCrop(height=256, width=256)(img_input)
    x = RandomFlip()(x)
    x = Rescaling(scale=1 / 255.0)(x)
    x = Normalization(
        axis=-1, mean=IMAGENET_MEAN, variance=IMAGENET_VARIANCE)(x)
    
    #initial MV2 block.
    x = conv_block(x, filters=16)
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=16
    )

    # Downsampling with MV2 block.
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=24, strides=2
    )
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24
    )
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24
    )

    # First MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=48, strides=2
    )
    x = mobilevit_block(x, num_blocks=2, projection_dim=64)

    # Second MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=64 * expansion_factor, output_channels=64, strides=2
    )
    x = mobilevit_block(x, num_blocks=4, projection_dim=80)

#     Third MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=80 * expansion_factor, output_channels=80, strides=2
    )
    x = mobilevit_block(x, num_blocks=3, projection_dim=96)
    
    x = conv_block(x, filters=320, kernel_size=1, strides=1)

    #classification layers
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pooling")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(
        num_classes, name="classifier",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
    model = tf.keras.models.Model(img_input, x)
    return model
