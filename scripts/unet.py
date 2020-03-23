"""Define u-net."""

import numpy as np
import tensorflow as tf


def axslice(axis, start, stop, step=None):
    """Slice array along a specified axis."""
    if axis < 0:
        raise ValueError("`axis` must be positive")
    slicelist = [slice(None)] * axis
    slicelist.append(slice(start, stop, step))
    return tuple(slicelist)


def pad_axis_periodic(tensor, axis, pad_left, pad_right):
    """Periodically pad tensor along a single axis."""
    N = tensor.shape[axis]
    left = tensor[axslice(axis, N-pad_left, N)]
    right = tensor[axslice(axis, 0, pad_right)]
    return tf.concat([left, tensor, right], axis)


def compress_axis_periodic(tensor, axis, pad_left, pad_right):
    """Additively compress periodic padding along a single axis."""
    N_ext = tensor.shape[axis]
    # Take center of tensor
    center = tensor[axslice(axis, pad_left, N_ext-pad_right)]
    # Build tensor of the periodic image
    left = tensor[axslice(axis, 0, pad_left)]
    right = tensor[axslice(axis, N_ext-pad_right, N_ext)]
    void_shape = tensor.shape.as_list()
    void_shape[axis] = N_ext - 2*pad_left - 2*pad_right
    void = tf.zeros(void_shape, dtype=tensor.dtype)
    image = tf.concat([right, void, left], axis)
    return center + image
    # Can't add via slices because tensor objects don't support assigment in eager mode
    #center[axslice(axis, 0, pad_right)] += tensor[axslice(axis, N_ext-pad_right, N_ext)]
    #center[axslice(axis, N_ext-pad_left, N_ext)] += tensor[axslice(axis, 0, pad_left)]
    #return center


class PeriodicConv2D(tf.keras.Model):
    """3D convolution layer with periodic padding."""

    def __init__(self, filters, kernel_size, kernel_center, strides=(1,1), batch_norm=False, **kw):
        super().__init__()
        # Store inputs
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_center = kernel_center
        self.strides = strides
        self.batch_norm = batch_norm
        # Calculate pads
        self.pad_left = kernel_center
        self.pad_right = [ks - kc - 1 for ks, kc in zip(kernel_size, kernel_center)]
        # Build layers
        self.conv_valid = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='valid', **kw)
        if batch_norm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()

    def check_input_shape(self, input_shape):
        # Check strides evenly divide data shape
        batch, *data_shape, channels = input_shape
        for n, s in zip(data_shape, self.strides):
            if n%s != 0:
                raise ValueError("Strides must evenly divide data shape in periodic convolution.")

    def __call__(self, x):
        # Check shape
        self.check_input_shape(x.shape)
        # Iteratively apply periodic padding, skipping first axis (batch)
        for axis in range(2):
            x = pad_axis_periodic(x, axis+1, self.pad_left[axis], self.pad_right[axis])
        # Apply valid convolution
        x = self.conv_valid(x)
        # Batch normalization
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        return x


class PeriodicConv2DTranspose(tf.keras.Model):
    """3D transposed convolution layer with periodic padding."""

    def __init__(self, filters, kernel_size, kernel_center, strides=(1, 1), batch_norm=False, **kw):
        super().__init__()
        # Store inputs
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_center = kernel_center
        self.strides = strides
        self.batch_norm = batch_norm
        # Calculate pads
        self.pad_left = kernel_center
        self.pad_right = [ks - kc - 1 for ks, kc in zip(kernel_size, kernel_center)]
        self.output_padding = [[0, 0]] + [[0, min(ks, s) - 1] for ks, s in zip(kernel_size, strides)] + [[0, 0]]
        # Build layers
        self.conv_valid = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='valid', **kw)
        if batch_norm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()

    def __call__(self, x):
        # Apply valid convolution
        x = self.conv_valid(x)
        # Pad with zeros to original expanded size
        x = tf.pad(x, self.output_padding)
        # Additively compress periodic padding, skipping first axis (batch)
        for axis in range(3):
            x = compress_axis_periodic(x, axis+1, self.pad_left[axis], self.pad_right[axis])
        # Batch normalization
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        return x


class Unet(tf.keras.Model):

    def __init__(self, stacks, stackwidth, filters_base, output_channels, kernel_size=(3,3), kernel_center=None, strides=(2,2), filters_mult=2, **kw):
        """
        Parameters
        ----------
        stacks : int
            Number of unet stacks.
        stackwidth : int
            Number of convolutions per stack.
        filters_base : int
            Base number of intermediate convolution filters.
        output_channels : int
            Number of output channels
        kernel_size : int or tuple of ints
            Kernel size for each dimension, default: (3,3)
        kernel_center : int or tuple of ints
            Kernel center for each dimension, default: kernel_size//2
        strides : tuple of ints
            Strides for each dimension, default: (2,2)
        filters_mult: int
            Multiplicative increase to filter size per stack depth

        Other keyword arguments are passed to the convolution layers.

        """
        super().__init__()

        # Handle integer kernel specifications
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(kernel_center, int):
            kernel_center = (kernel_center,) * 2
        if kernel_center is None:
            kernel_center = tuple(ks//2 for ks in kernel_size)

        # Save attributes
        self.stacks = stacks
        self.stackwidth = stackwidth
        self.filters_base = filters_base
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.kernel_center = kernel_center
        self.strides = strides
        self.filters_mult = filters_mult
        self.conv_kw = kw

        # Downward stacks, downsampling at the beginning of each except the first
        self.down_stacks = []
        for i in range(stacks):
            stack = []
            filters = filters_base * filters_mult**i
            # Downsampling
            if i != 0:
                stack.append(PeriodicConv2D(filters, kernel_size, kernel_center, strides=strides, **kw))
            # Convolutions
            for j in range(stackwidth):
                stack.append(PeriodicConv2D(filters, kernel_size, kernel_center, **kw))
            self.down_stacks.append(stack)

        # Upward stacks, upsampling at the end of each except the last
        self.up_stacks = []
        for i in reversed(range(stacks)):
            stack = []
            filters = filters_base * filters_mult**i
            # Convolutions
            for j in range(stackwidth):
                stack.append(PeriodicConv2D(filters, kernel_size, kernel_center, **kw))
            # Upsampling
            if i != 0:
                stack.append(PeriodicConv2DTranspose(filters//filters_mult, kernel_size, kernel_center, strides=strides, **kw))
            self.up_stacks.append(stack)

        # Output layer: linear recombination with no rectification
        self.outlayer = PeriodicConv2D(output_channels, (1, 1), (0, 0), activation=None)

    def call(self, x_list):
        """Call unet on multiple inputs, combining at bottom."""

        def eval_down(x):
            """Evaluate down unet stacks, saving partials."""
            partials = []
            for stack in self.down_stacks:
                # Apply stack
                for layer in stack:
                    x = layer(x)
                    
                # Save partials
                partials.append(x)
            return partials

        def eval_up(partials):
            """Evaluate up unet stacks, concatenating partials."""
            x = None
            for stack in self.up_stacks:
                if x is None:
                    # Start with last partial
                    x = partials.pop()
                else:
                    # Concatenate partial
                    x = tf.concat([x, partials.pop()], axis=3)
                # Apply stack
                for layer in stack:
                    x = layer(x)
            return x

        # Evaluate down for each input
        partials = [eval_down(x) for x in x_list]

        # Concatenate partials for each input
        partials = list(zip(*partials))

        partials = [tf.concat(p, axis=3) for p in partials]
        # Evaluate up
        x = eval_up(partials)
        # Apply output layer
        return self.outlayer(x)

