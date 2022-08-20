import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


basic_module = models.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding='same'), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding='same'), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding='same'), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding='same'), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding='same'))


@ARCH_REGISTRY.register()
class SpyNet(layers.Layer):
    """SpyNet architecture.
    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
    """

    def __init__(self, load_path=None):
        super(SpyNet, self).__init__()
        self.basic_module = models.Sequential([basic_module for _ in range(6)])
    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []

        ref = [ref]
        supp = [supp]

        for level in range(5):
            ref.insert(0, tf.nn.avg_pool2d(input=ref[0], kernel_size=2, stride=2))
            supp.insert(0, tf.nn..avg_pool2d(input=supp[0], kernel_size=2, stride=2))

        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             tf.cast(tf.math.floor(ref[0].shape[1] / 2.0),tf.int32),
             tf.cast(tf.math.floor(ref[0].shape[2]) / 2.0),tf.int32)])

        for level in range(len(ref)):
            upsampled_flow = tf.compat.v1.image.resize(flow, scale_factor=flow.shape[1:3]*2, mode='bilinear', align_corners=True) * 2.0

            if upsampled_flow.shape(1) != ref[level].size(1):
                upsampled_flow = tf.pad(upsampled_flow, [[0,0],[0,1],[0,0],[0,0]], 'REFLECT')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = tf.pad(input=upsampled_flow, [[0,0],[0,0],[0,1],[0,0]], 'REFLECT')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level], upsampled_flow.permute(0, 2, 3, 1), interp_mode='bilinear', padding_mode='border'),
                upsampled_flow
            ], 1)) + upsampled_flow

        return flow

    def call(self, ref, supp):
        assert ref.shape == supp.shape

        h, w = ref.shape[1], ref.shape[2]
        w_floor = tf.math.floor(tf.math.ceil(w / 32.0) * 32.0)
        h_floor = tf.math.floor(tf.math.ceil(h / 32.0) * 32.0)

        ref = tf.image.resize(ref, (h_floor, w_floor))
        supp = tf.image.resize(supp, (h_floor, w_floor))

        flow = tf.image.resize(self.process(ref, supp), (h, w))

        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)

        return flow
