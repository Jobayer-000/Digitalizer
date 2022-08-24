import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import torch 
from torch import nn
import gradient_checkpointing
    
    
def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return models.Sequential(layers)

class ResidualBlockNoBN(layers.Layer):
    """Residual block without BN.
    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
    """

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = layers.Conv2D(num_feat, 3, 1, padding='same', use_bias=True)
        self.conv2 = layers.Conv2D(num_feat, 3, 1, padding='same', use_bias=True)
        self.relu = layers.ReLU()

    def call(self, inputs):
       
        def inner(inputs):
            out = self.conv2(self.relu(self.conv1(inputs)))
            return inputs + out * self.res_scale
        return inner(inputs)
        
        
        




def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, img.dtype)
    y = tf.cast(y, img.dtype)
    x = tf.constant(0.5, dtype=img.dtype) * ((x + tf.constant(1.0, dtype=img.dtype)) * tf.cast(max_x-1,  img.dtype))
    y = tf.constant(0.5, dtype=img.dtype) * ((y +tf.constant(1.0, dtype=img.dtype)) * tf.cast(max_y-1,  img.dtype))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0,  img.dtype)
    x1 = tf.cast(x1,  img.dtype)
    y0 = tf.cast(y0,  img.dtype)
    y1 = tf.cast(y1,  img.dtype)

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.cast(tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id]), img.dtype)

    return out

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
    Returns:
        Tensor: Warped image or feature map.
    """
    shape = tf.shape(x)
    h, w = shape[1], shape[2]
    # create mesh grid
    grid_x, grid_y = tf.meshgrid(tf.range(w), tf.range(h))
    grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), x.dtype)
    vgrid = tf.add(grid, flow)
                            
    # scale grid to [-1,1]
    vgrid_x = tf.divide(tf.multiply(tf.constant(2.0, dtype=x.dtype), vgrid[:, :, :, 0]), tf.cast(tf.subtract(tf.reduce_max([tf.subtract(w, 1), 1]), 1), vgrid.dtype))
    vgrid_y = tf.divide(tf.multiply(tf.constant(2.0, dtype=x.dtype), vgrid[:, :, :, 1]), tf.cast(tf.subtract(tf.reduce_max([tf.subtract(h, 1), 1]), 1), vgrid.dtype))
    #vgrid_scaled = tf.stack((vgrid_x, vgrid_y), 3)
    output = bilinear_sampler(x, vgrid_x, vgrid_y)

    # TODO, what if align_corners=False
    return output

class BasicModule(layers.Layer):
    def __init__(self):
        super(BasicModule, self).__init__() 
        self.x1 = layers.Conv2D(32, 7, 1, padding='same', activation='relu')
        self.x2 = layers.Conv2D(64, 7, 1, padding='same', activation='relu')
        self.x3 = layers.Conv2D(32, 7, 1, padding='same', activation='relu')
        self.x4 = layers.Conv2D(16, 7, 1, padding='same', activation='relu')
        self.x5 = layers.Conv2D(2, 7, 1, padding='same')
    def call(self,x):
      
      def inner(x):
        x = self.x1(x)
        x = self.x2(x)
        x = self.x3(x)
        x = self.x4(x)
        x = self.x5(x)
        return x
      return inner(x)
class SpyNet(layers.Layer):
    """SpyNet architecture.
    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
    """

    def __init__(self, num_bblock, load_path=None):
        super(SpyNet, self).__init__()
        self.basic_module = [BasicModule() for _ in range(num_bblock)]
    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []

        ref = [ref]
        supp = [supp]

        for level in range(4):
            ref.insert(0, tf.nn.avg_pool2d(ref[0], ksize=2, strides=2, padding='VALID'))
            supp.insert(0, tf.nn.avg_pool2d(supp[0], ksize=2, strides=2, padding='VALID'))

        flow = tf.zeros(
            [tf.shape(ref[0])[0],
             tf.cast(tf.math.floor(tf.cast(tf.shape(ref[0])[1], ref[0].dtype) / 2.0), tf.int32),
             tf.cast(tf.math.floor(tf.cast(tf.shape(ref[0])[2], ref[0].dtype) / 2.0), tf.int32), 2], dtype=ref[0].dtype)
       
        for i in range(len(ref)):
            upsampled_flow = tf.cast(tf.compat.v1.image.resize(flow, tf.shape(flow)[1:3]*tf.constant(2), align_corners=True) * 2.0, flow.dtype)
            if upsampled_flow.shape[1] != ref[i].shape[1] or upsampled_flow.shape[2] != ref[i].shape[2]:
                upsampled_flow = tf.pad(upsampled_flow, [[0, 0], [tf.shape(ref[i])[1] - tf.shape(upsampled_flow)[1], 0],
                                                     [tf.shape(ref[i])[2] - tf.shape(upsampled_flow)[2], 0],[0, 0]],
                                      "SYMMETRIC")
            flow = self.basic_module[i](tf.concat([
                ref[i],
                tf.cast(flow_warp(
                    supp[i], upsampled_flow), upsampled_flow.dtype),
               upsampled_flow
            ], -1)) + upsampled_flow
           
        return flow

    def call(self, ref, supp):
        # h, w must be multiple of 32
        
        def inner(ref, supp):
            flow = self.process(ref, supp)
            return flow                    
        return inner(ref, supp)            
                    
                    
               
class BasicVSR(models.Model):
    """A recurrent network for video SR. Now only x4 is supported.
    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15, num_bblock = 6, output_ch=3, spynet_path=None, scale=False, shapes=(25,224,224,3)):
        super().__init__()
        self.shapes = shapes
        self.num_feat = num_feat
        self.scale = scale
        # alignment
        self.spynet = SpyNet(num_bblock, spynet_path)

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.fusion = layers.Conv2D(num_feat, 1, 1, use_bias=True)
        self.upconv1 = layers.Conv2D(num_feat * 4, 3, 1, padding='same', use_bias=True)
        self.upconv2 = layers.Conv2D(64 * 4, 3, 1, padding='same', use_bias=True)
        self.conv_hr = layers.Conv2D(64, 3, 1, padding='same')
        self.conv_hr2 = layers.Conv2D(64, 3, 1, padding='same')
        self.conv_hr1 = layers.Conv2D(64, 3, 1, padding='same')
        self.conv_last = layers.Conv2D(output_ch, 3, 1, padding='same', dtype=tf.float32)
        self.conv_last2 = layers.Conv2D(output_ch, 3, 1, padding='same', dtype=tf.float32)
        self.conv_last1 = layers.Conv2D(output_ch, 3, 1, padding='same', dtype=tf.float32)

        self.pixel_shuffle = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))
        # activation functions
        self.lrelu = layers.LeakyReLU(.1)

    def get_flow(self, x):
       
        shape = tf.shape(x)
        b, n, h, w, c = shape[0], shape[1], shape[2], shape[3], shape[4]
        
        x_1 = tf.reshape(x[:, :-1, :, :, :], (-1, h, w, c))
        x_2 = tf.reshape(x[:, 1:, :, :, :], (-1, h, w, c))
        flows_backward = tf.reshape(self.spynet(x_1, x_2), (b, n - 1, h, w, 2))
        flows_forward = tf.reshape(self.spynet(x_2, x_1), (b, n - 1, h, w, 2))
        return flows_forward, flows_backward

    def call(self, x):
        """Forward function of BasicVSR.
        Args:
            x: Input frames with shape (b, n, h, w, c). n is the temporal dimension / number of frames.
        """
        x = tf.cast(x,tf.keras.mixed_precision.global_policy().compute_dtype)
        flows_forward, flows_backward = self.get_flow(x)
        shape = tf.shape(x)
        b, n, h, w, c = shape[0], shape[1], shape[2], shape[3], shape[4]
        # backward branch
        out_l = []
        feat_prop = tf.zeros((b, h, w, self.num_feat), flows_forward.dtype)
        for i in range(x.shape[1] - 1, -1, -1):
            x_i = tf.cast(x[:, i, :, :, :], flows_forward.dtype)
            if i < self.shapes[0] - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = tf.cast(flow_warp(feat_prop, flow), flow.dtype)
            feat_prop = tf.concat([x_i, feat_prop], -1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        out_j, out_j1, out_j2 = [],[],[]
        feat_prop = tf.zeros_like((feat_prop), flows_forward.dtype)
        for i in range(0, x.shape[1]):
            x_i = tf.cast(x[:, i, :, :, :], flows_forward.dtype)
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = tf.cast(flow_warp(feat_prop, flow), flow.dtype)

            feat_prop = tf.concat([x_i, feat_prop], -1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = tf.concat([out_l[i], feat_prop], -1)
            
            def inner(x):
                x1 = self.lrelu(self.fusion(x))
                x2 = self.lrelu(self.pixel_shuffle(self.upconv1(x1)))
                x3 = self.lrelu(self.pixel_shuffle(self.upconv2(x2)))
                x4 = self.lrelu(self.conv_hr(x3))
                x5 = self.conv_last(x4)
                return x5, x2, x1
            
            if self.scale:
              out,out1,out2 = inner(out)
              out1 = self.conv_last1(self.lrelu(self.conv_hr1(out1)))
              out2 = self.conv_last2(self.lrelu(self.conv_hr2(out2)))
              base1 = tf.cast(tf.image.resize(x_i, x_i.shape[1:3]*tf.constant(2)), tf.float32)
              base2 = tf.cast(tf.image.resize(x_i, x_i.shape[1:3]), tf.float32)
            
              out_j1.append(tf.add(out1,base1))
              out_j2.append(tf.add(out2,base2))
            else:
                out = self.lrelu(self.fusion(out))
                out = self.conv_last(out)
                
            out_j.append(tf.add(out, x_i[...,:3]))
        
        return tf.stack(out_j, 1) if not self.scale else (tf.stack(out_j, 1), tf.stack(out_j1,1), tf.stack(out_j2,1))
   
    def build_model_with_grap(self, input_shape):
        inputs = keras.Input(shape=input_shape)
        models = keras.Model(inputs=inputs, outputs=self.call(inputs))
        return models

class ConvResidualBlocks(layers.Layer):
    """Conv and residual block used in BasicVSR.
    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = models.Sequential([
            layers.Conv2D(num_out_ch, 3, 1, padding='same', use_bias=True),
            layers.LeakyReLU(.1),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch)])
        self.cv = layers.Conv2D(num_out_ch, 3, 1, padding='same', use_bias=True, activation=layers.LeakyReLU(.1))
        self.rblock = [ResidualBlockNoBN(num_feat=num_out_ch) for _ in range(num_block)]
    def call(self, fea):
        
        def inner(fea):
            x = self.cv(fea)
            for block in self.rblock:
                x = block(x)
            return self.main(fea)
        return inner(fea)
