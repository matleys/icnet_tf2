import tensorflow as tf
from tf_layers import AvgSPP

class ICNetBlock(tf.keras.layers.Layer):
    def __init__(self, conv12_filters, conv3_filters, dilation_rate, name, do_bn=True):
        super().__init__(name=name)
        self.conv12_filters = conv12_filters
        self.conv3_filters = conv3_filters
        self.dilation_rate = dilation_rate
        self.do_bn = do_bn
    def get_config(self):
        return {
            "conv12_filters": self.conv12_filters,
            "conv3_filters": self.conv3_filters,
            "dilation_rate": self.dilation_rate
        }
    def build(self, input_shape):
        self.block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.conv12_filters, 1, use_bias=False, activation=None, name=self.name+'_1x1_reduce'),
            BatchNormalization(do=self.do_bn),
            tf.keras.layers.ReLU(name=self.name+'_1x1_reduce_bn'),
            tf.keras.layers.ZeroPadding2D(padding=self.dilation_rate, name=self.name+'padding1'),
            tf.keras.layers.Conv2D(self.conv12_filters, 3, dilation_rate=self.dilation_rate, use_bias=False, activation=None, name=self.name+'_3x3'),
            BatchNormalization(do=self.do_bn),
            tf.keras.layers.ReLU(name=self.name+'_3x3_bn'),
            tf.keras.layers.Conv2D(self.conv3_filters, 1, use_bias=False, activation=None, name=self.name+'_1x1_increase'),
            BatchNormalization(do=self.do_bn, name=self.name+'_1x1_increase_bn'),
        ])
    def call(self, inputs):
        return self.block(inputs)

class BatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, do=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.do = do
    def build(self, input_shape):
        if self.do:
            return super().build(input_shape)
    def call(self, inputs, training=None):
        if self.do:
            return super().call(inputs, training)
        return inputs

class ICNetModel(tf.keras.Model):
    def __init__(self, do_bn=True, skip_first_bilinear_resize=True, filter_number=32):
        super().__init__()
        self.skip_first_bilinear_resize = skip_first_bilinear_resize
        self.f1 = filter_number
        self.f2 = 2*self.f1
        self.f4 = 4*self.f1
        self.f8 = 8*self.f1
        self.f16 = 16*self.f1
        self.f32 = 32*self.f1
        self.do_bn = do_bn

    def get_config(self):
        return {"filter_number": self.f1, "do_bn": self.do_bn}

    def build(self, input_shape):
        interp = "bilinear"
        assert input_shape[1]%16 == 0, "Input tensor height must be dividable by 16. Received {}".format(input_shape[1])
        assert input_shape[2]%16 == 0, "Input tensor width must be dividable by 16. Received {}".format(input_shape[1])

        self.block1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.f1, 3, 2, use_bias=False, padding='SAME', name='conv1_1_3x3_s2'), # 1/2
            BatchNormalization(do=self.do_bn, name='conv1_1_3x3_s2_bn'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.f1, 3, 1, use_bias=False, padding='SAME', name='conv1_2_3x3'),
            BatchNormalization(do=self.do_bn, name='conv1_2_3x3_bn'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.f2, 3, 1, use_bias=False, padding='SAME', name='conv1_3_3x3'),
            BatchNormalization(do=self.do_bn, name='conv1_3_3x3_bn'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.ZeroPadding2D(),
            tf.keras.layers.MaxPool2D(3, 2, name='pool1_3x3_s2'),
        ])
        self.block2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.f4, 1, 1, use_bias=False, padding='VALID', name='conv2_1_1x1_proj'),
            BatchNormalization(do=self.do_bn, name='conv2_1_1x1_proj_bn')
        ])
        self.conv2_1 = ICNetBlock(self.f1, self.f4, dilation_rate=1, do_bn=self.do_bn, name='conv2_1')
        self.conv2_2 = ICNetBlock(self.f1, self.f4, dilation_rate=1, do_bn=self.do_bn, name='conv2_2')
        self.conv2_3 = ICNetBlock(self.f1, self.f4, dilation_rate=1, do_bn=self.do_bn, name='conv2_3')

        self.block3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.f8, 1, strides=2, use_bias=False, activation=None, name='conv3_1_1x1_proj'),
            BatchNormalization(do=self.do_bn, name='conv3_1_1x1_proj_bn')
        ])

        self.block4 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.f2, 1, strides=2, use_bias=False, activation=None, name="conv3_1"+'_1x1_reduce'),
            BatchNormalization(do=self.do_bn),
            tf.keras.layers.ReLU(name="conv3_1"+'_1x1_reduce_bn'),
            tf.keras.layers.ZeroPadding2D(padding=1, name="conv3_1"+'padding1'),
            tf.keras.layers.Conv2D(self.f2, 3, dilation_rate=1, use_bias=False, activation=None, name="conv3_1"+'_3x3'),
            BatchNormalization(do=self.do_bn),
            tf.keras.layers.ReLU(name="conv3_1"+'_3x3_bn'),
            tf.keras.layers.Conv2D(self.f8, 1, use_bias=False, activation=None, name="conv3_1"+'_1x1_increase'),
            BatchNormalization(do=self.do_bn, name="conv3_1"+'_1x1_increase_bn'),
        ])
        self.conv3_2 = ICNetBlock(self.f2, self.f8, dilation_rate=1, do_bn=self.do_bn, name='conv3_2')
        self.conv3_3 = ICNetBlock(self.f2, self.f8, dilation_rate=1, do_bn=self.do_bn, name='conv3_3')
        self.conv3_4 = ICNetBlock(self.f2, self.f8, dilation_rate=1, do_bn=self.do_bn, name='conv3_4')

        self.block5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.f16, 1, 1, use_bias=False, activation=None, name='conv4_1_1x1_proj'),
            BatchNormalization(do=self.do_bn, name='conv4_1_1x1_proj_bn')
        ])
        self.conv4_1 = ICNetBlock(self.f4, self.f16, dilation_rate=2, do_bn=self.do_bn, name='conv4_1')
        self.conv4_2 = ICNetBlock(self.f4, self.f16, dilation_rate=2, do_bn=self.do_bn, name='conv4_2')
        self.conv4_3 = ICNetBlock(self.f4, self.f16, dilation_rate=2, do_bn=self.do_bn, name='conv4_3')
        self.conv4_4 = ICNetBlock(self.f4, self.f16, dilation_rate=2, do_bn=self.do_bn, name='conv4_4')
        self.conv4_5 = ICNetBlock(self.f4, self.f16, dilation_rate=2, do_bn=self.do_bn, name='conv4_5')
        self.conv4_6 = ICNetBlock(self.f4, self.f16, dilation_rate=2, do_bn=self.do_bn, name='conv4_6')

        self.block6 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.f32, 1, 1, use_bias=False, activation=None, name='conv5_1_1x1_proj'),
            BatchNormalization(do=self.do_bn, name='conv5_1_1x1_proj_bn')
        ])

        self.conv5_1 = ICNetBlock(self.f8, self.f32, dilation_rate=4, do_bn=self.do_bn, name='conv5_1')
        self.conv5_2 = ICNetBlock(self.f8, self.f32, dilation_rate=4, do_bn=self.do_bn, name='conv5_2')
        self.conv5_3 = ICNetBlock(self.f8, self.f32, dilation_rate=4, do_bn=self.do_bn, name='conv5_3')

        self.block7 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.f8, 1, 1, use_bias=False, activation=None, name='conv5_4_k1'),
            BatchNormalization(do=self.do_bn),
            tf.keras.layers.ReLU(name='conv5_4_k1_bn'),
            tf.keras.layers.UpSampling2D(2, interpolation=interp, name='conv5_4_interp')
        ])

        self.block8 = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=2),
            tf.keras.layers.Conv2D(self.f4, 3, dilation_rate=2, use_bias=False, activation=None, name='conv_sub4'),
            BatchNormalization(do=self.do_bn, name='conv_sub4_bn')
        ])

        self.block9 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.f4, 1, 1, use_bias=False, activation=None, name='conv3_1_sub2_proj'),
            BatchNormalization(do=self.do_bn, name='conv3_1_sub2_proj_bn')
        ])

        self.block10 = tf.keras.Sequential([
            tf.keras.layers.ReLU(name='sub24_sum/relu'),
            tf.keras.layers.UpSampling2D(2, interpolation=interp, name='sub24_sum_interp')
        ])

        self.block11 = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(padding=2),
            tf.keras.layers.Conv2D(self.f4, 3, dilation_rate=2, use_bias=False, activation=None, name='conv_sub2'),
            BatchNormalization(do=self.do_bn)
        ])

        self.block12 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.f1, 3, strides=2, use_bias=False, padding='SAME', activation=None, name='conv1_sub1'),
            BatchNormalization(do=self.do_bn),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.f1, 3, strides=2, use_bias=False, padding='SAME', activation=None, name='conv2_sub1'),
            BatchNormalization(do=self.do_bn),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.f2, 3, strides=2, use_bias=False, padding='SAME', activation=None, name='conv3_sub1'),
            BatchNormalization(do=self.do_bn),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.f4, 1, 1, use_bias=False, padding='VALID', activation=None, name='conv3_sub1_proj'),
            BatchNormalization(do=self.do_bn, name='conv3_sub1_proj_bn'),
            tf.keras.layers.UpSampling2D(2, interpolation=interp) # added
        ])

        self.block13 = tf.keras.Sequential([
            tf.keras.layers.ReLU(),
            tf.keras.layers.UpSampling2D(2, interpolation=interp, name='sub12_sum_interp')
        ])

    def call(self, inputs):
        half_inputs = inputs if self.skip_first_bilinear_resize else tf.image.resize(inputs, size=tf.shape(inputs)[1:3]//2, method=tf.image.ResizeMethod.BILINEAR)
        pool1_3x3_s2 = self.block1(half_inputs)
        conv2_1_1x1_proj_bn = self.block2(pool1_3x3_s2)

        conv2_1_1x1_increase_bn = self.conv2_1(pool1_3x3_s2)
        conv2_1_relu = tf.keras.layers.ReLU(name='conv2_1/relu')(conv2_1_1x1_proj_bn + conv2_1_1x1_increase_bn)
        conv2_2_1x1_increase_bn = self.conv2_2(conv2_1_relu)
        conv2_2_relu = tf.keras.layers.ReLU(name='conv2_2/relu')(conv2_1_relu + conv2_2_1x1_increase_bn)
        conv2_3_1x1_increase_bn = self.conv2_3(conv2_2_relu)

        conv2_3_relu = tf.keras.layers.ReLU(name='conv2_3/relu')(conv2_2_relu + conv2_3_1x1_increase_bn)
        conv3_1_1x1_proj_bn = self.block3(conv2_3_relu)

        conv3_1_1x1_increase_bn = self.block4(conv2_3_relu)

        conv3_1_relu = tf.keras.layers.ReLU(name='conv3_1/relu')(conv3_1_1x1_proj_bn + conv3_1_1x1_increase_bn)
        #conv3_1_sub4 = tf.keras.layers.MaxPool2D(2)(conv3_1_relu)
        size = tf.shape(conv3_1_relu)[1:3]//2
        conv3_1_sub4 = tf.image.resize(conv3_1_relu, size=size, method=tf.image.ResizeMethod.BILINEAR)

        conv3_2_1x1_increase_bn = self.conv3_2(conv3_1_sub4)
        conv3_2_relu = tf.keras.layers.ReLU(name='conv3_2/relu')(conv3_1_sub4 + conv3_2_1x1_increase_bn)

        conv3_3_1x1_increase_bn = self.conv3_3(conv3_2_relu)
        conv3_3_relu = tf.keras.layers.ReLU(name='conv3_3/relu')(conv3_2_relu + conv3_3_1x1_increase_bn)

        conv3_4_1x1_increase_bn = self.conv3_4(conv3_3_relu)
        conv3_4_relu = tf.keras.layers.ReLU(name='conv3_4/relu')(conv3_3_relu + conv3_4_1x1_increase_bn)

        conv4_1_1x1_proj_bn = self.block5(conv3_4_relu)

        conv4_1_1x1_increase_bn = self.conv4_1(conv3_4_relu)
        conv4_1_relu = tf.keras.layers.ReLU(name='conv4_1/relu')(conv4_1_1x1_proj_bn + conv4_1_1x1_increase_bn)

        conv4_2_1x1_increase_bn = self.conv4_2(conv4_1_relu)
        conv4_2_relu = tf.keras.layers.ReLU(name='conv4_2/relu')(conv4_1_relu + conv4_2_1x1_increase_bn)

        conv4_3_1x1_increase_bn = self.conv4_3(conv4_2_relu)
        conv4_3_relu = tf.keras.layers.ReLU(name='conv4_3/relu')(conv4_2_relu + conv4_3_1x1_increase_bn)

        conv4_4_1x1_increase_bn = self.conv4_4(conv4_3_relu)
        conv4_4_relu = tf.keras.layers.ReLU(name='conv4_4/relu')(conv4_3_relu + conv4_4_1x1_increase_bn)

        conv4_5_1x1_increase_bn = self.conv4_5(conv4_4_relu)
        conv4_5_relu = tf.keras.layers.ReLU(name='conv4_5/relu')(conv4_4_relu + conv4_5_1x1_increase_bn)

        conv4_6_1x1_increase_bn = self.conv4_6(conv4_5_relu)
        conv4_6_relu = tf.keras.layers.ReLU(name='conv4_6/relu')(conv4_5_relu + conv4_6_1x1_increase_bn)

        conv5_1_1x1_proj_bn = self.block6(conv4_6_relu)

        conv5_1_1x1_increase_bn = self.conv5_1(conv4_6_relu)
        conv5_1_relu = tf.keras.layers.ReLU(name='conv5_1/relu')(conv5_1_1x1_proj_bn + conv5_1_1x1_increase_bn)

        conv5_2_1x1_increase_bn = self.conv5_2(conv5_1_relu)
        conv5_2_relu = tf.keras.layers.ReLU(name='conv5_2/relu')(conv5_1_relu + conv5_2_1x1_increase_bn)

        conv5_3_1x1_increase_bn = self.conv5_3(conv5_2_relu)
        conv5_3_relu = tf.keras.layers.ReLU(name='conv5_3/relu')(conv5_2_relu + conv5_3_1x1_increase_bn)

        shape_static = conv5_3_relu.get_shape().as_list()[1:3]
        hs, ws = shape_static
        shape_as_tensor = tf.shape(conv5_3_relu)[1:3]

        conv5_3_pool1_interp = AvgSPP(1, name='conv5_3_pool1_spp')(conv5_3_relu)
        conv5_3_pool2_interp = AvgSPP(2, name='conv5_3_pool2_spp')(conv5_3_relu)
        conv5_3_pool3_interp = AvgSPP(3, name='conv5_3_pool3_spp')(conv5_3_relu)
        conv5_3_pool6_interp = AvgSPP(6, name='conv5_3_pool6_spp')(conv5_3_relu)

        conv5_4_interp = self.block7(conv5_3_relu + conv5_3_pool6_interp + conv5_3_pool3_interp + conv5_3_pool2_interp + conv5_3_pool1_interp)

        conv_sub4_bn = self.block8(conv5_4_interp)
        conv3_1_sub2_proj_bn = self.block9(conv3_1_relu)
        sub24_sum_interp = self.block10(conv_sub4_bn + conv3_1_sub2_proj_bn)

        conv_sub2_bn = self.block11(sub24_sum_interp)
        conv3_sub1_proj_bn = self.block12(inputs)
        sub12_sum_interp = self.block13(conv_sub2_bn + conv3_sub1_proj_bn)

        return conv5_4_interp, sub24_sum_interp, sub12_sum_interp

class ICNetLoss():
    def __init__(self, ignore_label):
        self.ignore_label = ignore_label
    def create_loss(self, output, label, num_classes):
        raw_pred = tf.reshape(output, [-1, num_classes])
        label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
        resized_label = label
        label = tf.reshape(label, [-1,])

        indices = get_mask(label, num_classes, self.ignore_label)
        gt = tf.cast(tf.gather(label, indices), tf.int32)
        pred = tf.gather(raw_pred, indices)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)
        reduced_loss = tf.reduce_mean(loss)

        return reduced_loss, resized_label

    def __call__(self, chunk):
        batch_target = chunk["batch_target"]
        num_classes = batch_target.shape[3]

        # Get output from different branches
        sub4_out = chunk['sub4_out']
        sub24_out = chunk['sub24_out']
        sub124_out = chunk['sub124_out']

        loss_sub4, label_sub4 = create_loss(sub4_out, batch_target, num_classes, ignore_label)
        loss_sub24, label_sub24 = create_loss(sub24_out, batch_target, num_classes, ignore_label)
        loss_sub124, label_sub124 = create_loss(sub124_out, batch_target, num_classes, ignore_label)

        l2_losses = [self.cfg.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]

        # Calculate weighted loss of three branches, you can tune LAMBDA values to get better results.
        reduced_loss = self.cfg.LAMBDA1 * loss_sub4 +  self.cfg.LAMBDA2 * loss_sub24 + self.cfg.LAMBDA3 * loss_sub124 + tf.add_n(l2_losses)

        chunk["loss"] = reduced_loss

