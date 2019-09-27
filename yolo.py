from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.engine.topology import Layer     # 等价于keras.layers.Layer
import tensorflow as tf

# 继承keras.layers.Layer实现自定义层，常用的有三个方法：
# __init__, 在这里初始化与层输入无关的变量，这里创建的张量要明确指定形状
# build, 在这里定义该层的权重(张量)；且该层有input_shape参数，表示从上一层输入张量的shape
# call，在这里进行正向传播，该层有一个input参数，tensor, list/tuple of tensors, 表示上一层输入的张量或张量列表
class YoloLayer(Layer):
    def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh, 
                    grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale, 
                    **kwargs):
        # make the model settings persistent
        self.ignore_thresh  = ignore_thresh  # config['train']['ignore_thresh']，0.5
        self.warmup_batches = warmup_batches # config['train']['warmup_epochs'] * (config['train']['train_times'] * len(train_generator))
        self.anchors        = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2]) # 每个尺度的特征图设置3个先验框，比如[[17,18],[28,24],[36,34]]
        self.grid_scale     = grid_scale     # config['train']['grid_scales']，3个尺度的特征图的loss的权重
        self.obj_scale      = obj_scale      # config['train']['obj_scale']，有对象的loss的权重。由于图像中对象比较少，一般会提升权重，比如5
        self.noobj_scale    = noobj_scale    # config['train']['noobj_scale']，不存在对象的loss的权重
        self.xywh_scale     = xywh_scale     # config['train']['xywh_scale']，边框位置的loss的权重
        self.class_scale    = class_scale    # config['train']['class_scale']，对象分类loss的权重

        # make a persistent mesh grid
        # loss_yolo_1（大视野特征图）是输入图像的最大尺寸 [448, 448]，[config['model']['max_input_size'], config['model']['max_input_size']]
        # loss_yolo_2（中视野特征图）是输入图像的最大尺寸 [448, 448]*2
        # loss_yolo_3（小视野特征图）是输入图像的最大尺寸 [448, 448]*4
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        # shape=(batch, 图像最大宽, 图像最大高, 3个anchor, 2个grid坐标)
        self.cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(YoloLayer, self).build(input_shape)  # Be sure to call this somewhere!

    """
    一个神经网络层的计算，实际上在计算loss。
    YOLOv3输出3个尺度的特征图，这里是对1个尺度的特征图计算loss。
    input
        x = input_image, y_pred, y_true, true_boxes
        分别是：输入图像，YOLO输出的tensor，标签y（期望其输出的tensor），输入图像中所有ground truth box。
    return
        loss = 边框位置xy loss + 边框位置wh loss + 边框置信度loss + 对象分类loss
    """
    def call(self, x):
        # true_boxes 对应 BatchGenerator 里面的 t_batch，shape=(batch,1,1,1,一个图像中最多几个对象,4个坐标)
        # y_true 对应 BatchGenerator 里面的 yolo_1/yolo_2/yolo_3，即一个特征图tensor
        input_image, y_pred, y_true, true_boxes = x

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        # shape=(batch, 特征图高，特征图宽，3个anchor，4个边框坐标+1个置信度+检测对象类别数)
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))
        
        # initialize the masks
        # object_mask 是一个特征图上所有预测框的置信度（objectness），这里来自标签y_true，除了负责检测对象的那些anchor，其它置信度都是0。
        # shape = (batch, 特征图高，特征图宽，3个anchor，1个置信度)
        # y_true[..., 4]提取边框置信度（最后一维tensor中，前4个是边框坐标，第5个就是置信度），expand_dims将其恢复到原来的tensor形状。
        object_mask     = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)        

        # compute grid factor and net factor
        # 特征图的宽高
        grid_h      = tf.shape(y_true)[1]
        grid_w      = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])

        # 输入图像的宽高
        net_h       = tf.shape(input_image)[1]
        net_w       = tf.shape(input_image)[2]            
        net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])
        
        """
        Adjust prediction
        """
        # pred_box_xy 是预测框在特征图上的中心点坐标，特征图网格大小归一化为1*1，=(sigma(t_xy) + c_xy)
        pred_box_xy    = (self.cell_grid[:,:grid_h,:grid_w,:,:] + tf.sigmoid(y_pred[..., :2]))  # shape=(batch,特征图高,特征图宽,3预测框,2坐标)
        # pred_box_wh 是预测对象的t_w, t_h。注：truth_wh = anchor_wh * exp(t_wh)
        pred_box_wh    = y_pred[..., 2:4]                                                       # shape=(batch,特征图高,特征图宽,3预测框,2坐标)
        pred_box_conf  = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)                          # shape=(batch,特征图高,特征图宽,3预测框,1confidence)
        pred_box_class = y_pred[..., 5:]                                                        # shape=(batch,特征图高,特征图宽,3预测框,c个对象)

        """
        Adjust ground truth
        """
        # true_box_xy 是实际边框在特征图上的中心点坐标，=(sigma(t_xy) + c_xy)，参见y_true
        true_box_xy    = y_true[..., 0:2]                  # shape=(batch,特征图高,特征图宽,3预测框,2坐标)
        # true_box_wh 是对象的t_w, t_h。注：truth_wh = anchor_wh * exp(t_wh)
        true_box_wh    = y_true[..., 2:4]                  # shape=(batch,特征图高,特征图宽,3预测框,2坐标)
        true_box_conf  = tf.expand_dims(y_true[..., 4], 4) # shape=(batch,特征图高,特征图宽,3预测框,1confidence)
        true_box_class = tf.argmax(y_true[..., 5:], -1)    # shape=(batch,特征图高,特征图宽,3预测框)

        """
        Compare each predicted box to all true boxes
        这一部分是为了计算出IOU低于阈值的那些预测框，也可以理解为找出那些检测到背景的预测框。
        一个特征图上有 宽*高*3anchor 个预测框，YOLO的策略是，一个对象其中心点所在gird的3个anchor，IOU最大的那个anchor负责预测（其confidence=1）该对象。
        但是附近还有一些IOU比较大的anchor，如果要求其confidence=0是不合理的，于是不计入loss也是合理的选择。剩下那些框里面就是背景了，其confidence=0。
        下面先计算出每个预测框对每个真实框的IOU（iou_scores），然后每个预测框选一个最大的IOU，低于阈值的框就认为是背景，将计算loss。
        """
        # initially, drag all objectness of all boxes to 0
        conf_delta  = pred_box_conf - 0 

        # then, ignore the boxes which have good overlap with some true box
        # true_xy,true_wh 的值是相当于将原始图像的宽高归一化为1*1
        true_xy = true_boxes[..., 0:2] / grid_factor  # shape=(batch,1,1,1,一个图像中最多几(3)个对象,2个xy坐标),xy是特征图上的坐标，与y_true中的xy一样
        true_wh = true_boxes[..., 2:4] / net_factor   # shape=(batch,1,1,1,一个图像中最多几(3)个对象,2个wh坐标),wh是原始图像上对象的宽和高
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)                        # shape=(batch,特征图高,特征图宽,3预测框,1,2坐标)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)  # shape=(batch,特征图高,特征图宽,3预测框,1,2坐标)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    

        intersect_mins  = tf.maximum(pred_mins,  true_mins)  # shape=(batch, 特征图高,特征图宽, 3预测框, 一个图像中最多几(3)个对象, 2个坐标)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes) # shape=(batch, 特征图高,特征图宽, 3预测框, 一个图像中最多几(3)个对象, 2个坐标)

        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)  # shape=(batch, 特征图高,特征图宽, 3预测框, 一个图像中最多几(3)个对象, 2个坐标)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]       # shape=(batch, 特征图高,特征图宽, 3预测框, 一个图像中最多几(3)个对象)
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]  # shape=(batch,1,       1,       1,      一个图像中最多几(3)个对象)
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]  # shape=(batch,特征图高,特征图宽,3预测框,1)

        union_areas = pred_areas + true_areas - intersect_areas  # shape=(batch, 特征图高,特征图宽, 3预测框, 一个图像中最多几(3)个对象)
        iou_scores  = tf.truediv(intersect_areas, union_areas)   # shape=(batch, 特征图高,特征图宽, 3预测框, 一个图像中最多几(3)个对象)

        # 每个预测框与最接近的实际对象的IOU
        best_ious   = tf.reduce_max(iou_scores, axis=4)  # shape=(batch, 特征图高,特征图宽, 3预测框)

        # IOU低于阈值的那些预测边框，才计算其（检测到背景的）置信度的loss
        conf_delta *= tf.expand_dims(tf.to_float(best_ious < self.ignore_thresh), 4) # shape=(batch,特征图高,特征图宽,3预测框,1confidence)

        """
        Compute some online statistics
        """            
        true_xy = true_box_xy / grid_factor
        true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half

        pred_xy = pred_box_xy / grid_factor
        pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor 
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half      

        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        iou_scores  = object_mask * tf.expand_dims(iou_scores, 4)
        
        count       = tf.reduce_sum(object_mask)
        count_noobj = tf.reduce_sum(1 - object_mask)
        detect_mask = tf.to_float((pred_box_conf*object_mask) >= 0.5)
        class_mask  = tf.expand_dims(tf.to_float(tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
        recall50    = tf.reduce_sum(tf.to_float(iou_scores >= 0.5 ) * detect_mask  * class_mask) / (count + 1e-3)
        recall75    = tf.reduce_sum(tf.to_float(iou_scores >= 0.75) * detect_mask  * class_mask) / (count + 1e-3)    
        avg_iou     = tf.reduce_sum(iou_scores) / (count + 1e-3)
        avg_obj     = tf.reduce_sum(pred_box_conf  * object_mask)  / (count + 1e-3)
        avg_noobj   = tf.reduce_sum(pred_box_conf  * (1-object_mask))  / (count_noobj + 1e-3)
        avg_cat     = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3) 

        """
        Warm-up training
        """
        batch_seen = tf.assign_add(batch_seen, 1.)
        
        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches+1),
                              # 根据YOLOv2开始的设计，前self.warmup_batches 个batch 计算的是预测框与先验框的误差，不是与真实对象边框的误差。
                              # 但这里代码好像有点问题。
                              lambda: [true_box_xy + (0.5 + self.cell_grid[:,:grid_h,:grid_w,:,:]) * (1-object_mask), 
                                       true_box_wh + tf.zeros_like(true_box_wh) * (1-object_mask),   # zeros_like 导致后面的项为0，实际还是true_box_wh，需要修改
                                       tf.ones_like(object_mask)],                                   # 每个预测框的位置都计入loss
                              # 之后的batch不做特殊处理
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       object_mask])

        """
        Compare each true box to all anchor boxes
        """
        # 注：exp(true_box_wh) = exp(t_wh) = truth_wh / anchor_wh
        # exp(true_box_wh) * self.anchors / net_factor = truth_wh / anchor_wh * self.anchors / net_factor = truth_wh / net_factor
        # wh_scale 是实际对象相对输入图像的大小。
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor   # shape=(batch,特征图高,特征图宽,3anchor,2坐标)
        # wh_scale 与实际对象边框的面积负相关，小尺寸对象对边框误差提升敏感度，the smaller the box, the bigger the scale
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4)

        # 正常情况下（warmup_batches之后），xywh_mask = object_mask，即存在对象的那些预测框（其位置、置信度、对象类型有意义）才计算loss。
        # 不存在对象的那些预测框，其置信度有意义（不过conf_delta已过滤掉了那些IOU超过阈值的边框），计入loss。而位置和对象类型无意义，不计入loss。
        xy_delta    = xywh_mask   * (pred_box_xy-true_box_xy) * wh_scale * self.xywh_scale  # shape=(batch,特征图高,特征图宽,3个预测框,2个位置)
        wh_delta    = xywh_mask   * (pred_box_wh-true_box_wh) * wh_scale * self.xywh_scale  # shape=(batch,特征图高,特征图宽,3个预测框,2个位置)
        # shape=(batch,特征图高,特征图宽,3个预测框,1个置信度)，前一半是检测到对象的置信度，后一半是检测到背景的置信度
        conf_delta  = object_mask * (pred_box_conf-true_box_conf) * self.obj_scale + (1-object_mask) * conf_delta * self.noobj_scale
        # shape=(batch,特征图高,特征图宽,3个预测框,1个交叉熵)
        class_delta = object_mask * \
                      tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * \
                      self.class_scale

        # shape=(batch_size,)
        loss_xy    = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5)))
        loss_wh    = tf.reduce_sum(tf.square(wh_delta),       list(range(1,5)))
        loss_conf  = tf.reduce_sum(tf.square(conf_delta),     list(range(1,5)))
        loss_class = tf.reduce_sum(class_delta,               list(range(1,5)))

        loss = loss_xy + loss_wh + loss_conf + loss_class

        loss = tf.Print(loss, [grid_h, avg_obj], message='avg_obj \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_noobj], message='avg_noobj \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_iou], message='avg_iou \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_cat], message='avg_cat \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, recall50], message='recall50 \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, recall75], message='recall75 \t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, count], message='count \t\t\t', summarize=1000)
        loss = tf.Print(loss, [grid_h, tf.reduce_sum(loss_xy), 
                                       tf.reduce_sum(loss_wh), 
                                       tf.reduce_sum(loss_conf), 
                                       tf.reduce_sum(loss_class)],  message='loss xy, wh, conf, class: \t',   summarize=1000)   

        # loss 的shape=(batch_size,)
        return loss*self.grid_scale

    def compute_output_shape(self, input_shape):
        return [(None, 1)]

def _conv_block(inp, convs, do_skip=True):
    # 这里定义的是一个块(一个_covn_block内最后两层是一个残差块)
    # inp: 输入tensor； convs: 卷积参数； do_skip: True=残差块,False=非残差块
    x = inp
    count = 0
    
    for conv in convs:
        # 如果使用残差结构，需要找到第 -2 层，暂存为skip_connection变量，后面再 add 到卷积后的输出层
        if count == (len(convs) - 2) and do_skip:
            skip_connection = x
        count += 1
        # YOLO3使用步幅为2的卷积操作来替代池化层，这里在输入矩阵的左边有上边各pad一层，确保stride=2，且kernel_size=3时输出矩阵的宽高=输入矩阵宽高/2
        # ((top_pad, bottom_pad), (left_pad, right_pad))
        if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # unlike tensorflow darknet prefer left and top paddings
        x = Conv2D(conv['filter'],  # filters number
                   conv['kernel'],  # kernel size
                   strides=conv['stride'], 
                   padding='valid' if conv['stride'] > 1 else 'same', # unlike tensorflow darknet prefer left and top paddings
                   name='conv_' + str(conv['layer_idx']), 
                   use_bias=False if conv['bnorm'] else True)(x)
        if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
        if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

    return add([skip_connection, x]) if do_skip else x        

def create_yolov3_model(
    nb_class, 
    anchors, 
    max_box_per_image, 
    max_grid, 
    batch_size, 
    warmup_batches,
    ignore_thresh,
    grid_scales, # 3个尺度的特征图的loss的权重
    obj_scale,   # 有对象的loss的权重
    noobj_scale, # 没有对象的loss的权重
    xywh_scale,  # 边框位置loss的权重
    class_scale  # 分类loss的权重
):
    input_image = Input(shape=(None, None, 3)) # net_h, net_w, 3
    true_boxes  = Input(shape=(1, 1, 1, max_box_per_image, 4))  # 对应 BatchGenerator 里面的 t_batch
    true_yolo_1 = Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class，32倍下采样的label
    true_yolo_2 = Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class，16倍下采样的label
    true_yolo_3 = Input(shape=(None, None, len(anchors)//6, 4+1+nb_class)) # grid_h, grid_w, nb_anchor, 5+nb_class，8倍下采样的label

    # Layer  0 => 4
    x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

    # Layer  5 => 8
    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

    # Layer  9 => 11
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

    # Layer 12 => 15
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

    # Layer 16 => 36
    for i in range(7):
        x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
        
    skip_36 = x
        
    # Layer 37 => 40
    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

    # Layer 41 => 61
    for i in range(7):
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
        
    skip_61 = x
        
    # Layer 62 => 65
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

    # Layer 66 => 74
    for i in range(3):
        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
        
    # Layer 75 => 79
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], do_skip=False)

    # Layer 80 => 82
    pred_yolo_1 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                             {'filter': (3*(5+nb_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], do_skip=False)
    loss_yolo_1 = YoloLayer(anchors[12:], # 较大对象的anchor
                            [1*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[0],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_1, true_yolo_1, true_boxes])

    # Layer 83 => 86
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], do_skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])

    # Layer 87 => 91
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], do_skip=False)

    # Layer 92 => 94
    pred_yolo_2 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
                             {'filter': (3*(5+nb_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], do_skip=False)
    loss_yolo_2 = YoloLayer(anchors[6:12],  # 中等对象的anchor
                            [2*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[1],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_2, true_yolo_2, true_boxes])

    # Layer 95 => 98
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], do_skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])

    # Layer 99 => 106
    pred_yolo_3 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
                             {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
                             {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
                             {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
                             {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
                             {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
                             {'filter': (3*(5+nb_class)), 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], do_skip=False)
    loss_yolo_3 = YoloLayer(anchors[:6],  # 较小对象的anchor
                            [4*num for num in max_grid], 
                            batch_size, 
                            warmup_batches, 
                            ignore_thresh, 
                            grid_scales[2],
                            obj_scale,
                            noobj_scale,
                            xywh_scale,
                            class_scale)([input_image, pred_yolo_3, true_yolo_3, true_boxes]) 

    train_model = Model([input_image, true_boxes, true_yolo_1, true_yolo_2, true_yolo_3], [loss_yolo_1, loss_yolo_2, loss_yolo_3])
    infer_model = Model(input_image, [pred_yolo_1, pred_yolo_2, pred_yolo_3])

    return [train_model, infer_model]

def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))