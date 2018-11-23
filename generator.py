import cv2
import copy
import numpy as np
from keras.utils import Sequence
from utils.bbox import BoundBox, bbox_iou
from utils.image import apply_random_scale_and_crop, random_distort_image, random_flip, correct_bounding_boxes

class BatchGenerator(Sequence):
    def __init__(self, 
        instances,            # 训练样本，其结构参见 train.py 之 create_training_instances()
        anchors,              # 先验框，[55,69, 75,234, 133,240, 136,129, 142,363, 203,290, 228,184, 285,359, 341,260]
        labels,               # 通常就是config['model']['labels']，比如["raccoon"]；如果没有指定，则为样本图像中的所有对象。
        downsample=32,        # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=30, # 每张图像中最多有几个对象。是根据样本中的对象标注信息统计的来。
        batch_size=1,
        min_net_size=320,     # config['model']['min_input_size']，输入图像的最小尺寸（宽和高）
        max_net_size=608,     # config['model']['max_input_size']，输入图像的最大尺寸（宽和高）
        shuffle=True, 
        jitter=True, 
        norm=None
    ):
        self.instances          = instances
        self.batch_size         = batch_size
        self.labels             = labels
        self.downsample         = downsample
        self.max_box_per_image  = max_box_per_image
        self.min_net_size       = (min_net_size//self.downsample)*self.downsample
        self.max_net_size       = (max_net_size//self.downsample)*self.downsample
        self.shuffle            = shuffle
        self.jitter             = jitter
        self.norm               = norm
        self.anchors            = [BoundBox(0, 0, anchors[2*i], anchors[2*i+1]) for i in range(len(anchors)//2)] # 9个BoundBox
        self.net_h              = 416  
        self.net_w              = 416

        if shuffle: np.random.shuffle(self.instances)
            
    def __len__(self):
        return int(np.ceil(float(len(self.instances))/self.batch_size))           

    """
    每次构造一个batch的训练样本。Gets batch at position `index`.
    input
        index: position of the batch in the Sequence.
    return
        一个batch：[x_batch, t_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]
        x_batch: 输入图像
        t_batch: 输入图像中所有实际对象的bounding box
        yolo_1,yolo_2,yolo_3: 标签y，分别是32、16、8倍下采样后期望输出的张量
        dummy_yolo_1, dummy_yolo_2, dummy_yolo_3: dummy标签y，因为yolo_1,yolo_2,yolo_3输出了真实的标签y，所以这里返回的就是占位变量。
    """
    def __getitem__(self, idx):
        # get image input size, change every 10 batches
        # net_h, net_w 是输入图像的高宽，每10个batch随机变换一次
        net_h, net_w = self._get_net_size(idx)
        # 32倍下采样的特征图的高宽
        base_grid_h, base_grid_w = net_h//self.downsample, net_w//self.downsample

        # determine the first and the last indices of the batch
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size

        # 这个感觉不是很合理
        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        # 准备样本，一个batch的输入图像
        x_batch = np.zeros((r_bound - l_bound, net_h, net_w, 3))             # input images
        # 每个图像中的所有对象边框，shape=(batch,1,1,1,一个图像中最多几个对象,4个坐标)
        t_batch = np.zeros((r_bound - l_bound, 1, 1, 1,  self.max_box_per_image, 4))   # list of groundtruth boxes

        # initialize the inputs and the outputs，分别对应32、16、8倍下采样的输出特征图
        # [batch_size，特征图高，特征图宽，anchor数量3，边框坐标4+置信度1+预测对象类别数]
        yolo_1 = np.zeros((r_bound - l_bound, 1*base_grid_h,  1*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 1
        yolo_2 = np.zeros((r_bound - l_bound, 2*base_grid_h,  2*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 2
        yolo_3 = np.zeros((r_bound - l_bound, 4*base_grid_h,  4*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 3

        # 8、16、32倍下采样对应到先验框 [55,69, 75,234, 133,240,   136,129, 142,363, 203,290,   228,184, 285,359, 341,260]
        yolos = [yolo_3, yolo_2, yolo_1]

        dummy_yolo_1 = np.zeros((r_bound - l_bound, 1))
        dummy_yolo_2 = np.zeros((r_bound - l_bound, 1))
        dummy_yolo_3 = np.zeros((r_bound - l_bound, 1))

        instance_count = 0  # batch中的第几张图像
        true_box_index = 0  # 图像中的第几个对象

        # do the logic to fill in the inputs and the output
        for train_instance in self.instances[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self._aug_image(train_instance, net_h, net_w)
            
            for obj in all_objs:
                # find the best anchor box for this object
                max_anchor = None  # IOU最大的那个anchor
                max_index  = -1     # IOU最大的那个anchor 的index
                max_iou    = -1

                shifted_box = BoundBox(0, 
                                       0,
                                       obj['xmax']-obj['xmin'],                                                
                                       obj['ymax']-obj['ymin'])    
                
                for i in range(len(self.anchors)):
                    anchor = self.anchors[i]
                    iou    = bbox_iou(shifted_box, anchor)

                    if max_iou < iou:
                        max_anchor = anchor
                        max_index  = i
                        max_iou    = iou                
                
                # determine the yolo to be responsible for this bounding box
                # 3种尺度的特征图，与当前对象最匹配的那种anchor，所属的那个特征图的tensor，就是这里的yolo
                yolo = yolos[max_index//3]
                grid_h, grid_w = yolo.shape[1:3]
                
                # determine the position of the bounding box on the grid
                # 对象的边框中心坐标 被转换到 特征图网格上，其值相当于 期望预测的坐标 sigma(t_x) + c_x，sigma(t_y) + c_y
                center_x = .5*(obj['xmin'] + obj['xmax'])
                center_x = center_x / float(net_w) * grid_w # 期望预测的坐标 sigma(t_x) + c_x = center_x
                center_y = .5*(obj['ymin'] + obj['ymax'])
                center_y = center_y / float(net_h) * grid_h # 期望预测的坐标 sigma(t_y) + c_y = center_y
                
                # determine the sizes of the bounding box
                w = np.log((obj['xmax'] - obj['xmin']) / float(max_anchor.xmax)) # t_w，注：truth_w = anchor_w * exp(t_w)
                h = np.log((obj['ymax'] - obj['ymin']) / float(max_anchor.ymax)) # t_h，注：truth_h = anchor_h * exp(t_h)

                box = [center_x, center_y, w, h]

                # determine the index of the label
                obj_indx = self.labels.index(obj['name'])  

                # determine the location of the cell responsible for this object
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                # assign ground truth x, y, w, h, confidence and class probs to y_batch
                # max_index%3 对应到最佳匹配的anchor，一个对象仅有一个anchor负责检测
                yolo[instance_count, grid_y, grid_x, max_index%3]      = 0
                yolo[instance_count, grid_y, grid_x, max_index%3, 0:4] = box      # 边框坐标
                yolo[instance_count, grid_y, grid_x, max_index%3, 4  ] = 1.       # 边框置信度
                yolo[instance_count, grid_y, grid_x, max_index%3, 5+obj_indx] = 1 # 对象分类

                # assign the true box to t_batch. true_box的x、y是特征图上的坐标（比如13*13特征图），宽和高是原始图像上对象的宽和高
                true_box = [center_x, center_y, obj['xmax'] - obj['xmin'], obj['ymax'] - obj['ymin']]
                t_batch[instance_count, 0, 0, 0, true_box_index] = true_box
                # 因为有 instance_count 区分不同的图像，true_box_index 应该只需在每次图像切换时 true_box_index=0 即可。这里在整个batch累加true_box_index，暂不确定是否有特别的用意。
                true_box_index += 1
                true_box_index  = true_box_index % self.max_box_per_image    

            # assign input image to x_batch
            if self.norm != None: 
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    cv2.rectangle(img, (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                    cv2.putText(img, obj['name'], 
                                (obj['xmin']+2, obj['ymin']+12), 
                                0, 1.2e-3 * img.shape[0], 
                                (0,255,0), 2)
                
                x_batch[instance_count] = img

            # increase instance counter in the current batch
            instance_count += 1                 
                
        return [x_batch, t_batch, yolo_1, yolo_2, yolo_3], [dummy_yolo_1, dummy_yolo_2, dummy_yolo_3]

    def _get_net_size(self, idx):
        if idx%10 == 0:
            net_size = self.downsample*np.random.randint(self.min_net_size/self.downsample, \
                                                         self.max_net_size/self.downsample+1)
            print("resizing: ", net_size, net_size)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w
    
    def _aug_image(self, instance, net_h, net_w):
        image_name = instance['filename']
        image = cv2.imread(image_name) # RGB image
        
        if image is None: print('Cannot find ', image_name)
        image = image[:,:,::-1] # RGB image
            
        image_h, image_w, _ = image.shape
        
        # determine the amount of scaling and cropping
        dw = self.jitter * image_w;
        dh = self.jitter * image_h;

        new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh));
        scale = np.random.uniform(0.25, 2);

        if (new_ar < 1):
            new_h = int(scale * net_h);
            new_w = int(net_h * new_ar);
        else:
            new_w = int(scale * net_w);
            new_h = int(net_w / new_ar);
            
        dx = int(np.random.uniform(0, net_w - new_w));
        dy = int(np.random.uniform(0, net_h - new_h));
        
        # apply scaling and cropping
        im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)
        
        # randomly distort hsv space
        im_sized = random_distort_image(im_sized)
        
        # randomly flip
        flip = np.random.randint(2)
        im_sized = random_flip(im_sized, flip)
            
        # correct the size and pos of bounding boxes
        all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h)
        
        return im_sized, all_objs   

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.instances)
            
    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.instances)    

    def get_anchors(self):
        anchors = []

        for anchor in self.anchors:
            anchors += [anchor.xmax, anchor.ymax]

        return anchors

    def load_annotation(self, i):
        annots = []

        for obj in self.instances[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.labels.index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.instances[i]['filename'])     