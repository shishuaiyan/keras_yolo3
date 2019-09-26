import numpy as np
import os
import xml.etree.ElementTree as ET
import pickle

def parse_voc_annotation(ann_dir, img_dir, cache_name, labels=[]):
    '''

    Parameters
    ----------
    ann_dir: annotation注释文件(.xml)目录
    img_dir: 对应图片目录
    cache_name: 保存.pkl文件名(保存在项目根目录下)
    labels: 关注的label列表，忽视非label列表的图片

    Returns
    -------
    all_insts：所有输入img的属性[{'filename':xx, 'height':xx, 'width':xx, 'object':[{'name':xx, 'xmax', 'xmin', 'ymax', 'ymin'}]}]
    seen_labels: 各label对应的annotation个数{labels[0]:xx}
    '''
    if os.path.exists(cache_name):  # 已有pkl文件则直接加载
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:       # 不存在pkl文件，listdir(ann_dir)并依次解析xml文件，并生成pkl文件
        all_insts = []
        seen_labels = {}
        
        for ann in sorted(os.listdir(ann_dir)):
            img = {'object':[]}

            try:
                tree = ET.parse(ann_dir + ann)  # xml文件时一种分级的数据形式，因此可以用ET将其解析为一棵树
            except Exception as e:
                print(e)
                print('Ignore this bad annotation: ' + ann_dir + ann)
                continue
            
            for elem in tree.iter():    # iter()方法对子节点进行深度优先遍历
                if 'filename' in elem.tag:
                    img['filename'] = img_dir + elem.text
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}
                    
                    for attr in list(elem):     # list所有elem的子节点
                        if 'name' in attr.tag:  # type(attr)=Element   attr.tag='name'当前节点的标签
                            obj['name'] = attr.text

                            if obj['name'] in seen_labels:
                                seen_labels[obj['name']] += 1
                            else:
                                seen_labels[obj['name']] = 1
                            
                            if len(labels) > 0 and obj['name'] not in labels:
                                break
                            else:
                                img['object'] += [obj]
                                
                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text)))

            if len(img['object']) > 0:
                all_insts += [img]

        cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
        with open(cache_name, 'wb') as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                        
    return all_insts, seen_labels