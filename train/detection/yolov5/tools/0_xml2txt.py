import os
import xml.etree.ElementTree as ET
import traceback

CLASSES = ['stone']
XML_DIR = '/data8/user/shanpengfei/EdgeAI-Engine-main/train/object-detection/data/dataset/stone/voc/Annotations'
JPEG_DIR = '/data8/user/shanpengfei/EdgeAI-Engine-main/train/object-detection/data/dataset/stone/voc/JPEGImages'
LABEL_DIR = '/data8/user/shanpengfei/EdgeAI-Engine-main/train/object-detection/data/dataset/stone/yolo/labels_temp'


def parse_xml(fname):
    xml_anno = ET.parse(os.path.join(XML_DIR, fname))
    annos = []
    obj = xml_anno.find('size')
    width = float(obj.find('width').text)
    height = float(obj.find('height').text)

    for obj in xml_anno.findall('object'):
        bndbox_anno = obj.find('bndbox')
        xmin = float(bndbox_anno.find('xmin').text)
        ymin = float(bndbox_anno.find('ymin').text)
        xmax = float(bndbox_anno.find('xmax').text)
        ymax = float(bndbox_anno.find('ymax').text)
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > width: xmax = width
        if ymax > height: ymax = height
        class_name = obj.find('name').text.strip()
        if class_name not in CLASSES:
            continue
        annos.append((class_name, xmin, ymin, xmax, ymax, width, height))
    return annos


def gen_txt(fname, annos):
    os.makedirs(LABEL_DIR, exist_ok=True)
    f = open(os.path.join(LABEL_DIR, fname.replace('.xml', '.txt')), 'w')
    for anno in annos:
        try:
            class_name, xmin, ymin, xmax, ymax, width, height = anno
            cls = CLASSES.index(class_name)
            xmid = (xmin + xmax) / 2 / width
            ymid = (ymin + ymax) / 2 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            f.write(' '.join((str(cls), '%6f' % xmid, '%6f' % ymid, '%6f' % w, '%6f' % h)) + '\n')
        except:
            print(fname)
            traceback.print_exc()
    f.close()


if __name__ == '__main__':
    num = 0
    xml_list = os.listdir(XML_DIR)
    for f in xml_list:
        num += 1
        try:
            annos = parse_xml(os.path.join(f))
            gen_txt(f, annos)
        except:
            traceback.print_exc()
    print('txt ok')
