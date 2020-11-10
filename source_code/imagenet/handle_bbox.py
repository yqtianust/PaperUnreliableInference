import xml.etree.ElementTree as ET
import os


def GetItem(name, root, index=0):
    count = 0
    for item in root.iter(name):
        if count == index:
            return item.text
        count += 1
    # Failed to find "index" occurrence of item.
    return -1


def GetInt(name, root, index=0):
    return int(GetItem(name, root, index))


def FindNumberBoundingBoxes(root):
    index = 0
    while True:
        if GetInt('xmin', root, index) == -1:
            break
        index += 1
    return index


class BoundingBox(object):
    pass


def get_gt_synset_label(img_file):
    for l in open("./filename_label.txt", 'r').readlines():
        if img_file in l:
        # filename = l.split("; ")[0][0:l.split("; ")[0].index(".")]
        # filename = l.split("; ")[0]
            label_gt = l.split("; ")[1].rstrip()
        # filename_label[filename] = label_gt.replace(" ", "_")
            return label_gt


def get_bbox(jpeg_filename):
    return process_file(os.path.join("val2012_xml", jpeg_filename.replace("JPEG", "xml")))

def process_file(xml_file):
    # print(xml_file)
    tree = ET.parse(xml_file)

    # pylint: enable=broad-except
    root = tree.getroot()

    num_boxes = FindNumberBoundingBoxes(root)
    # print(num_boxes)
    boxes = []

    filename = GetItem('filename', root) + '.JPEG'
    synset_label = get_gt_synset_label(filename)

    for index in range(num_boxes):
        box = BoundingBox()
        # Grab the 'index' annotation.
        box.xmin = GetInt('xmin', root, index)
        box.ymin = GetInt('ymin', root, index)
        box.xmax = GetInt('xmax', root, index)
        box.ymax = GetInt('ymax', root, index)

        box.width = GetInt('width', root)
        box.height = GetInt('height', root)
        box.filename = GetItem('filename', root) + '.JPEG'
        box.label = GetItem('name', root)
        assert box.label == synset_label

        xmin = float(box.xmin) / float(box.width)
        xmax = float(box.xmax) / float(box.width)
        ymin = float(box.ymin) / float(box.height)
        ymax = float(box.ymax) / float(box.height)

        # Some images contain bounding box annotations that
        # extend outside of the supplied image. See, e.g.
        # n03127925/n03127925_147.xml
        # Additionally, for some bounding boxes, the min > max
        # or the box is entirely outside of the image.
        min_x = min(xmin, xmax)
        max_x = max(xmin, xmax)
        box.xmin_scaled = min(max(min_x, 0.0), 1.0)
        box.xmax_scaled = min(max(max_x, 0.0), 1.0)

        min_y = min(ymin, ymax)
        max_y = max(ymin, ymax)
        box.ymin_scaled = min(max(min_y, 0.0), 1.0)
        box.ymax_scaled = min(max(max_y, 0.0), 1.0)

        boxes.append(box)

    return boxes


if __name__ == '__main__':
    for r, d, fs in os.walk("val2012_xml"):
        for f in fs:
            bbox = process_file(os.path.join(r, f))