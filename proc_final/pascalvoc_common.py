# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Pascal VOC Dataset (images + annotations).
"""
import os

import tensorflow as tf
import dataset_utils

slim = tf.contrib.slim

TSINGHUA_TECENT = {'i1': (1, 'i1'), 'i10': (2, 'i10'), 'i11': (3, 'i11'), 'i12': (4, 'i12'), 'i13': (5, 'i13'), 'i14': (6, 'i14'), 'i15': (7, 'i15'), 'i2': (8, 'i2'), 'i3': (9, 'i3'), 'i4': (10, 'i4'), 'i5': (11, 'i5'), 'il100': (12, 'il100'), 'il110': (13, 'il110'), 'il50': (14, 'il50'), 'il60': (15, 'il60'), 'il70': (16, 'il70'), 'il80': (17, 'il80'), 'il90': (18, 'il90'), 'io': (19, 'io'), 'ip': (20, 'ip'), 'p1': (21, 'p1'), 'p10': (22, 'p10'), 'p11': (23, 'p11'), 'p12': (24, 'p12'), 'p13': (25, 'p13'), 'p14': (26, 'p14'), 'p15': (27, 'p15'), 'p16': (28, 'p16'), 'p17': (29, 'p17'), 'p18': (30, 'p18'), 'p19': (31, 'p19'), 'p2': (32, 'p2'), 'p20': (33, 'p20'), 'p21': (34, 'p21'), 'p22': (35, 'p22'), 'p23': (36, 'p23'), 'p24': (37, 'p24'), 'p25': (38, 'p25'), 'p26': (39, 'p26'), 'p27': (40, 'p27'), 'p28': (41, 'p28'), 'p3': (42, 'p3'), 'p4': (43, 'p4'), 'p5': (44, 'p5'), 'p6': (45, 'p6'), 'p7': (46, 'p7'), 'p8': (47, 'p8'), 'p9': (48, 'p9'), 'pa10': (49, 'pa10'), 'pa12': (50, 'pa12'), 'pa13': (51, 'pa13'), 'pa14': (52, 'pa14'), 'pa8': (53, 'pa8'), 'pb': (54, 'pb'), 'pc': (55, 'pc'), 'pg': (56, 'pg'), 'ph1.5': (57, 'ph1.5'), 'ph2': (58, 'ph2'), 'ph2.1': (59, 'ph2.1'), 'ph2.2': (60, 'ph2.2'), 'ph2.4': (61, 'ph2.4'), 'ph2.5': (62, 'ph2.5'), 'ph2.8': (63, 'ph2.8'), 'ph2.9': (64, 'ph2.9'), 'ph3': (65, 'ph3'), 'ph3.2': (66, 'ph3.2'), 'ph3.5': (67, 'ph3.5'), 'ph3.8': (68, 'ph3.8'), 'ph4': (69, 'ph4'), 'ph4.2': (70, 'ph4.2'), 'ph4.3': (71, 'ph4.3'), 'ph4.5': (72, 'ph4.5'), 'ph4.8': (73, 'ph4.8'), 'ph5': (74, 'ph5'), 'ph5.3': (75, 'ph5.3'), 'ph5.5': (76, 'ph5.5'), 'pl10': (77, 'pl10'), 'pl100': (78, 'pl100'), 'pl110': (79, 'pl110'), 'pl120': (80, 'pl120'), 'pl15': (81, 'pl15'), 'pl20': (82, 'pl20'), 'pl25': (83, 'pl25'), 'pl30': (84, 'pl30'), 'pl35': (85, 'pl35'), 'pl40': (86, 'pl40'), 'pl5': (87, 'pl5'), 'pl50': (88, 'pl50'), 'pl60': (89, 'pl60'), 'pl65': (90, 'pl65'), 'pl70': (91, 'pl70'), 'pl80': (92, 'pl80'), 'pl90': (93, 'pl90'), 'pm10': (94, 'pm10'), 'pm13': (95, 'pm13'), 'pm15': (96, 'pm15'), 'pm1.5': (97, 'pm1.5'), 'pm2': (98, 'pm2'), 'pm20': (99, 'pm20'), 'pm25': (100, 'pm25'), 'pm30': (101, 'pm30'), 'pm35': (102, 'pm35'), 'pm40': (103, 'pm40'), 'pm46': (104, 'pm46'), 'pm5': (105, 'pm5'), 'pm50': (106, 'pm50'), 'pm55': (107, 'pm55'), 'pm8': (108, 'pm8'), 'pn': (109, 'pn'), 'pne': (110, 'pne'), 'po': (111, 'po'), 'pr10': (112, 'pr10'), 'pr100': (113, 'pr100'), 'pr20': (114, 'pr20'), 'pr30': (115, 'pr30'), 'pr40': (116, 'pr40'), 'pr45': (117, 'pr45'), 'pr50': (118, 'pr50'), 'pr60': (119, 'pr60'), 'pr70': (120, 'pr70'), 'pr80': (121, 'pr80'), 'ps': (122, 'ps'), 'pw2': (123, 'pw2'), 'pw2.5': (124, 'pw2.5'), 'pw3': (125, 'pw3'), 'pw3.2': (126, 'pw3.2'), 'pw3.5': (127, 'pw3.5'), 'pw4': (128, 'pw4'), 'pw4.2': (129, 'pw4.2'), 'pw4.5': (130, 'pw4.5'), 'w1': (131, 'w1'), 'w10': (132, 'w10'), 'w12': (133, 'w12'), 'w13': (134, 'w13'), 'w16': (135, 'w16'), 'w18': (136, 'w18'), 'w20': (137, 'w20'), 'w21': (138, 'w21'), 'w22': (139, 'w22'), 'w24': (140, 'w24'), 'w28': (141, 'w28'), 'w3': (142, 'w3'), 'w30': (143, 'w30'), 'w31': (144, 'w31'), 'w32': (145, 'w32'), 'w34': (146, 'w34'), 'w35': (147, 'w35'), 'w37': (148, 'w37'), 'w38': (149, 'w38'), 'w41': (150, 'w41'), 'w42': (151, 'w42'), 'w43': (152, 'w43'), 'w44': (153, 'w44'), 'w45': (154, 'w45'), 'w46': (155, 'w46'), 'w47': (156, 'w47'), 'w48': (157, 'w48'), 'w49': (158, 'w49'), 'w5': (159, 'w5'), 'w50': (160, 'w50'), 'w55': (161, 'w55'), 'w56': (162, 'w56'), 'w57': (163, 'w57'), 'w58': (164, 'w58'), 'w59': (165, 'w59'), 'w60': (166, 'w60'), 'w62': (167, 'w62'), 'w63': (168, 'w63'), 'w66': (169, 'w66'), 'w8': (170, 'w8'), 'wo': (171, 'wo'), 'i6': (172, 'i6'), 'i7': (173, 'i7'), 'i8': (174, 'i8'), 'i9': (175, 'i9'), 'ilx': (176, 'ilx'), 'p29': (177, 'p29'), 'w29': (178, 'w29'), 'w33': (179, 'w33'), 'w36': (180, 'w36'), 'w39': (181, 'w39'), 'w4': (182, 'w4'), 'w40': (183, 'w40'), 'w51': (184, 'w51'), 'w52': (185, 'w52'), 'w53': (186, 'w53'), 'w54': (187, 'w54'), 'w6': (188, 'w6'), 'w61': (189, 'w61'), 'w64': (190, 'w64'), 'w65': (191, 'w65'), 'w67': (192, 'w67'), 'w7': (193, 'w7'), 'w9': (194, 'w9'), 'pax': (195, 'pax'), 'pd': (196, 'pd'), 'pe': (197, 'pe'), 'phx': (198, 'phx'), 'plx': (199, 'plx'), 'pmx': (200, 'pmx'), 'pnl': (201, 'pnl'), 'prx': (202, 'prx'), 'pwx': (203, 'pwx'), 'w11': (204, 'w11'), 'w14': (205, 'w14'), 'w15': (206, 'w15'), 'w17': (207, 'w17'), 'w19': (208, 'w19'), 'w2': (209, 'w2'), 'w23': (210, 'w23'), 'w25': (211, 'w25'), 'w26': (212, 'w26'), 'w27': (213, 'w27'), 'pl0': (214, 'pl0'), 'pl4': (215, 'pl4'), 'pl3': (216, 'pl3'), 'pm2.5': (217, 'pm2.5'), 'ph4.4': (218, 'ph4.4'), 'pn40': (219, 'pn40'), 'ph3.3': (220, 'ph3.3'), 'ph2.6': (221, 'ph2.6'), 'None': 0}



def get_split(split_name, dataset_dir, file_pattern, reader,
              split_to_sizes, items_to_descriptions, num_classes):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.

    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader
    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        # 'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        # 'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        # 'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        # 'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)
    # else:
    #     labels_to_names = create_readable_names_for_imagenet_labels()
    #     dataset_utils.write_label_file(labels_to_names, dataset_dir)

    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=split_to_sizes[split_name],
            items_to_descriptions=items_to_descriptions,
            num_classes=num_classes,
            labels_to_names=labels_to_names)
