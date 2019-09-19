import keras_ssd7_person as ssd_face
import configparser
import tensorflow as tf
import keras.backend as K
sess = tf.Session()
K.set_session(sess)
import numpy as np
def read_ssd_config(cfg):
    if cfg is None:
        raise ('no config file found')
        return
    if isinstance(cfg, str):
        config = configparser.ConfigParser()
        config.read(cfg)
        cfg = config
    img_height = cfg.getint('ssd', 'img_height')  # Height of the input images
    img_width = cfg.getint('ssd', 'img_width')  # Width of the input images
    img_channels = cfg.getint('ssd', 'img_channels')  # Number of color channels of the input images
    intensity_mean = cfg.getfloat('ssd',
                                  'intensity_mean')  # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
    intensity_range = cfg.getfloat('ssd',
                                   'intensity_range')  # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
    n_classes = cfg.getint('ssd', 'n_classes')  # Number of positive classes
    scales = list(map(float, cfg.get('ssd', 'scales').split(
        ',')))  # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
    # aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
    aspect_ratios = list(
        map(float, cfg.get('ssd', 'aspect_ratios').split(',')))  # The list of aspect ratios for the anchor boxes
    two_boxes_for_ar1 = cfg.getboolean('ssd',
                                       'two_boxes_for_ar1')  # Whether or not you want to generate two anchor boxes for aspect ratio 1
    steps = None  # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
    offsets = None  # In case you'd like to set the offsets for the anchor box grids manually; not recommended
    clip_boxes = cfg.getboolean('ssd',
                                'clip_boxes')  # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
    variances = list(map(float, cfg.get('ssd', 'variances').split(
        ',')))  # The list of variances by which the encoded target coordinates are scaled
    normalize_coords = cfg.getboolean('ssd',
                                      'normalize_coords')  # Whether or not the model is supposed to use coordinates relative to the image size
    return {'image_size':(img_height, img_width, img_channels),
            'n_classes':n_classes,
            # 'mode': 'anchor',
            'mode': 'hardware',
            'l2_regularization':0.0005,
            'scales':scales,
            'aspect_ratios_global':aspect_ratios,
            'aspect_ratios_per_layer':None,
            'two_boxes_for_ar1':two_boxes_for_ar1,
            'steps':steps,
            'offsets':offsets,
            'clip_boxes':clip_boxes,
            'variances':variances,
            'normalize_coords':normalize_coords,
            'subtract_mean':None,
            'divide_by_stddev':None}


option = read_ssd_config('/home/bodong/playground/keras_kneron_v2/deploy/ssd_person/person.cfg')

model = ssd_face.build_model(**option)
model.summary()
model.load_weights('/home/bodong/playground/keras_kneron_v2/deploy/ssd_person/ssd200_pascal_07+12_epoch-194_loss-1.3749_val_loss-1.2931.h5')

if option['mode'] == 'anchor':
    data = model.predict(np.zeros((1,200,200,3)))
    np.save('./anchor_person_ssd10.npy', {'0':data[0], '1':data[1], '2':data[2], '3':data[3]})
elif option['mode'] == 'hardware':
    model.save('./ssd_person_hw.hdf5')
