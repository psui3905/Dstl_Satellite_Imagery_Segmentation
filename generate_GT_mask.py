import pandas as pd
import shapely
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import numpy as np
import cv2
import os

def _get_xmax_ymin(image_id):
    xmax, ymin = gs[gs['ImageId'] == image_id].iloc[0, 1:].astype(float)
    return xmax, ymin

def get_scalers(height, width, x_max, y_min):
    """

    :param height:
    :param width:
    :param x_max:
    :param y_min:
    :return: (xscaler, yscaler)
    """
    w_ = width * (width / (width + 1))
    h_ = height * (height / (height + 1))
    return w_ / x_max, h_ / y_min

def polygons2mask_layer(height, width, polygons, image_id):
    """

    :param height:
    :param width:
    :param polygons:
    :return:
    """

    x_max, y_min = _get_xmax_ymin(image_id)
    x_scaler, y_scaler = get_scalers(height, width, x_max, y_min)

    polygons = shapely.affinity.scale(polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
    img_mask = np.zeros((height, width), np.uint8)

    if not polygons:
        return img_mask

    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]

    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

def polygons2mask_channel(height, width, polygons, image_id):
    result = np.zeros((height, width))
    result[:, :] = polygons2mask_layer(height, width, polygons, image_id)
    return result

if __name__=='__main__':

    data_path = '../data'
    output_path = os.path.join(data_path, 'GT_Mask')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))
    gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

    # image ids for the images with groundtruths
    ImageId = ['6010_1_2','6010_4_2','6010_4_4','6040_1_0','6040_1_3','6040_2_2','6040_4_4','6060_2_3','6070_2_3',
               '6090_2_0','6100_1_3','6100_2_2','6100_2_3','6110_1_2','6110_3_1','6110_4_0','6120_2_0','6120_2_2',
               '6140_1_2','6140_3_1','6150_2_3','6160_2_1','6170_0_4','6170_2_4','6170_4_1']

    num_channels = 10

    CLASSES = {
        1: 'Building',
        2: 'Structure',
        3: 'Road',
        4: 'Track',
        5: 'Trees',
        6: 'Crops',
        7: 'Fast_H20',
        8: 'Slow_H20',
        9: 'Truck',
        10: 'Car'}

    # The priority for the labels
    ZORDER = {
        1: 4,
        2: 6,
        3: 5,
        4: 3,
        5: 2,
        6: 1,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
    }


    for image_id in ImageId:

        print("Processing image: {0}".format(image_id))

        # width & height of given image
        img = tiff.imread("../data/three_band/{}.tif".format(image_id))
        height, width = img.shape[1:]

        background = np.ones((height, width)) * 255
        accumulated = np.zeros((height, width))

        for z in range(10, 0, -1):

            # labels
            df_label = pd.read_csv('../data/train_wkt_v4.csv')
            for index, row in df_label.iterrows():
                if row['ImageId'] == image_id and row['ClassType'] == ZORDER[z]:
                    MultiPolygons1 = wkt_loads(row['MultipolygonWKT'])

            # get mask of labels & predicted
            mask = polygons2mask_channel(height, width, MultiPolygons1, image_id)

            mask = mask * 255
            # low priority minus high priority
            mask = mask - accumulated
            mask[mask < 0] = 0

            accumulated = accumulated + mask
    
            cv2.imwrite('../data/GT_Mask/' + image_id + '_' + CLASSES[ZORDER[z]] + '.png', mask)

        background = background - accumulated
        cv2.imwrite('../data/GT_Mask/' + image_id + '_Background.png', background)
