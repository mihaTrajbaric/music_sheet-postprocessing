import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np


def get_filenames(path):
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    return filenames


def edge_line_removal(image, edge_size=50):
    blank = np.full(image.shape, np.amax(image))
    blank[edge_size:image.shape[0] - edge_size, edge_size:image.shape[1] - edge_size] = \
        image[edge_size:image.shape[0] - edge_size, edge_size:image.shape[1] - edge_size]
    return blank


def remove_suffix(img_name):
    name = img_name[:img_name.rfind(".")]
    return name


def deskew_image(im, max_skew=10):
    height, width = im.shape

    # inverte image
    im_bw = cv2.bitwise_not(im)

    # Detect lines in this image. Parameters here mostly arrived at by trial and error.
    lines = cv2.HoughLinesP(
        im_bw, 1, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150
    )

    # Collect the angles of these lines (in radians)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(np.arctan2(y2 - y1, x2 - x1))

    # If the majority of our lines are vertical, this is probably a landscape image
    landscape = np.sum([abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 2

    # Filter the angles to remove outliers based on max_skew
    if landscape:
        angles = [
            angle
            for angle in angles
            if np.deg2rad(90 - max_skew) < abs(angle) < np.deg2rad(90 + max_skew)
        ]
    else:
        angles = [angle for angle in angles if abs(angle) < np.deg2rad(max_skew)]

    if len(angles) < 5:
        # Insufficient data to deskew
        return im

    # Average the angles to a degree offset
    angle_deg = np.rad2deg(np.median(angles))

    # If this is landscape image, rotate the entire canvas appropriately
    if landscape:
        if angle_deg < 0:
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            angle_deg += 90
        elif angle_deg > 0:
            im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            angle_deg -= 90

    # Rotate the image by the residual offset
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1)
    im = cv2.warpAffine(im, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return im


def thresholding(img, just_mask=True):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if just_mask:
        return mask
    th = img
    th[mask > 0] = 255
    return th


def align_image(img_name, img, align_mode):

    """
    # opened = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    # kernel = cv2.getStructuringElement(cv2.MORPH_HITMISS, (9, 9))
    kernel = np.array([1, 1, 1,1,1], np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    # kernel = np.array([[0,1,0],[0, 1, 0],[0,1,0]], np.uint8)
    dilated = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    dilated = cv2.bitwise_not(dilated)
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    inversed = cv2.bitwise_not(dilated)

    coords = np.column_stack(np.where(inversed > 0))
    min_val = np.min(coords, axis=0)
    max_val = np.max(coords, axis=0)
    center = np.mean([min_val, max_val], axis=0).tolist()

    rows, cols = img.shape
    offset_x = int(cols / 2) - int(center[1]) if align_mode == 'x' or align_mode == 'full' else 0
    offset_y = int(rows / 2) - int(center[0]) if align_mode == 'y' or align_mode == 'full' else 0

    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    dst = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    cv2.imwrite(os.path.join('music_sheet_production', img_name + "-opening_test" + "." + 'bmp'), dilated)
    # cv2.imwrite(os.path.join('music_sheet_production', img_name + "-warp_test" + "." + 'bmp'), dst)

    return dst


def process_images(in_path, out_path, out_format='bmp', mode='production', edge_size=50, deskew=False, max_skew=10,
                   align_mode='none'):
    """

    :param in_path: path of input images
    :param out_path: path of output processed images
    :param out_format: image format for output files (default bmp)
    :param mode: either for further photoshoping (photoshop) or for final processing (production)
    :param edge_size: size (in px) of border, that is removed
    :param deskew: to deskew or not
    :param max_skew: max angle (in degrees)
    :param align_mode: one of 'none', 'x', 'y' or 'full'
    :return: void
    """
    if os.path.exists(in_path):
        filenames = get_filenames(path=in_path)
    else:
        raise IOError("Path {} does not exist".format(in_path))

    if mode != 'photoshop' and mode != 'production':
        raise IOError('Mode can be either photoshop or production, not {}'.format(mode))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for img_name in filenames:
        print(img_name)
        img = cv2.imread(os.path.join(in_path, img_name), 0)
        # img = cv2.fastNlMeansDenoising(img, h=3)
        img = edge_line_removal(image=img, edge_size=edge_size)
        img = thresholding(img=img, just_mask=mode == 'production')
        if deskew:
            img = deskew_image(img, max_skew=max_skew)
        if align_mode is not 'none':
            img = align_image(img_name=img_name, img=img, align_mode=align_mode)
        back_to_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        name = remove_suffix(img_name)
        cv2.imwrite(os.path.join(out_path, name + "_" + mode + "_24bit" + "." + out_format), back_to_rgb)


if __name__ == '__main__':
    process_images(
        in_path='music_sheet_scanned',
        out_path='music_sheet_production',
        out_format='bmp',
        # mode='production',
        mode='photoshop',
        edge_size=50,
        deskew=True,
        max_skew=10,
        # align_mode='none'
        # align_mode='x'
        # align_mode='y'
        align_mode='full'
    )
