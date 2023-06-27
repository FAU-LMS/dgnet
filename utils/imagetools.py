import math
import numpy as np
import time
from PIL import Image, ImageDraw
import cv2 as cv
from skimage.color import rgb2hsv

def create_mask(rgb_image, all=False):
    funcs = [brush_stroke_mask, edge_mask, random_mask, random_block_mask, border_mask]

    num_funcs = len(funcs)
    if not all:
        num_funcs = np.random.randint(1, len(funcs) + 1)

    funcs_ind = np.arange(len(funcs))
    np.random.shuffle(funcs_ind)
    comb_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]))
    for i in range(num_funcs):
        mask = funcs[funcs_ind[i]](rgb_image)
        comb_mask += mask

    comb_mask[comb_mask > 1] = 1
    return comb_mask

def brush_stroke_mask(rgb_image):
    H = rgb_image.shape[0]
    W = rgb_image.shape[1]
    min_num_vertex = 4
    max_num_vertex = 12
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 40
    average_radius = math.sqrt(H * H + W * W) / 8
    mask = Image.new('L', (W, H), 0)

    for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius // 2),
                0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width // 2,
                          v[1] - width // 2,
                          v[0] + width // 2,
                          v[1] + width // 2),
                         fill=1)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.float32)
    return mask

def edge_mask(rgb_image):
    gray = np.dot(rgb_image[...,:3], [0.2989, 0.5870, 0.1140])
    mag = cv.Canny(np.uint8(gray * 255), threshold1=150, threshold2=220)

    start_y, start_x = np.asarray(np.where(mag > 0))
    angle = np.random.uniform(0, np.pi * 2)
    mask_length = np.random.randint(5, 21)
    end_x = np.cos(angle) * mask_length + start_x
    end_y = np.sin(angle) * mask_length + start_y

    mask = np.zeros_like(gray)
    steps = mask_length * 4
    for i in range(steps):
        alpha = i/steps
        cur_x = ((1 - alpha) * start_x + alpha * end_x).astype(np.int)
        cur_y = ((1 - alpha) * start_y + alpha * end_y).astype(np.int)
        cur_x[cur_x < 0] = 0
        cur_y[cur_y < 0] = 0
        cur_x[cur_x >= gray.shape[1]] = gray.shape[1] - 1
        cur_y[cur_y >= gray.shape[0]] = gray.shape[0] - 1
        mask[cur_y, cur_x] = 1
    return mask

def random_mask(rgb_image):
    mask = np.random.randint(0, 101, size=(rgb_image.shape[0], rgb_image.shape[1]))
    perc = np.random.randint(70, 96)
    mask[mask <= perc] = 0
    mask[mask > perc] = 1
    return mask

def random_block_mask(rgb_image):
    block_size = 16
    num_steps = 2
    combined_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]))
    for i in range(num_steps):
        block_size = block_size//2
        blocks_y = rgb_image.shape[0] // block_size + 1
        blocks_x = rgb_image.shape[1] // block_size + 1
        block_mask = np.random.randint(0, 101, size=(blocks_y, blocks_x))
        perc = np.random.randint(80, 101)
        block_mask[block_mask <= perc] = 0
        block_mask[block_mask > perc] = 1
        pos_y = np.arange(rgb_image.shape[0]) // block_size
        pos_y = np.tile(pos_y[:, np.newaxis], (1, rgb_image.shape[1]))
        pos_x = np.arange(rgb_image.shape[1]) // block_size
        pos_x = np.tile(pos_x[np.newaxis, :], (rgb_image.shape[0], 1))
        mask = block_mask[pos_y, pos_x]
        combined_mask += mask

    combined_mask[combined_mask > 1] = 1
    return combined_mask

def border_mask(rgb_image):
    mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]))
    lower_mask_size = 5
    upper_mask_size = 100
    edges = [0, 1, 2, 3]
    np.random.shuffle(edges)
    num_masks = np.random.randint(1, len(edges) + 1)
    for i in range(num_masks):
        if edges[i] == 0:
            mask_size = np.random.randint(lower_mask_size, upper_mask_size)
            mask[:mask_size, :] = 1
        if edges[i] == 1:
            mask_size = np.random.randint(lower_mask_size, upper_mask_size)
            mask[:, :mask_size] = 1
        if edges[i] == 2:
            mask_size = np.random.randint(lower_mask_size, upper_mask_size)
            mask[(mask.shape[0] - mask_size):, :] = 1
        if edges[i] == 3:
            mask_size = np.random.randint(lower_mask_size, upper_mask_size)
            mask[:, (mask.shape[1] - mask_size):] = 1
    return mask

def rgb_to_random_gray_hsv(rgb_image, number_hue_values=10):
    hue_values = np.random.uniform(0, 1, size=number_hue_values)
    white_value = np.random.uniform(0, 1)
    black_value = np.random.uniform(0, 1)
    hsv_image = cv.cvtColor(rgb_image.astype(np.float32), cv.COLOR_RGB2HSV)
    hue = hsv_image[:, :, 0]
    sat = hsv_image[:, :, 1]
    val = hsv_image[:, :, 2]

    gray_image = np.zeros((hue.shape[0], hue.shape[1]))

    for i in range(number_hue_values):
        current = i / number_hue_values
        next = (i + 1) / number_hue_values
        hue_cond = (hue >= current) & (hue <= next)
        alpha = (hue[hue_cond] - current) * number_hue_values
        gray_image[hue_cond] = alpha * hue_values[(i + 1) % number_hue_values] + (1 - alpha) * hue_values[i]

    gray_image = sat * gray_image + (1 - sat) * white_value
    gray_image = val * gray_image + (1 - val) * black_value
    return post_process_gray(gray_image)

def post_process_gray(gray_image):
    image_range = np.max(gray_image) - np.min(gray_image)
    if image_range == 0:
        image_range = 1
    gray_image = (gray_image - np.min(gray_image)) / image_range
    gray_image *= np.random.uniform(low=0.2, high=1.5)
    # noise
    std = np.random.uniform(0, 0.02)
    gray_image += np.random.randn(*gray_image.shape) * std
    gray_image = np.clip(gray_image, 0, 1)
    return gray_image
