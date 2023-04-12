from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp2d

def set_3x3(tl_diag, br_diag):
    matrix = np.eye(3)
    matrix[0, 0] = tl_diag
    matrix[2, 2] = br_diag
    return matrix

def white_balance(tl_diag, br_diag, transformed):
    matrix = set_3x3(tl_diag, br_diag)
    white_balanced = np.copy(transformed)
    white_balanced[::2, ::2] *= matrix[0, 0]
    white_balanced[1::2, 1::2] *= matrix[2, 2]
    return white_balanced

def demosaic(img):
    # slice images to get quadrants
    red_img = np.zeros(img.shape)
    red_img[::2, ::2] = img[::2, ::2]
    green_img = np.zeros(img.shape)
    green_img[::2, 1::2] = img[::2, 1::2]
    green_img[1::2, ::2] = img[1::2, ::2]
    blue_img = np.zeros(img.shape)
    blue_img[1::2, 1::2] = img[1::2, 1::2]

    # interpolate on each quadrant
    widths = np.arange(WIDTH)
    heights = np.arange(HEIGHT)

    interp_red = interp2d(widths, heights, red_img, kind='linear')
    interp_green = interp2d(widths, heights, green_img, kind='linear')
    interp_blue = interp2d(widths, heights, blue_img, kind='linear')

    # interpolate
    red_interp = interp_red(widths, heights)
    green_interp = interp_green(widths, heights)
    blue_interp = interp_blue(widths, heights)

    # Create the output image
    output_image = np.zeros((transformed.shape[0], transformed.shape[1], 3))

    # Assign the color channels to the output image
    output_image[::, ::, 0] = red_interp[::, ::]
    output_image[::, ::, 1] = green_interp[::, ::]
    output_image[::, ::, 2] = blue_interp[::, ::]
    return output_image

# python initialization
im1 = io.imread('data/Thayer.tiff')

print('image 1 is', im1.shape)
# convert bytes to bits
bits_per_pixel = im1.dtype.itemsize * 8
width, height = im1.shape[:2]

# Print the results
print(f"Bits per pixel: {bits_per_pixel}")
print(f"Width: {width}")
print(f"Height: {height}")
float_img = im1.astype(np.float64)

# linearization
BLACK = 2044
WHITE = 16383
# make black 0 by subtracting all values by black
transformed = (float_img - BLACK) / (WHITE - BLACK)
transformed = np.clip(transformed, 0, 1)

# Bayer Pattern
print(float_img[:2, :2])

# white balancing
red_img = transformed[::2, ::2]
green1 = transformed[::2, 1::2]
green2 = transformed[1::2, ::2]
blue_img = transformed[1::2, 1::2]

red_avg = np.average(red_img)
red_max = np.max(red_img)

green_avg = (np.average(green1) + np.average(green2))/2
green_max = max(np.max(green1), np.max(green2))

blue_avg = np.average(blue_img)
blue_max = np.max(blue_img)

grey_world_img = white_balance(green_avg / blue_avg, green_avg / red_avg, transformed)
white_world_img = white_balance(green_max / blue_max, green_max / red_max, transformed)
# r_scale, b_scale (g_scale is 1)
third_wb_img = white_balance(2.165039, 1.643555, transformed)

# save images
im = Image.fromarray((grey_world_img*255).astype('uint8'))
im.save("grey_world1.jpeg")

im = Image.fromarray((white_world_img*255).astype('uint8'))
im.save("white_world1.jpeg")

im = Image.fromarray((third_wb_img*255).astype('uint8'))
im.save("third_wb1.jpeg")

# demosaicing
HEIGHT = transformed.shape[0]
WIDTH = transformed.shape[1]

grey_world_demosaic = demosaic(grey_world_img)
white_world_demosaic = demosaic(white_world_img)
third_wb_demosaic = demosaic(third_wb_img)

print('grey demosaic is', grey_world_demosaic, np.count_nonzero(grey_world_demosaic == np.nan))
im = Image.fromarray((grey_world_demosaic*255).astype('uint8'), 'RGB')
im.save("grey_world2.jpeg")

im = Image.fromarray((white_world_demosaic*255).astype('uint8'), 'RGB')
im.save("white_world2.jpeg")

im = Image.fromarray((third_wb_demosaic*255).astype('uint8'), 'RGB')
im.save("third_wb2.jpeg")

# color correction

M_sRGBtoXYZ = np.array([[0.4124564, 0.3575761, 0.1804375], 
                        [0.2126729, 0.7151522, 0.0721750], 
                        [0.0193339, 0.1191920, 0.9503041]])

M_XYZtoCam = np.array([[24542, -10860, -3401],
                       [-1490, 11370, -297],
                       [2858, -605, 3225]]) / 10000

prod = M_XYZtoCam @ M_sRGBtoXYZ
prod = prod / np.sum(prod)
print('sum is', np.sum(prod))
M_sRGBtoCam_inv = np.linalg.inv((prod))
print("inverted matrix is", M_sRGBtoCam_inv)

# transpose?

# .transpose(0, 2, 1)
image_dup = grey_world_demosaic.copy()
image_dup = np.transpose(image_dup, [2,0,1])
print("size:", image_dup.shape)
image_dup = np.reshape(image_dup, (3, WIDTH*HEIGHT))
print("size:", image_dup.shape)

image_col_tf = M_sRGBtoCam_inv @ image_dup
print("size:", image_col_tf.shape)
image_col_tf = np.reshape(image_col_tf, (3, HEIGHT, WIDTH))
print("size:", image_col_tf.shape)
image_col_tf = np.transpose(image_col_tf, [1, 2, 0])
print("size:", image_col_tf.shape)
print('the image is', image_col_tf)
im = Image.fromarray((image_col_tf*255).astype('uint8'), 'RGB')
im.save("colorcorrected.jpeg")
plt.imshow(image_col_tf)

