"""
6.1010 Spring '23 Lab 2: Image Processing 2
"""

#!/usr/bin/env python3

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image

# lab 1 functions


def get_pixel(image, row, col, boundary_behavior):
    """
    Gets the pixel value in the image given row and col with specified boundary behavior

    Args:
    image: a dict that represents an image with height, width and pixels
    row: row of the pixel
    col: column of the pixel
    boundary_behavior: "zero", "wrap" or "extend"
    a string indicating what to return when row and/or col of pixel are out of boundary

    """
    im_height = image["height"]
    im_width = image["width"]
    # out of bounds
    if (row not in range(im_height)) or (col not in range(im_width)):
        if boundary_behavior == "zero":
            return 0
        if boundary_behavior == "extend":
            if row >= im_height:
                row = im_height - 1
            else:
                row = max(row, 0)

            if col >= im_width:
                col = im_width - 1
            else:
                col = max(col, 0)

        if boundary_behavior == "wrap":
            row = (row) % im_height
            col = (col) % im_width

    return image["pixels"][(row * image["width"]) + col]


def set_pixel(image, row, col, color):
    """set the pixel at (row, col) to the given color value"""
    image["pixels"][(row * image["width"]) + col] = color


def apply_per_pixel(image, func):
    """
    Applies the effec of func to every pixel in the given image

    Args:
    func: a function to change the value of pixel

    Returns:
    result: a new dict with changed pixels

    """
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [],
    }

    for color in image["pixels"]:
        result["pixels"].append(func(color))
    return result


def inverted(image):
    """
    Reflects pixels about the middle gray value in the given image
    (0 black becomes 255 white and vice versa)
    """
    return apply_per_pixel(image, lambda color: 255 - color)


# HELPER FUNCTIONS


def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    DESCRIBE YOUR KERNEL REPRESENTATION HERE

    kernel: a dictionary with size and values as two key/value pairs;
        size: int representing width/height of the kernel
        values: a list of all pixel values in the kernel
    A kenerl has to be square and have odd number of cols and rows;

    Example:

    3*3 kernel representation:
    kernel: {
    "size" = 3,
    "values" = [0,0,0,0,1,0,0,0,0],
    }


    """
    boundary_key = ["zero", "extend", "wrap"]
    if boundary_behavior not in boundary_key:
        return None

    def get_new_value(
        image, row, col, kernel, boundary_behavior
    ):  # function that returns the correlated value for a single pixel
        size = kernel["size"]
        result = 0
        count = (
            0  # makes it easier to keep track of and iterate throughg each kernel value
        )

        for i in range(row - (size - 1) // 2, row + (size - 1) // 2 + 1):
            for j in range(col - (size - 1) // 2, col + (size - 1) // 2 + 1):
                old_value = get_pixel(image, i, j, boundary_behavior)
                new_value = old_value * kernel["values"][count]
                result += new_value
                count += 1

        return result

    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": list(image["pixels"]),
    }

    for row in range(image["height"]):
        for col in range(image["width"]):
            new_color = get_new_value(image, row, col, kernel, boundary_behavior)
            set_pixel(result, row, col, new_color)

    return result


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """

    for i in range(len(image["pixels"])):
        image["pixels"][i] = round(image["pixels"][i])
        if image["pixels"][i] > 255:
            image["pixels"][i] = 255
        if image["pixels"][i] < 0:
            image["pixels"][i] = 0


# FILTERS

# HELPER FUNCTION THAT TAKES A SINGLE ARGUMENT AND RETURNS AN N-BY-N BOX BLUR KERNEL


def blur_kernel(n):
    """Takes a single argument n and returns an n-by-n box blur kernel."""
    value = 1 / n / n
    values = []
    values = [value] * (n * n)
    k = {
        "size": n,
        "values": values,
    }
    return k


def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)
    kernel = blur_kernel(kernel_size)
    # then compute the correlation of the input image with that kernel
    result = correlate(image, kernel, "extend")
    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.
    round_and_clip_image(result)
    return result


def sharpened(image, n):
    """
    Subtracts an "unsharp" version of the image
    from a scaled version of the original image.

    The sharpened value at each location is 2 times original value minus blurred value

    Args:
    image: a dict that represents an image with height, width and pixels
    n: int of the size of the blur kernel
    Returns:
    a new image with sharpened effect
    """
    values = []
    value = -1 / n / n
    values = [value] * (n * n)
    values[round((n * n - 1) / 2)] = 2 + value
    k = {
        "size": n,
        "values": values,
    }
    new = correlate(image, k, "extend")
    round_and_clip_image(new)
    return new


def edges(image):
    """
    Return a new image representing the result applying the Sobel Operator filter
    to detec edges to the given input image.
    The edge detector involveds two specifc kernels: Krow and Kcol;
    Each pixel of the output is the square root of the sum of
    squares of corresponding pixels with Krow and Kcol.

    """
    kernel_row = {
        "size": 3,
        "values": [-1, -2, -1, 0, 0, 0, 1, 2, 1],
    }

    kernel_col = {
        "size": 3,
        "values": [-1, 0, 1, -2, 0, 2, -1, 0, 1],
    }
    new_krow_pixels = correlate(image, kernel_row, "extend")["pixels"]
    new_kcol_pixels = correlate(image, kernel_col, "extend")["pixels"]

    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [],
    }

    for i, j in zip(new_krow_pixels, new_kcol_pixels):
        new_pixel = round(math.sqrt(i**2 + j**2))
        result["pixels"].append(new_pixel)

    round_and_clip_image(result)
    return result


# VARIOUS FILTERS


# helper function
def split_color_to_three(color_im):
    """
    Given a color image, split the given image into its three (r, g, b)components
    so that it returns three new images
    """
    pixels = color_im["pixels"]
    red_pixel = list(list(zip(*pixels))[0])
    green_pixel = list(list(zip(*pixels))[1])
    blue_pixel = list(list(zip(*pixels))[2])
    red_im = {
        "height": color_im["height"],
        "width": color_im["width"],
        "pixels": red_pixel,
    }
    green_im = {
        "height": color_im["height"],
        "width": color_im["width"],
        "pixels": green_pixel,
    }
    blue_im = {
        "height": color_im["height"],
        "width": color_im["width"],
        "pixels": blue_pixel,
    }

    return red_im, green_im, blue_im


def combine_three_to_color(red_im, green_im, blue_im):
    """
    Given three images with only one of (r, g, b) values,
    combines them as components of a color image
    returns a new color image
    """
    red_pixels = red_im["pixels"]
    green_pixels = green_im["pixels"]
    blue_pixels = blue_im["pixels"]
    pixels = list(zip(red_pixels, green_pixels, blue_pixels))
    color_im = {
        "height": red_im["height"],
        "width": red_im["width"],
        "pixels": pixels,
    }
    return color_im


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """

    def new_color_image(im):
        red, green, blue = split_color_to_three(im)
        new_r = filt(red)
        new_g = filt(green)
        new_b = filt(blue)
        result = combine_three_to_color(new_r, new_g, new_b)
        return result

    return new_color_image


def make_blur_filter(kernel_size):
    """
    returns a blur filter (which takes a single image as argument)
    """

    def one_arg_blur_filter(im):
        result = blurred(im, kernel_size)
        return result

    return one_arg_blur_filter


def make_sharpen_filter(kernel_size):
    def one_arg_sharpen_filter(im):
        result = sharpened(im, kernel_size)
        return result

    return one_arg_sharpen_filter


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """

    def combined_filter(im):
        result = im
        for filt in filters:
            result = filt(result)
        return result

    return combined_filter


# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": image["pixels"][:],
    }

    filt = filter_cascade(
        [
            greyscale_image_from_color_image,
            compute_energy,
            cumulative_energy_map,
            minimum_energy_seam,
        ]
    )

    for i in range(ncols):
        seam = filt(result)
        seam_twocats = image_without_seam(result, seam)
        result = seam_twocats

    return result


# Optional Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """

    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [],
    }
    color_pixels = image["pixels"]
    red_pixel = list(list(zip(*color_pixels))[0])
    green_pixel = list(list(zip(*color_pixels))[1])
    blue_pixel = list(list(zip(*color_pixels))[2])
    for i, j, k in zip(red_pixel, green_pixel, blue_pixel):
        new_pixel = round(0.299 * i + 0.587 * j + 0.114 * k)
        result["pixels"].append(new_pixel)
    return result


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    # energy_length = len(energy["pixels"])

    result = {
        "height": energy["height"],
        "width": energy["width"],
        "pixels": energy["pixels"][:],
    }

    # first_row = energy["pixels"][:energy_length-1]
    # for col in range(energy["width"]):
    #     set_pixel(result, 0, col, first_row[col])
    # # result["pixels"].extend(first_row)
    for row in range(1, energy["height"]):
        for col in range(energy["width"]):
            middle = get_pixel(result, row - 1, col, "zero")
            right = get_pixel(result, row - 1, col + 1, "zero")
            left = get_pixel(result, row - 1, col - 1, "zero")
            if col - 1 == -1:
                left = float("inf")
            if col + 1 == energy["width"]:
                right = float("inf")
            set_pixel(
                result,
                row,
                col,
                min(middle, right, left) + get_pixel(result, row, col, "zero"),
            )

    return result


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    seam = []
    # bottom_list = cem("pixels")[-cem["wiedth"]:]
    # bottom_value = min(bottom_list)
    row = cem["height"] - 1
    min_col = 0
    minimum = float("inf")
    for col in range(cem["width"]):
        new_value = get_pixel(cem, row, col, "zero")
        if new_value < minimum:
            minimum = new_value
            min_col = col

    seam.append(get_pixel_index(cem, row, min_col))

    while row > 0:
        left = get_pixel(cem, row - 1, min_col - 1, "zero")
        right = get_pixel(cem, row - 1, min_col + 1, "zero")
        middle = get_pixel(cem, row - 1, min_col, "zero")
        if min_col - 1 == -1:
            left = float("inf")
        if min_col + 1 == cem["width"]:
            right = float("inf")
        minimum = min(left, middle, right)
        if minimum == right:
            new_col = min_col + 1
        if minimum == middle:
            new_col = min_col
        if minimum == left:
            new_col = min_col - 1
        min_col = new_col
        row -= 1
        seam.append(get_pixel_index(cem, row, min_col))

    return seam


def get_pixel_index(cem, row, col):
    index = (row * cem["width"]) + col
    return index


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    result = {
        "height": image["height"],
        "width": image["width"] - 1,
        "pixels": image["pixels"][:],
    }
    for i in sorted(seam)[
        ::-1
    ]:  # the list is not necessarily in increasing order, so reverse itself wont work
        result["pixels"].pop(i)
    return result


def custom_feature(image, color, c_row, c_col, outer_radius, inner_radius=0):
    """
    Given an image and the color, center position(row and column),
    and the radius (out_radius) of the solid circle to be drawn
    Inner radius is needed if a hollow circle is wanted; defaulted to 0
    return a new image that contains the newly drawn circle
    does not modify the original image

    Args:
    image: a dictionary representing the image to be modified
    c_row: an int representing of the row of the center of the circle
    c_col: an int representig the col of the center of the circle
    color: a tuple of rgb value, ordered as (r,g,b)
    outer_radius: an int of the (outer) radius of the circle to be drawn
    inner_radius: an int of the inner circle radius if to draw a hollow circle;
                  deault value = 0

    Return:
    the new image with a new circle of given color and position
    """
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": image["pixels"][:],
    }

    center = [c_row, c_col]
    for row in range(image["height"]):
        for col in range(image["width"]):
            current = [row, col]
            if inner_radius <= math.dist(current, center) <= outer_radius:
                set_pixel(result, row, col, color)

    return result


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.

    # #test 1-3 and 3-1 functions
    # i = {
    # 'height': 3,
    # 'width': 2,
    # 'pixels': [(255, 0, 0), (39, 143, 230),
    #            (255, 191, 0), (0, 200, 0),
    #            (100, 100, 100), (179, 0, 199)],
    # }

    # print(split_color_to_three(i))

    # r = {'height': 3, 'width': 2, 'pixels': [255, 39, 255, 0, 100, 179]}
    # g = {'height': 3, 'width': 2, 'pixels': [0, 143, 191, 200, 100, 0]}
    # b = {'height': 3, 'width': 2, 'pixels': [0, 230, 0, 0, 100, 199]}
    # print(combine_three_to_color(r,g,b))

    # # grey filter - color filter
    # color_inverted = color_filter_from_greyscale_filter(inverted)
    # inverted_color_cat = color_inverted(load_color_image('test_images/cat.png'))
    # save_color_image(inverted_color_cat, "inverted_cat.png", mode="PNG")

    # # new blur and sharpened
    # blur_filter = make_blur_filter(9)
    # color_blurred = color_filter_from_greyscale_filter(blur_filter)
    # blurred_color_python = color_blurred(load_color_image('test_images/python.png'))
    # save_color_image(blurred_color_python, "blurred_python.png", mode="PNG")

    # sharpen_filter = make_sharpen_filter(7)
    # color_sharpened = color_filter_from_greyscale_filter(sharpen_filter)
    # new_python = color_sharpened(load_color_image('test_images/sparrowchick.png'))
    # save_color_image(new_python, "sharpened_python.png", mode="PNG")

    # # combined function
    # filter1 = color_filter_from_greyscale_filter(edges)
    # filter2 = color_filter_from_greyscale_filter(make_blur_filter(5))
    # filt = filter_cascade([filter1, filter1, filter2, filter1])
    # combined_frog = filt(load_color_image('test_images/frog.png'))
    # save_color_image(combined_frog, "combined_frog.png", mode="PNG")

    # seam_carving
    # im = load_color_image('test_images/twocats.png')
    # seam_twocats = seam_carving(im, 100)
    # save_color_image(seam_twocats, "test1.png", mode="PNG")

    # im = load_color_image("test_images/twocats.png")
    # color1 = (0, 20, 30)
    # new_image = custom_feature(im, color1, 45, 110, 10)
    # color2 = (10, 50, 60)
    # new_image_circle = custom_feature(new_image, color2, 60, 230, 60,50)

    # save_color_image(new_image_circle, "hello.png", mode="PNG")

    pass
