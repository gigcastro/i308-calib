import cv2
from i308_calib.calib import calib_utils

def parse_checkerboard(checkerboard):
    return tuple(map(int, checkerboard.split("x")))


def detect_checkerboard(args, image):
    checkerboard = args.checkerboard

    # 1. convert image to grayscale
    shape = image.shape
    if len(shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # if the image is large we optimize
    w, h = shape[1], shape[0]
    if w < 1080:
        scale = 1.0
    else:
        scale = 0.7

    found, corners = calib_utils.detect_board(
        checkerboard,
        gray,
        scale=scale
    )

    detection = {
        "found": found,
        "corners": corners,
        "image": image.copy()
    }

    return detection


def add_common_args(arg_parser):
    arg_parser.add_argument(
        "-c", "--checkerboard",
        default='10x7',
        help="checkerboard squares layout (default '10x7')"
    )

    arg_parser.add_argument(
        "-sq", "--square-size",
        type=float,
        default=24.2,
        help="checkerboard square size [mm], (default 24.2)"
    )
