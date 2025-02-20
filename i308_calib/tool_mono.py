import argparse
import os
import cv2

import glob
from capture import new_video_capture


from i308_calib.calib_set import CalibSet
from i308_calib.tool_base import parse_checkerboard, add_common_args, get_capture_config
from i308_calib.threaded_capture import ThreadedCapture


def parse_args():
    arg_parser = argparse.ArgumentParser()

    add_common_args(arg_parser)

    arg_parser.add_argument(
        "-d", "--data",
        default="data",
        help="directory where images are going to be stored/retrieved"
    )

    args = arg_parser.parse_args()

    if not args.video and not args.config:
        arg_parser.error("Either -v (video) or -cfg (config) must be specified, or both.")

    # parse checkerboard
    args.checkerboard = parse_checkerboard(args.checkerboard)

    return args


def save_capture(
        args,
        image,
        number,
        dir=""
):
    file_name = os.path.join(
        args.data,
        dir,
        f"image_{number}.jpg"
    )
    print(f"saving {file_name}")
    cv2.imwrite(
        file_name, image
    )


def detect_checkerboard(args, image):
    checkerboard = args.checkerboard

    # 1. convert image to grayscale
    shape = image.shape
    if len(shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found, corners = calib.detect_board(
        checkerboard,
        gray
    )

    detection = {
        "found": found,
        "corners": corners,
        "image": image.copy()
    }

    return detection


def add_detection(
        args,
        object_points,
        calib_set,
        detection,
        save=True
):
    if detection is None:
        print("please enable detection using 'd' key")

    elif not detection['found']:

        print("board was not found, cannot add image to set")

    else:

        image = detection['image']
        image_points = detection["corners"]

        image_no = calib_set.add(
            image,
            object_points,
            image_points
        )

        if save:
            save_capture(
                args,
                image,
                image_no,
                dir="calib",
            )


def calibrate(args, cs: CalibSet):
    object_points = cs.object_points
    image_points = cs.image_points
    img_shape = cs.image_shape

    print("num_points", len(object_points))
    print("calibrating...")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        img_shape, None, None
    )

    print("# Camera matrix : \n")
    print("K = ", calib.np_print(K))
    print()

    print("# Distortion Coefficients : \n")
    print("dist = ", calib.np_print(dist))
    print()


def load_calib_set(
        args
):
    checkerboard = args.checkerboard
    checkerboard_world_points = args.square_size * calib.board_points(checkerboard)

    calib_set = CalibSet()

    directory = os.path.join(args.data, "calib")
    files_pattern = "*.jpg"
    find_files = os.path.join(directory, files_pattern)

    def numeric_sort(file_name):
        return int(file_name.split("_")[-1].split(".")[0])

    file_names = sorted(
        glob.glob(find_files),
        key=numeric_sort
    )

    image_shape = None

    for file_name in file_names:

        print("processing", file_name)

        # read image
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        # get the images sizes
        shape = (image.shape[1], image.shape[0])

        # checks that images sizes match
        if image_shape is None:
            # remembers the images size
            image_shape = shape
        else:
            if image_shape != shape:
                raise Exception(f"there are images with different sizes: {image_shape} vs {shape}")

        # finds the checkerboard in the image
        detection = detect_checkerboard(args, image)

        if not detection["found"]:
            print("warning, checkerboard was not found")
            continue

        # checkerboard was found in both images.

        add_detection(
            args,
            checkerboard_world_points,
            calib_set,
            detection,
            save=False
        )

    return calib_set


def make_dirs(args):
    data_dir = args.data
    calib_dir = os.path.join(data_dir, "calib")
    captures_dir = os.path.join(data_dir, "captures")

    for d in [data_dir, captures_dir, calib_dir]:
        if not os.path.exists(d):
            print(f"Directory {d} doesn't exist, creating...")
            os.makedirs(d)


def start(args):
    make_dirs(args)

    # gets capture configuration
    cfg = get_capture_config(args)

    cap = new_video_capture(cfg)

    th_cap = ThreadedCapture(cap)
    th_cap.start()

    print("Checkerboard: ", args.checkerboard)
    print("Square Size: ", args.square_size)

    object_points = calib.board_points(args.checkerboard)
    detection_enabled = False
    detection = None

    calib_set = CalibSet()

    draw_corners = True
    capture = 0

    while True:
        # Capture frame-by-frame
        ret, frame = th_cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        show_img = frame.copy()

        # draws so-far detected corners
        if draw_corners:

            for corners in calib_set.image_points:
                show_img = cv2.drawChessboardCorners(
                    show_img, args.checkerboard, corners, True
                )

        if detection_enabled:

            # detects board
            detection = detect_checkerboard(args, frame)

            found = detection['found']
            if found:
                show_img = calib.draw_checkerboard(
                    show_img,
                    args.checkerboard,
                    detection['corners'],
                    found,
                )

        cv2.imshow('frame', show_img)

        k = cv2.waitKey(1)
        if k == ord('q'):
            # quit
            break

        if k == ord('h'):

            keys_help = """

                h: help, 
                    shows help.

                q: quit, 
                    terminates the calibration app.

                d: detect, 
                    enables or disables checkerboard detection.

                a: add, 
                    adds a new checkerboard detection to the calibration set.
                    detection must be enabled

                c: calibrate, 
                    performs the camera calibration,.
                    the calibration set must have at least 10 detected checkerboards.
                    calibration results are stored into a pickle file: calibration.pkl

                m: rectification maps, 
                    Crea los mapas de rectification tanto de corrección de distorsión para cada lente,
                    cómo los mapas de rectificación estéreo.
                    Guarda los mapas de rectificación en el archivo pickle stereo_calibration.pkl

                s: snapshot, 
                    captures the current frame and stores it into the captures directory

                l: load calibration images,
                    reads images from the calibration directory, 
                    and reprocesses the images to detect the checkerboards


            """
            print(keys_help)

        elif k == ord('s'):
            print("saving capture...")
            save_capture(
                args,
                frame,
                capture,
                "captures"
            )
            capture += 1

        elif k == ord('d'):

            # toggles detection on / off
            detection_enabled = not detection_enabled
            if not detection_enabled:
                detection = None

        elif k == ord('a'):

            # add image to calibration set

            add_detection(
                args,
                object_points,
                calib_set,
                detection
            )

        elif k == ord('c'):

            if len(calib_set.image_points) < 10:
                print("not enough images to calibrate")
            else:
                # calibrate
                calibrate(args, calib_set)


        elif k == ord('l'):

            print(f"loading calibration images:")
            calib_set = load_calib_set(args)

    # When everything done, release the capture
    th_cap.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()

    start(args)
