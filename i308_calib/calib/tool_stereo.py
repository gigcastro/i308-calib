import argparse
import os
import pickle

import cv2

import glob

import numpy as np

from i308_calib.calib import calib_utils


from i308_calib.calib.calib_utils import np_print, board_points, draw_checkerboard
from i308_calib.calib.dataset import StereoDataset
from i308_calib.calib.tool_base import add_common_args, parse_checkerboard, detect_checkerboard

from i308_calib import capture
from i308_calib.capture import get_capture_config, new_video_capture, ThreadedCapture


def parse_args():
    epilog = """

           Examples:

               # calibrate camera 0 with default parameters
               calib-stereo --video 0

               # calibrate the camera /dev/video3 specifying some parameters
               calib-stereo --video /dev/video3 --resolution 2560x720 --checkerboard 9x6 --square-size 32.0 --data data/stereo

               # using a capture configuration file
               calib --config cfg/stereo.yaml 

       """

    arg_parser = argparse.ArgumentParser(
        description="Stereo Camera calibration tool.",
        epilog=epilog,
        formatter_class=argparse.RawTextHelpFormatter
    )

    capture.add_capture_args(arg_parser)
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
        cam_name,
        dir=""
):
    file_name = os.path.join(
        args.data,
        dir,
        f"{cam_name}_{number}.jpg"
    )
    print(f"saving {file_name}")
    cv2.imwrite(
        file_name, image
    )


def add_detection(
        args,
        object_points,
        dataset: StereoDataset,
        detection_left,
        detection_right,
        save=True
):
    if detection_left is None or detection_right is None:
        print("please enable detection using 'd' key")

    elif not detection_left['found'] or not detection_right['found']:

        print("board was not found, cannot add image to dataset")

    else:

        left_image = detection_left["image"]
        left_corners = detection_left["corners"]

        left = (
            left_image,
            object_points,
            left_corners,
        )

        right_image = detection_right["image"]
        right_corners = detection_left["corners"]

        right = (
            right_image,
            object_points,
            right_corners,
        )

        image_no = dataset.add(
            left,
            right,
        )

        if save:
            save_capture(args, left_image, image_no, cam_name="left", dir="calib")
            save_capture(args, right_image, image_no, cam_name="right", dir="calib")


# def calibrate(calibration):
#     obj_points = calibration["object_points"]
#     world_points = calibration["image_points"]
#     img_shape = calibration["image_shape"]
#
#     print("num_points", len(obj_points))
#     print("calibrating...")
#
#     ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
#         obj_points,
#         world_points,
#         img_shape, None, None
#     )
#
#     # np.set_printoptions(suppress=True)
#     # print("Camera matrix : \n")
#     # cam_matrix = mtx.round(3)
#     # print([list(i) for i in cam_matrix])
#     ## print(mtx.round(3))
#     # print("dist : \n")
#     # print(dist)
#     print("# Intrinsic Camera Parameters")
#     print("cam_matrix = " + np_print(K))
#
#     print("# Distortion Coefficients")
#     print("dist_coeffs = " + np_print(dist))
#
#     return K, dist, rvecs, tvecs


def calibrate_stereo(args, calibration_info, left_calibration, right_calibration):
    left_info = calibration_info["left"]
    right_info = calibration_info["right"]

    left_images_points = left_info["image_points"]
    right_images_points = right_info["image_points"]
    world_points = left_info["object_points"]
    image_size = left_info["image_shape"]

    world_points = [p.reshape(-1, 3) for p in world_points]
    left_images_points = [p.reshape(-1, 2) for p in left_images_points]
    right_images_points = [p.reshape(-1, 2) for p in right_images_points]

    if left_calibration:
        left_K, left_dist = left_calibration[0], left_calibration[1]
    else:
        left_K, left_dist = None, None

    if right_calibration:
        right_K, right_dist = right_calibration[0], right_calibration[1]
    else:
        right_K, right_dist = None, None

    err, left_K, left_dist, right_K, right_dist, R, T, E, F = cv2.stereoCalibrate(
        world_points,
        left_images_points,
        right_images_points,
        left_K,
        left_dist,
        right_K,
        right_dist,
        image_size,
        flags=0
    )

    to_print = [

        "# Left camera Intrinsics:",
        ("left_K", left_K),
        ("left_dist", left_dist),

        "# Right camera Intrinsics:",
        ("right_K", right_K),
        ("right_dist", right_dist),

        "# Rotation:",
        ("R", R),

        "# Translation:",
        ("T", T),

        "# Essential Matrix:",
        ("E", E),

        "# Fundamental Matrix:",
        ("F", F),

    ]
    print("# STEREO CALIBRATION")
    for line in to_print:

        if isinstance(line, str):
            print(line)
        else:
            var_name, np_array = line
            print(f"{var_name} = {np_print(np_array)}\n")

    calibration_results = {
        'left_K': left_K,
        'left_dist': left_dist,
        'right_K': right_K,
        'right_dist': right_dist,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'image_size': image_size,
        # 'left_pts': left_pts,
        # 'right_pts': right_pts
    }

    calibration_file = os.path.join(args.data, "stereo_calibration.pkl")
    with open(calibration_file, "wb") as f:
        f.write(pickle.dumps(calibration_results))

    return calibration_results


def create_stereo_rectifying_maps(calibration_results):
    left_K = calibration_results['left_K']
    left_dist = calibration_results['left_dist']
    right_K = calibration_results['right_K']
    right_dist = calibration_results['right_dist']
    image_size = calibration_results['image_size']
    R = calibration_results['R']
    T = calibration_results['T']

    print("rectifying stereo...")
    R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(
        left_K, left_dist, right_K, right_dist, image_size, R, T, alpha=0
    )

    print("creating undistortion maps...")
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(left_K, left_dist, R1, P1, image_size, cv2.CV_32FC1)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(right_K, right_dist, R2, P2, image_size, cv2.CV_32FC1)

    stereo_maps = {

        # undistorting maps
        "left_map_x": left_map_x,
        "left_map_y": left_map_y,
        "right_map_x": right_map_x,
        "right_map_y": right_map_y,

        # add also rectifying info:
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "validRoi1": validRoi1,
        "validRoi2": validRoi2,

    }

    stereo_maps_file = os.path.join(args.data, "stereo_maps.pkl")
    with open(stereo_maps_file, "wb") as f:
        f.write(pickle.dumps(stereo_maps))

    return stereo_maps



def load_calib_set(
    args
):
    checkerboard = args.checkerboard
    object_points = args.square_size * calib_utils.board_points(checkerboard)

    dataset = StereoDataset()

    directory = os.path.join(args.data, "calib")
    left_files_pattern = "*left*.jpg"
    right_files_pattern = "*right*.jpg"
    left_find_files = os.path.join(directory, left_files_pattern)
    right_find_files = os.path.join(directory, right_files_pattern)

    def numeric_sort(file_name):
        return int(file_name.split("_")[-1].split(".")[0])

    left_file_names = sorted(glob.glob(left_find_files), key=numeric_sort)
    right_file_names = sorted(glob.glob(right_find_files), key=numeric_sort)

    for left_file_name, right_file_name in zip(left_file_names, right_file_names):

        print("processing", left_file_name, right_file_name)

        # read images
        left_image = cv2.imread(left_file_name, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_file_name, cv2.IMREAD_GRAYSCALE)

        # finds the checkerboard in both images
        left_detection = detect_checkerboard(args, left_image)
        if not left_detection["found"]:
            print("warning, left checkerboard was not found")
            continue
        right_detection = detect_checkerboard(args, right_image)
        if not right_detection["found"]:
            print("warning, right checkerboard was not found")
            continue

        # checkerboard was found in both images.
        add_detection(
            args,
            object_points,
            dataset,
            left_detection,
            right_detection,
            save=False
        )

    return dataset


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

    checkerboard = args.checkerboard
    checkerboard_world_points = args.square_size * board_points(checkerboard)
    detection_enabled = False
    detection_left = None
    detection_right = None

    dataset = StereoDataset()

    draw_corners = True
    capture_no = 0  # 22
    frame_no = 0

    calibration_results = None

    # window_flags = cv2.WINDOW_NORMAL
    # window_flags = cv2.WINDOW_FREERATIO
    # cv2.namedWindow("stereo", window_flags)

    while True:

        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_no += 1

        # if frame_no % 2 == 0:
        #    continue

        # split frame
        shape = frame.shape
        w, h = shape[1], shape[0]

        if frame_no == 1:
            print("resolution: ", (frame.shape[1], frame.shape[0]))

            if cfg.resolution:
                print("requested: ", cfg.resolution)

                is_stereo = cfg.resolution == (w, h)
                if not is_stereo:
                    message = f"specified camara ({cfg.video}) IS NOT the stereo camera"
                    print("terminating.", message)
                    break

        left_frame = frame[:, :int(w / 2), :]
        right_frame = frame[:, int(w / 2):, :]

        # left_frame = cv2.rotate(left_frame, cv2.ROTATE_180)
        # right_frame = cv2.rotate(right_frame, cv2.ROTATE_180)

        show_img_left = left_frame.copy()
        show_img_right = right_frame.copy()

        # draws so-far detected corners
        if draw_corners:

            for corners in dataset.left.image_points:
                show_img_left = cv2.drawChessboardCorners(
                    show_img_left, args.checkerboard, corners, True
                )

            for corners in dataset.right.image_points:
                show_img_right = cv2.drawChessboardCorners(
                    show_img_right, args.checkerboard, corners, True
                )

        if detection_enabled:

            # detects board
            detection_left = detect_checkerboard(args, left_frame)
            detection_right = detect_checkerboard(args, right_frame)

            found = detection_left['found']
            if found:
                show_img_left = calib_utils.draw_checkerboard(
                    show_img_left,
                    args.checkerboard,
                    detection_left['corners'],
                    found
                )

            found = detection_right['found']
            if found:
                show_img_right = calib_utils.draw_checkerboard(
                    show_img_right,
                    args.checkerboard,
                    detection_right['corners'],
                    found,
                )

        # cv2.imshow('left', show_img_left)
        # cv2.imshow('right', show_img_right)
        show_img = np.hstack((show_img_left, show_img_right))
        show_img = cv2.resize(show_img, (int(w / 2), int(h / 2)))
        cv2.imshow("stereo", show_img)

        k = cv2.waitKey(10)

        if k == ord('h'):

            keys_help = """

                h: help, 
                    Muestra ayuda.

                q: quit, 
                    Termina la app

                d: detect, 
                    Habilita o inhabilita la detección del checkerboard.

                a: add, 
                    Agrega detecciones del checkerboard al set de calibración.
                    Debe estar la detección habilitada.

                c: calibrate, 
                    Realiza la calibración estéreo.
                    Deben haberse detectado al menos 10 pares estéreo de checkerboards.
                    Guarda los resultados de calibración en un archivo pickle stereo_calibration.pkl

                m: rectification maps, 
                    Crea los mapas de rectification tanto de corrección de distorsión para cada lente,
                    cómo los mapas de rectificación estéreo.
                    Guarda los mapas de rectificación en el archivo pickle stereo_calibration.pkl

                s: snapshot, 
                    Captura par estéreo de imágenes (left, right)
                    y las guarda en el directorio de capturas.

                l: load calibration images,
                    Lee las imágenes del directorio de calibración, 
                    y reprocesa todas detecciones de checkerboards.


            """
            print(keys_help)

        elif k == ord('q'):
            # quit
            break

        elif k == ord('d'):

            # toggles detection on / off
            detection_enabled = not detection_enabled
            if not detection_enabled:
                detection_left = None
                detection_right = None

        elif k == ord('a'):

            # add image to calibration set

            add_detection(
                args,
                checkerboard_world_points,
                dataset,
                detection_left,
                detection_right
            )

        elif k == ord('c'):

            if len(dataset["left"]["detections"]) < 10:
                print("not enough images to calibrate")
            else:

                # calibrate
                # print("LEFT CALIBRATION:")
                # calib_left = calibrate(dataset["left"])

                # print("RIGHT CALIBRATION:")
                # calib_right = calibrate(dataset["right"])
                calib_left = None
                calib_right = None

                print("STEREO CALIBRATION:")
                calibration_results = calibrate_stereo(args, dataset, calib_left, calib_right)

        elif k == ord('m'):

            if calibration_results is None:
                print("can't create rectification maps, first calibrate stereo.")
            else:
                print(f"creating rectification maps...")
                create_stereo_rectifying_maps(calibration_results)

        elif k == ord('s'):

            print(f"saving snapshot: {capture_no}")
            save_capture(args, left_frame, capture_no, "left", "captures")
            save_capture(args, right_frame, capture_no, "right", "captures")
            capture_no += 1

        elif k == ord('l'):

            print(f"loading calibration images:")
            dataset = load_calib_set(args)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()

    start(args)
