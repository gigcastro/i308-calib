from i308_calib.capture import CaptureConfig, from_yaml


def parse_checkerboard(checkerboard):
    return tuple(map(int, checkerboard.split("x")))


def add_common_args(arg_parser):

    arg_parser.add_argument(
        "-cfg", "--config",
        default=None,
        help="capture configuration file (.yaml)"
    )

    arg_parser.add_argument(
        "-v", "--video",
        default=None,
        help="video device to be opened for calibration eg. 0"
    )

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

    arg_parser.add_argument(
        "-r", "--resolution",
        default=None,
        help=f"requested resolution. "
    )


def get_capture_config(
    args
):

    video = args.video
    config = args.config

    if config is not None:
        ret = from_yaml(config)

    else:

        ret = CaptureConfig(video)

    if args.video:
        # Overrides config with args
        ret.set_video(args.video)

    if args.resolution:
        # Overrides config with args
        ret.set_resolution(args.resolution)

    return ret
