import platform
import cv2
import yaml

from i308_calib.capture import ThreadedCapture

CAPTURE_MODES = {
    'auto',
    'dshow',
    'any'
}

# CAMERA_TYPES = {
#     'mono',
#     'stereo',
# }

COMPRESSION = {
    'XVID': "XviD MPEG-4",
    'MJPG': "Motion JPEG",
    'H264': "H.264 / AVC",
    'DIVX': "DivX MPEG-4",
    'MP4V': "MPEG-4 Video",
    'I420': "Uncompressed YUV format"
}



class SysInfo:
    def __init__(self):
        self.os_platform = platform.platform()
        self.os_system = platform.system()


def check_video(source):
    if isinstance(source, str):
        if str.isdigit(source):
            source = int(source)

    return source


def parse_resolution(resolution):
    if isinstance(resolution, str):
        ret = tuple(map(int, resolution.split("x")))
    else:
        ret = resolution
    return ret


def format_resolution(resolution):
    return f"{resolution[0]}x{resolution[1]}"


def check_resolutions(resolution, resolutions):
    if resolutions:
        resolutions = [parse_resolution(r) for r in resolutions]

    resolution = check_resolution(resolution, resolutions)

    return resolution, resolutions


def check_resolution(resolution, resolutions):
    if resolution:
        resolution = parse_resolution(resolution)

        if resolutions and resolution not in resolutions:
            raise Exception(f"resolution {format_resolution(resolution)} not available")

    return resolution


def check_capture_mode(mode=None):
    if not mode:
        mode = 'auto'
    if mode not in CAPTURE_MODES:
        raise Exception(f"mode {mode} not available")

    return mode


def check_compression(compression=None):

    if not compression:
        return None

    compression = compression.upper()
    if compression not in COMPRESSION:
        raise Exception(f"compression type '{compression}' not valid.")

    return compression


class CaptureConfig:

    def __init__(
            self,
            video,
            resolution=None,
            resolutions=None,
            fps=None,
            capture_mode=None,
            name=None,
            threaded=None,
            compression=None

    ):
        self.video = check_video(video)
        self.resolution, self.resolutions = check_resolutions(resolution, resolutions)
        self.name = name

        self.fps = fps
        self.capture_mode = check_capture_mode(capture_mode)
        self.threaded = threaded
        self.compression = check_compression(compression)

    def __str__(self):
        ret = f"source: {self.video}; "
        if self.resolution is not None:
            ret += f"resolution: {self.resolution};"

        return ret

    def set_video(self, video):
        self.video = check_video(video)

    def set_resolution(self, resolution):
        self.resolution = check_resolution(resolution, self.resolutions)

    def set_resolutions(self, resolution, resolutions):
        self.resolution, self.resolutions = check_resolutions(resolution, resolutions)

    def set_capture_mode(self, capture_mode):
        self.capture_mode = check_capture_mode(capture_mode)

    def set_compression(self, camera_type):
        self.compression = check_compression(camera_type)


def from_yaml(file):
    with open(file, 'r') as f:
        parsed = yaml.safe_load(f)

    video = parsed.get("video")

    ret = CaptureConfig(
        video
    )

    name = parsed.get("name")
    resolution = parsed.get("resolution")
    resolutions = parsed.get("resolutions")
    fps = parsed.get("fps")
    capture_mode = parsed.get("capture_mode")
    threaded = parsed.get("threaded", True)
    compression = parsed.get("compression")

    ret.set_resolutions(resolution, resolutions)
    ret.set_capture_mode(capture_mode)
    ret.set_compression(compression)
    ret.name = name
    ret.fps = fps
    ret.threaded = threaded

    return ret


# def to_yaml(config: CaptureConfig, file=None):
#     """
#     Serializes the CaptureConfig object to a YAML string or file.
#
#     Args:
#         file (str, optional): If provided, the YAML will be written to this file.
#
#     Returns:
#         str: YAML string if `file` is None, otherwise None.
#     """
#     data = {
#         "video": config.video,
#         "resolution": config.resolution,
#         "resolutions": config.resolutions,
#         "fps": config.fps,
#         "capture_mode": config.capture_mode,
#         "name": config.name,
#         "threaded": config.threaded,
#         "camera_type": config.camera_type,
#     }
#
#     if file:
#         with open(file, 'w') as f:
#             yaml.safe_dump(data, f)
#     else:
#         return yaml.safe_dump(data)


def guess_capture_mode(sys_info: SysInfo) -> str:
    if sys_info.os_platform == "Windows":
        mode = "dshow"
    else:
        mode = "any"
    return mode


def new_video_capture(config: CaptureConfig):
    device = config.video
    resolution = config.resolution

    print(f"starting video capture: {config}")

    mode = config.capture_mode
    if mode == 'auto':
        info = SysInfo()
        mode = guess_capture_mode(info)
    print(f"capture engine: {mode}")
    if mode == 'dshow':
        mode = cv2.CAP_DSHOW
    else:
        mode = cv2.CAP_ANY

    cap = cv2.VideoCapture(device, mode)
    if config.compression:
        print(f"compression: {config.compression}")
        four_cc = cv2.VideoWriter_fourcc(*config.compression)
        cap.set(cv2.CAP_PROP_FOURCC, four_cc)

    if config.fps:
        cap.set(cv2.CAP_PROP_FPS, config.fps)

    if config.resolution is not None:
        req_w, req_h = resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)

    if not cap.isOpened():
        raise Exception("Cannot open capture")

    if config.threaded:
        th_cap = ThreadedCapture(cap)
        th_cap.start()

        cap = th_cap

    return cap


def get_capture_config(
    args
) -> CaptureConfig:

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


def add_capture_args(arg_parser):

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
        "-r", "--resolution",
        default=None,
        help=f"requested resolution. "
    )

    return arg_parser

