import platform
import cv2
import yaml

FPS = 30
CAPTURE_MODES = {
    'auto',
    'dshow',
    'any'
}

class SysInfo:
    def __init__(self):
        self.os_platform = platform.platform()
        self.os_system = platform.system()


def guess_capture_mode(sys_info: SysInfo) -> str:
    mode = cv2.CAP_ANY
    if sys_info.os_system == 'Windows':
        mode = cv2.CAP_DSHOW
    return mode


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
            camera_type=None
    ):
        self.source = check_video(video)
        self.resolution, self.resolutions = check_resolutions(resolution, resolutions)
        self.name = name

        self.fps = fps
        self.capture_mode = check_capture_mode(capture_mode)
        self.threaded = threaded
        self.camera_type = camera_type

    def __str__(self):
        ret = f"source: {self.source}; "
        if self.resolution is not None:
            ret += f"resolution: {self.resolution};"

        return ret

    def set_source(self, source):
        self.source = check_video(source)

    def set_resolution(self, resolution):
        self.resolution = check_resolution(resolution, self.resolutions)

    def set_capture_mode(self, capture_mode):
        self.capture_mode = check_capture_mode(capture_mode)


def from_yaml(file):
    with open(file, 'r') as f:
        parsed = yaml.safe_load(f)

    video = parsed.get("video")

    ret = CaptureConfig(
        video
    )

    ret.name = parsed.get("name")
    ret.resolution = parsed.get("resolution")
    ret.resolutions = parsed.get("resolutions")
    ret.mode = parsed.get("mode", 'auto')
    ret.threaded = parsed.get("threaded", True)
    ret.fps = parsed.get("fps")

    return ret


def new_video_capture(config: CaptureConfig):
    device = config.source
    resolution = config.resolution
    mode = config.capture_mode

    print(f"starting video capture: {config}")

    cap = cv2.VideoCapture(device, mode)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if resolution is not None:
        req_w, req_h = resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)

    if not cap.isOpened():
        raise Exception("Cannot open capture")

    return cap
