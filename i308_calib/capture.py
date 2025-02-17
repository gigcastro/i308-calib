import platform
import cv2

FPS = 30


class SysInfo:
    def __init__(self):
        self.os_platform = platform.platform()
        self.os_system = platform.system()


def guess_capture_mode(sys_info: SysInfo) -> str:
    mode = cv2.CAP_ANY
    if sys_info.os_system == 'Windows':
        mode = cv2.CAP_DSHOW
    return mode


class CaptureConfig:

    def __init__(self, device, resolution=None):

        if isinstance(device, str) and device.isdigit():
            device = int(device)

        self.device = device
        self.resolution = resolution

        self.sys_info = SysInfo()
        self.fps = FPS

        self.capture_mode = guess_capture_mode(self.sys_info)

    def __str__(self):
        ret = f"device: {self.device}; "
        if self.resolution is not None:
            ret += f"resolution: {self.resolution};"

        return ret


def new_video_capture(config: CaptureConfig):

    device = config.device
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
