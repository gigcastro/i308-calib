class CroppedCapture:
    def __init__(self, cap, crop=None):
        self.cap = cap
        self.crop = crop

    def read(self):
        ret, frame = self.cap.read()
        if ret and self.crop:
            x0, xf, y0, yf = self.crop

            # Convert proportions to pixel coordinates
            h, w = frame.shape[:2]
            x1, x2 = int(x0 * w), int(xf * w)
            y1, y2 = int(y0 * h), int(yf * h)
            frame = frame[y1:y2, x1:x2]

        return ret, frame

    def __getattr__(self, name):
        """Delegate all unknown attributes/methods to the original capture"""
        return getattr(self.cap, name)