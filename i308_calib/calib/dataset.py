class Dataset:

    def __init__(self):

        self.image_number = 0
        self.image_shape = None
        self.object_points = []
        self.image_points = []

    def add(
            self,
            image,
            object_points,
            image_points
    ):

        w, h = image.shape[1], image.shape[0]

        shape = (w, h)
        if self.image_shape is None:
            self.image_shape = shape
        else:
            if self.image_shape != shape:
                raise Exception(f"image shape has changed! {self.image_shape} != (added) {shape}")

        self.object_points.append(object_points)
        self.image_points.append(image_points)
        self.image_number += 1
        return self.image_number


class StereoDataset:

    def __init__(self):
        self.image_number = 0
        self.left = Dataset()
        self.right = Dataset()

    def add(
            self,
            left,
            right
    ):
        self.left.add(*left)
        self.right.add(*right)

        self.image_number += 1
        return self.image_number
