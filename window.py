import cv2

class Window():
    """

    """
    windows = []
    def __init__(self, name=None):
        self.cv2 = cv2
        if name is None:
            name = str(len(self.windows))
        self.name = name
        if type(self.name) is not str:
            raise Exception('String names only')
        elif name in self.windows:
            raise Exception('Duplicate name')
        self.windows.append(self.name)
        self.cv2.namedWindow(self.name, self.cv2.WINDOW_NORMAL)
    def close(self):
        self.cv2.destroyWindow(self.name)

    def imshow(self, image, correct_color = True):
        if correct_color:
            image = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
        self.image = image
        self.cv2.imshow(self.name, self.image)
        cv2.waitKey(1)

    def draw(self):
        self.cv2.imshow(self.name, self.image)
        cv2.waitKey(1)
