from collections import deque
import numpy as np
from detection import utils


class WindowTracker:
    def __init__(self, img, size=5):
        self.queue = deque()
        self.size = size
        self.img = img

    def add_windows(self, windows_list):
        if len(self.queue) == self.size:
            self.queue.popleft()

        self.queue.append(windows_list)

    def get_windows(self):
        heat_map = np.zeros_like(self.img[:, :, 0]).astype(np.float)
        windows_list = []

        for window_list in self.queue:
            heat_map = utils.add_heat(heat_map, window_list)
            windows_list.extend(window_list)

        labels, heat_map = utils.apply_threshold(heat_map, threshold=len(self.queue)+1)
        return labels, heat_map
