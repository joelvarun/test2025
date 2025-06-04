import cv2
import numpy as np
from dataclasses import dataclass
from typing import List

try:
    import kociemba
except ImportError:
    kociemba = None

# Color thresholds in HSV space for basic Rubik's cube colors
COLOR_RANGES = {
    'U': ((0, 0, 200), (180, 30, 255)),      # white
    'R': ((0, 120, 70), (10, 255, 255)),     # red (approx)
    'F': ((40, 70, 70), (80, 255, 255)),     # green
    'D': ((20, 100, 100), (30, 255, 255)),   # yellow
    'L': ((10, 100, 20), (20, 255, 255)),    # orange
    'B': ((90, 70, 70), (130, 255, 255)),    # blue
}

@dataclass
class Face:
    colors: List[str]

@dataclass
class CubeState:
    faces: List[Face]

    def to_kociemba_string(self) -> str:
        return ''.join(''.join(face.colors) for face in self.faces)

def detect_face_colors(image: np.ndarray) -> List[str]:
    """Detect colors of a single Rubik's cube face image.

    The face image is expected to be a square containing a 3x3 grid of stickers.
    The algorithm divides the image into 3x3 cells and classifies the color of
    each cell using predefined HSV color ranges.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    cell_h, cell_w = h // 3, w // 3
    colors = []
    for row in range(3):
        for col in range(3):
            cell = hsv[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w]
            avg_color = cell.mean(axis=(0, 1))
            color_code = classify_color(avg_color)
            colors.append(color_code)
    return colors

def classify_color(hsv_color: np.ndarray) -> str:
    for code, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        if cv2.inRange(np.uint8([[hsv_color]]), lower, upper).any():
            return code
    return 'U'  # default/fallback

def load_images(image_paths: List[str]) -> List[np.ndarray]:
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Unable to load image: {path}")
        images.append(img)
    return images

def detect_cube_state(image_paths: List[str]) -> CubeState:
    if len(image_paths) != 6:
        raise ValueError("Six face images are required")
    images = load_images(image_paths)
    faces = [Face(detect_face_colors(img)) for img in images]
    return CubeState(faces)

def solve_cube(state: CubeState) -> str:
    cube_str = state.to_kociemba_string()
    if kociemba is None:
        raise RuntimeError("kociemba package not installed")
    return kociemba.solve(cube_str)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Solve a Rubik's cube from face images")
    parser.add_argument('images', nargs=6, help='Paths to six face images in URFDLB order')
    args = parser.parse_args()

    cube_state = detect_cube_state(args.images)
    solution = solve_cube(cube_state)
    print("Solution:", solution)
