# Rubik's Cube Solver

This example application demonstrates solving a Rubik's cube using simple
computer vision techniques. It expects six images of the cube faces in URFDLB
order (Up, Right, Front, Down, Left, Back). The application attempts to detect
sticker colors on each face and computes a solution using the `kociemba` library.

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- kociemba

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Capture six images of the cube faces and run:

```bash
python -m cube_solver.solver face1.jpg face2.jpg face3.jpg face4.jpg face5.jpg face6.jpg
```

The program outputs a sequence of moves to solve the cube.

This implementation uses naive color thresholds and may require tuning for
different lighting conditions.
