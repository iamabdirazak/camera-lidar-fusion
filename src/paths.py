"""Where every file lives, and the settings every stage shares."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
KITTI = DATA / "kitti"
FRAMES = DATA / "frames.npz"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
CACHE = REPORTS / "cache"

DATE = "2011_09_26"
# Drives and the frame step taken from each. Variety of streets matters more than frame count:
# the first two drives give every second frame, the others one frame in twenty from new places.
TRAINING_DRIVES = {"0005": 2, "0013": 2, "0009": 20, "0011": 20, "0014": 20, "0018": 20,
                   "0022": 20, "0028": 20, "0035": 20}
VALIDATION_DRIVE = "0039"   # a whole street held out, to pick the best weights on a place never trained on
TEST_DRIVE = "0020"         # never seen until the final evaluation
STEPS = {**TRAINING_DRIVES, VALIDATION_DRIVE: 20, TEST_DRIVE: 2}

# The model sees the scene at a third of KITTI's resolution: 1 degree of rotation moves a point
# by about 4 pixels here, which is enough to see and cheap enough to run on every frame.
WIDTH, HEIGHT = 416, 128

# A calibration is "drifting" once fusion visibly breaks: LiDAR depth slides off object edges.
# Rotation is what breaks it. One degree moves every point about 4 pixels here, while a 5 cm
# shift moves a point 10 m away by about 1 pixel, too little to measure or to matter.
DRIFT_DEGREES = 1.0

RANGE_DEGREES = 10.0       # the widest rotation drift the system is trained to recover from
NUISANCE_CENTIMETRES = 5.0 # translation drift added to every sample, so the models learn to ignore it

WINDOW = 5         # frames the monitor and the corrector pool their evidence over
PASSES = 3         # corrector passes, each starting from the previous fix
SEED = 0
THREADS = 4
