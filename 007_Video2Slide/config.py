import imagehash

# -------------- Initializations ---------------------

DOWNLOAD_DIR = "downloads"

FRAME_BUFFER_HISTORY = 15  # Length of the frame buffer history to model background.
DEC_THRESH = (
    0.75  # Threshold value, above which it is marked foreground, else background.
)
DIST_THRESH = 100  # Threshold on the squared distance between the pixel and the sample to decide whether a pixel is close to that sample.

MIN_PERCENT = (
    0.15  # %age threshold to check if there is motion across subsequent frames
)
MAX_PERCENT = (
    0.01  # %age threshold to determine if the motion across frames has stopped.
)

# Post processing

SIM_THRESHOLD = 96

HASH_SIZE = 12

HASH_FUNC = "dhash"

HASH_BUFFER_HISTORY = 5

HASH_FUNC_DICT = {
    "dhash": imagehash.dhash,
    "phash": imagehash.phash,
    "ahash": imagehash.average_hash,
}

# ----------------------------------------------------