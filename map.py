import base64
import collections
from PIL import Image
import numpy as np
import io
piecesMap = {
    "white_bishop": "B",
    "white_queen": "Q",
    "white_king": "K",
    "white_knight": "N",
    "white_rook": "R",
    "white_pawn": "P",
    "black_queen": "q",
    "black_rook": "r",
    "black_king": "k",
    "black_knight": "n",
    "black_pawn": "p",
    "black_bishop": "b",
    "empty": "1"
}

drawableMap = {
    "B": "wb_foreground",
    "Q": "wq_foreground",
    "K": "wk_foreground",
    "N": "wn_foreground",
    "R": "wr_foreground",
    "P": "wp_foreground",
    "q": "bq_foreground",
    "r": "br_foreground",
    "k": "bk_foreground",
    "n": "bn_foreground",
    "p": "bp_foreground",
    "b": "bb_foreground",
    "1": None
}

def pyimageToBitmap(obj):
    if obj is None:
        return None
    decode_byte = base64.b64decode(obj)
    image = Image.open(io.BytesIO(decode_byte))
    return np.array(image)

def mapStringToIndex(s):
    return piecesMap.get(s)

def mapDrawableFromIndex(s):
    return drawableMap.get(s)