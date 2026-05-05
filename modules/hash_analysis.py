from PIL import Image
import imagehash

HASH_DB = "hash_db.txt"

def get_image_hash(path):
    return str(imagehash.phash(Image.open(path)))

def is_duplicate(hash_val):
    try:
        with open(HASH_DB, "r") as f:
            return hash_val in f.read()
    except:
        return False

def save_hash(hash_val):
    with open(HASH_DB, "a") as f:
        f.write(hash_val + "\n")