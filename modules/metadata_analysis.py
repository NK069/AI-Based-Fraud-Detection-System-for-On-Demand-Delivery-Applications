from PIL import Image
from PIL.ExifTags import TAGS

def extract_metadata(image_path):
    try:
        img = Image.open(image_path)
        exif = img._getexif()

        if not exif:
            return {}

        data = {}
        for tag, val in exif.items():
            name = TAGS.get(tag, tag)
            data[name] = str(val)

        return data

    except:
        return {}