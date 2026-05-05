import pickle
import numpy as np

from modules.dct_analysis import dct_analysis
from modules.fft_analysis import fft_analysis
from modules.prnu_analysis import prnu_analysis
from modules.hash_analysis import check_duplicate
from modules.metadata_analysis import metadata_analysis

def analyze_image(image_path):

    duplicate = check_duplicate(image_path)
    metadata = metadata_analysis(image_path)

    dct = dct_analysis(image_path)
    fft = fft_analysis(image_path)
    prnu = prnu_analysis(image_path)

    features = np.array([[dct, fft, prnu]])

    model = pickle.load(open("model/model.pkl", "rb"))

    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0].max() * 100

    if prediction == 1:
        result = "AI Generated Image"
        reason = "Weak camera noise and artificial patterns detected"
    else:
        result = "Real Image"
        reason = "Natural texture and camera noise present"

    return {
        "duplicate": duplicate,
        "metadata": metadata,
        "dct_msg": f"DCT Score: {round(dct,2)}",
        "fft_msg": f"FFT Score: {round(fft,2)}",
        "prnu_msg": f"PRNU Score: {round(prnu,2)}",
        "result": result,
        "confidence": round(confidence,2),
        "reason": reason
    }