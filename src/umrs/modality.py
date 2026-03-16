# list of valid modalities implemented
# Must be checked by client and server

import sys

VALID_MODALITIES = ["image", "text", "text file", "tomography_embedding", "tomography_reconstruction", "tomography_mae_reconstruction", "tomography_features_extraction"]

def is_valid_modality(modality):
    return modality in VALID_MODALITIES
    
def check_or_quit(modality):
    if not is_valid_modality(modality):
        sys.exit(f"Trying to use unsuported modality {modality}\nExiting")
