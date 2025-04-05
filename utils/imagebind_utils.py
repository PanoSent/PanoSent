import torch
from imagebind.data.loaders import load_and_transform_vision_data, load_and_transform_audio_data
from imagebind.data.loaders import load_and_transform_video_data
from imagebind.models.modality import ModalityType

def preprocess_inputs(inputs, device):
    processed = {}
    if "image" in inputs:
        processed[ModalityType.VISION] = load_and_transform_vision_data([inputs["image"]], device)
    elif "audio" in inputs:
        processed[ModalityType.AUDIO] = load_and_transform_audio_data([inputs["audio"]], device)
    elif "video" in inputs:
        processed[ModalityType.VIDEO] = load_and_transform_video_data([inputs["video"]], device)
    return processed