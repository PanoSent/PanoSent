import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data

class ImageBindEncoder:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = imagebind_model.imagebind_huge(pretrained=True).eval().to(device)

    def encode_batch(self, input_list: list[dict]) -> torch.Tensor:
        """
        input_list: [{"image": path}, {"audio": path}, {"video": path}]
        Returns: Tensor [B, 1024]
        """
        if not input_list:
            raise ValueError("Empty multimodal input list.")

        batch_paths = []
        modality = None
        for entry in input_list:
            if "image" in entry:
                batch_paths.append(entry["image"])
                modality = ModalityType.VISION
            elif "audio" in entry:
                batch_paths.append(entry["audio"])
                modality = ModalityType.AUDIO
            elif "video" in entry:
                batch_paths.append(entry["video"])
                modality = ModalityType.VIDEO
            else:
                raise ValueError("Unsupported modality key.")

        if modality == ModalityType.VISION:
            transformed = data.load_and_transform_vision_data(batch_paths, self.device)
        elif modality == ModalityType.AUDIO:
            transformed = data.load_and_transform_audio_data(batch_paths, self.device)
        elif modality == ModalityType.VIDEO:
            transformed = data.load_and_transform_video_data(batch_paths, self.device)

        with torch.no_grad():
            outputs = self.model({modality: transformed})
        return outputs[modality]  # shape: [B, 1024]