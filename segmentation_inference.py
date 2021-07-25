import numpy
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from Model import Model
class SegmentationInference:
    def __init__(self,classes_count):
        # Load model
        self.model = Model()
        self.device = self.model.device
        # Path to trained weights
        self.PATH = "./150 epoch firma.pth"
        # Load weights
        self.model.load_state_dict(torch.load(self.PATH))
        # Set GPU device if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.colors = self._make_colors(classes_count)

        print("SegmentationInference ready")

    def process(self, image_np, channel_first=False, alpha=0.35):

        # Put image to device
        image_t = torch.from_numpy(image_np).float().to(self.device)
        image_t = image_t / 256.0

        if channel_first == False:
            image_in_t = image_t.transpose(0, 2).transpose(1, 2)
        else:
            image_in_t = image_t

        image_in_t = image_in_t

        prediction_t = self.model(image_in_t.unsqueeze(0)).squeeze(0)
        prediction_t = torch.argmax(prediction_t, dim=0)

        prediction_t = prediction_t.transpose(0, 1)

        mask_t = self.colors[prediction_t, :].transpose(0, 1)

        # Mix mask with image with alpha 0.35
        result_t = (1.0 - alpha) * image_t + alpha * mask_t

        # Get results back to CPU
        prediction = prediction_t.detach().to("cpu").numpy()
        mask = mask_t.detach().to("cpu").numpy()
        result = result_t.detach().to("cpu").numpy()

        return prediction, mask, result

    # Make colors for mask
    def _make_colors(self,count):

        result = []

        result.append([0, 0, 0])
        result.append([0, 0, 1])
        result.append([0, 0, 0])
        result.append([0, 0, 0])
        result.append([0, 0, 0])

        result = torch.from_numpy(numpy.array(result)).to(self.device)

        return result
