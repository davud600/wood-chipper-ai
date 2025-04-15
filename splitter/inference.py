import torch
import numpy as np

from torchvision import transforms
from PIL import Image

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from .model import FusionModel

from config.settings import max_length, image_output_size, doc_length_bins


def is_first_page(
    tokenizer: "PreTrainedTokenizer",
    model: "FusionModel",
    content: str,
    images: list[np.ndarray],
    prev_first_page_distance: int,
) -> tuple[bool, int]:
    """
    Predicts whether a given page is the first page of a new document.

    Combines encoded text and preprocessed image inputs to make a prediction
    using the trained fusion model. Returns a boolean and an offset (currently always 0).

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        A tokenizer compatible with the model's language backbone (e.g., from HuggingFace).

    model : FusionModel
        A trained fusion model for multimodal inference.

    content : str
        The text content of the current page or group of pages.

    images : list of np.ndarray
        A list of image arrays corresponding to the page(s).

    Returns
    -------
    is_first : bool
        True if the model predicts this is the first page of a document.

    offset : int
        Page offset (currently unused and always 0).
    """

    # split_content = content.split("</curr_page>")[0].lower()
    #
    # if (
    #     "newdocumentseparator" in split_content
    #     or "indicator of the starting of new document" in content.lower()
    #     or "this serves as an" in split_content
    #     or "indicator of the starting of new document" in content.lower()
    # ):
    #     return True, 1

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_output_size),
            transforms.ToTensor(),
        ]
    )

    tensor_images = []
    for image in images:
        pil_img = Image.fromarray(image)
        tensor_img = transform(pil_img)
        tensor_images.append(tensor_img)

    cnn_input = torch.cat(tensor_images, dim=0)
    cnn_input = cnn_input.unsqueeze(0).to(torch.float16).to("cuda")

    encoding = tokenizer(
        content,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to("cuda")  # type: ignore
    attention_mask = encoding["attention_mask"].to("cuda")  # type: ignore
    distance = prev_first_page_distance / max(doc_length_bins)
    distance = torch.tensor([distance], dtype=torch.float16).to("cuda")

    # with torch.no_grad():
    with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(input_ids, attention_mask, cnn_input, distance)

    pred_prob = torch.sigmoid(logits[0]).detach().cpu().numpy()
    print("pred_prob", pred_prob)

    return bool(torch.sigmoid(logits) > 0.5), 0
