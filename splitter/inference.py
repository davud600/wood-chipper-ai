import torch
import numpy as np

from torchvision import transforms
from PIL import Image

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from .model import FusionModel


from .config import use_fp16
from config.settings import max_length, image_output_size

dtype = torch.float16 if use_fp16 else torch.float32


def is_first_page(
    # tokenizer: "PreTrainedTokenizer",
    # model: "FusionModel",
    content: str,
    prev_first_page_distance: int,
    image: np.ndarray | None,
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

    try:
        a = content.split("</prev_page_1>")[1].lower()
        b = a.split("</curr_page>")[0]
    except:
        return False, 0

    if (
        "newdocumentseparator"
        in b
        # or "indicator of the starting of new document" in content.lower()
        # or "this serves as an" in b
        # or "indicator of the starting of new document" in content.lower()
    ):
        return True, 1

    return False, 0

    if prev_first_page_distance < 4:
        return False, 0

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_output_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.ConvertImageDtype(dtype),
        ]
    )

    encoding = tokenizer(
        content,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to("cuda")  # type: ignore
    attention_mask = encoding["attention_mask"].to("cuda")  # type: ignore
    distance = prev_first_page_distance / 700
    distance = torch.tensor([[distance]], dtype=dtype).to("cuda")  # shape (1, 1)

    cnn_input = (
        transform(Image.fromarray(image))
        if image is not None
        else transform(
            Image.fromarray(torch.zeros(image_output_size, dtype=dtype))  # type: ignore
        )
    )

    with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=dtype):
        logits = model(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "cnn_input": cnn_input,
                "distance": distance,
            }
        )

    print(f"pred_prob: {torch.sigmoid(logits).detach().cpu().numpy().item():.4f}")
    return bool(torch.sigmoid(logits) > 0.5), 0
