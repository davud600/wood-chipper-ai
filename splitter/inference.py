import torch
import numpy as np

from torchvision import transforms
from PIL import Image

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from .model import FusionModel


from .config import device
from config.settings import max_length, image_output_size


def is_first_page(
    tokenizer: "PreTrainedTokenizer",
    model: "FusionModel",
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

    # if (
    #     "newdocumentseparator"
    #     in b
    #     # or "indicator of the starting of new document" in content.lower()
    #     # or "this serves as an" in b
    #     # or "indicator of the starting of new document" in content.lower()
    # ):
    #     print(f"{b[:100]}...")
    #     return True, 1
    #
    # if (
    #     "insurance options"
    #     in b
    #     # or "indicator of the starting of new document" in content.lower()
    #     # or "this serves as an" in b
    #     # or "indicator of the starting of new document" in content.lower()
    # ):
    #     print(f"{b[:100]}...")
    #     return False, 0
    #
    # # return False, 0

    if "newdocument" in b:
        # print(f"\nfound start of new doc page")
        # print(b)
        return True, 0
    else:
        return False, 0

    if prev_first_page_distance < 4:
        return False, 0

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_output_size),
            transforms.ToTensor(),
        ]
    )

    # tensor_images = []
    # for image in images:
    #     pil_img = Image.fromarray(image)
    #     tensor_img = transform(pil_img)
    #     tensor_images.append(tensor_img)

    # cnn_input = torch.cat(tensor_images, dim=0)
    # cnn_input = cnn_input.unsqueeze(0).to(torch.float16).to(device)

    encoding = tokenizer(
        content,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to("cuda")  # type: ignore
    attention_mask = encoding["attention_mask"].to("cuda")  # type: ignore
    # distance = prev_first_page_distance / 700
    # distance = torch.tensor([distance], dtype=torch.float16).to("cuda")
    distance = prev_first_page_distance / 700
    distance = torch.tensor([[distance]], dtype=torch.float16).to(
        "cuda"
    )  # shape (1, 1)

    # images = []
    # for i, image_tensor in enumerate(cnn_input):
    #     # 📌 ADD POSITION MASK
    #     pos_mask = torch.full_like(image_tensor, 0.1)
    #     if i == 1:  # only current page gets 1s
    #         pos_mask.fill_(1.0)
    #
    #     # Combine image + mask → (2, H, W)
    #     combined = torch.stack([image_tensor, pos_mask], dim=0)
    #
    #     image_tensor = image_tensor.squeeze(0)  # (H, W)
    #     pos_mask = torch.full_like(image_tensor, 0.25)
    #     if i == 1:
    #         pos_mask.fill_(1.0)
    #
    #     combined = torch.stack([image_tensor, pos_mask], dim=0)  # (2, H, W)
    #     images.append(combined)

    images_out = []
    for i, image in enumerate(images):
        pil_img = Image.fromarray(image)
        image_tensor = transform(pil_img)  # (1, H, W)

        pos_mask = torch.full_like(image_tensor, 0.25)
        if i == 1:  # current page
            pos_mask.fill_(1.0)

        combined = torch.cat([image_tensor, pos_mask], dim=0)  # (2, H, W)
        images_out.append(combined)

    cnn_input = torch.stack(images_out, dim=0)  # (num_pages, 2, H, W)
    cnn_input = (
        cnn_input.view(1, -1, *cnn_input.shape[-2:]).to(torch.float16).to(device)
    )  # (1, C, H, W)

    # with torch.no_grad():
    with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "cnn_input": cnn_input,
                "distance": distance,
            }
        )

    # pred_prob = torch.sigmoid(logits[0]).detach().cpu().numpy()
    print("pred_prob", torch.sigmoid(logits[0]).detach().cpu().numpy())

    return bool(torch.sigmoid(logits) > 0.4), 0
