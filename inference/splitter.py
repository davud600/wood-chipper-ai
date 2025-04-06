import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
    from training.model import SplitterModel

from config import max_length


def is_first_page(
    tokenizer: "PreTrainedTokenizer", model: "SplitterModel", content: str
) -> tuple[bool, int]:
    """
    returns -> (is first page (bool), offset (int))
    """

    if "newdocumentseparator" in content.split("</curr_page>")[0]:
        return True, 1

    tokenized = tokenizer(
        [content],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    features = tokenized.input_ids.to("cuda")

    with torch.amp.autocast_mode.autocast(device_type="cuda", dtype=torch.float16):
        logit = model(features)
        page_class = int(logit > 0)

        if page_class == 1:
            return True, 0

    return False, 0
