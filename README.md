DONE todo: start loop... -> convert pdf page to image -> get contents from image -> save to memory -> split -> ...end loop -> process -> delete image (if processed).

DONE todo: save doc contents in persistent memory (redis).

DONE todo: use easyOCR with gpu acceleration.

todo: fix:
DONE processing,
DONE training (bce with logit loss pos weight),
DONE evaluation (f1 score),
DONE inference (in server),

DONE todo: processing endpoint.
DONE avoid downloading file if contents already saved.

todo: auto correct with regex on ocr.

todo: custom tokenizer.

todo: expirement with embedding layer.

GOATED!!
`find . -maxdepth 1 -type f | head -n 10 | xargs -I{} cp "{}" .../wood-chipper-ai/dataset/pdfs/`
