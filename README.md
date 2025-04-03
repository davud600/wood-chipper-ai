DONE: start loop... -> convert pdf page to image -> get contents from image -> save to memory -> split -> ...end loop -> process -> delete image (if processed).

DONE: save doc contents in persistent memory (redis).

DONE: use easyOCR with gpu acceleration.

DONE: processing,
DONE: training (bce with logit loss pos weight),
DONE: evaluation (f1 score),
DONE: inference (in server),

DONE: todo: processing endpoint.
DONE: avoid downloading file if contents already saved.

DONE: auto correct on ocr.

DONE: dataset parallel processing,
DONE: split pdf list & spawn multiple processes writing to different files processing different pdfs.

todo: expirement with embedding layer.

GOATED!!
`find . -maxdepth 1 -type f | head -n 10 | xargs -I{} cp "{}" .../wood-chipper-ai/dataset/pdfs/`

pipeline reports:
13 page pdf.

4 img workers (parallel), 2 ocr workers (threads) 2x2 img: 45 seconds.
4 img workers (parallel), 3 ocr workers (threads) 2x2 img: 42 seconds.
4 img workers (parallel), 2 ocr workers (parallel) 2x2 img: 41 seconds.
4 img workers (parallel), 3 ocr workers (parallel) 2x2 img: 35 seconds.
4 img workers (parallel), 3 ocr workers (parallel) 1x1 img: 35 seconds.
