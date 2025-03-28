#!/bin/bash

PDF_DIR="/home/davud/bombonjero-ai/pdfs/3/"

for pdf in "$PDF_DIR"/*.pdf; do
    mupdf "$pdf" &
    echo "$pdf" &
    basename "$pdf" | xclip -selection clipboard
    wait
done

