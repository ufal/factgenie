#!/bin/bash

for dir in sample-data/*/; do
  HTML_FILE="${dir}DGT-Evaluation.html"
  if [[ -f "$HTML_FILE" ]]; then
    python convert_dgt_data.py "$HTML_FILE" --split sample $DOC_LEVEL $SEGMENT_SIZE
  fi
done

python convert_dgt_data.py sample-data/ENER-2024-00625-00-00-CS-TRA-00/DGT-Evaluation.html --split seg-1 --segment-size 1 --sdlxliff-file sample-data/ENER-2024-00625-00-00-00-CS-TRA-00.sdlxliff

python convert_dgt_data.py sample-data/ENER-2024-00625-00-00-CS-TRA-00/DGT-Evaluation.html --split seg-10 --segment-size 10 --sdlxliff-file sample-data/ENER-2024-00625-00-00-00-CS-TRA-00.sdlxliff

python convert_dgt_data.py sample-data/ENER-2024-00625-00-00-CS-TRA-00/DGT-Evaluation.html --split doc --doc-level --sdlxliff-file ENER-2024-00625-00-00-00-CS-TRA-00.sdlxliff