#!/bin/bash

# Usage:
# ./convert_all.sh [--doc-level] [--segment-size N]

DOC_LEVEL=""
SEGMENT_SIZE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --doc-level)
      DOC_LEVEL="--doc-level"
      shift
      ;;
    --segment-size)
      SEGMENT_SIZE="--segment-size $2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

for dir in sample-data/*/; do
  HTML_FILE="${dir}DGT-Evaluation.html"
  if [[ -f "$HTML_FILE" ]]; then
    python convert_dgt_data.py "$HTML_FILE" --split html-only $DOC_LEVEL $SEGMENT_SIZE
  fi
done