#!/bin/bash

# Delete .npy, .csv, .sps, and .fits files recursively
echo "=== Searching for data files to delete ==="

# File extensions to remove
EXTENSIONS=("npy" "csv" "sps" "fits" "png")

for ext in "${EXTENSIONS[@]}"; do
    echo "  Deleting *.$ext files..."
    find . -type f -name "*.$ext" -print -delete
done

echo "=== DONE ==="