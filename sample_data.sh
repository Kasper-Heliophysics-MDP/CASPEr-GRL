#!/bin/bash

# Run dropbox and collect data
echo "@@@ Collecting Data..."
#python3 ./source/dbx_sync.py
echo "@@@ Data Collection Complete!"
echo ""

echo "@@@ Convert SPS data to NPY..."
python3 ./source/convert_SPS.py
echo "@@@ Conversion Complete!"
echo ""

echo "@@@ Preprocessing Data..."
python3 ./source/preprocess_data.py
echo "@@@ Preprocessing Complete!"
echo ""

echo "@@@ Identifying Bursts..."
python3 ./source/radon_analysis.py
echo "@@@ Identification Complete!"
echo ""


