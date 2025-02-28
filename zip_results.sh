#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_folder>"
  exit 1
fi

cd ~/abus23

input_folder="$1"
parent_name=$(basename $(dirname "$input_folder"))
base_name=$(basename "$input_folder")
output_folder="${parent_name}_${base_name}"

# Check if the input_folde exists
if [ ! -d "$input_folder" ]; then
  echo "Error: Folder '$input_folder' does not exist."
  exit 1
fi

# Create the target folder 'submit_data/seg/{B}/submit'
target_folder="submit_data/seg/${output_folder}"
mkdir -p "$target_folder/submit"

# Copy data from folder A to the target folder
cp -r "$input_folder"/*.nii.gz "$target_folder/submit"

# Create a zip file of the 'submit' folder
cd "$target_folder"

zip_file="${output_folder}_submit.zip"
zip -r "$zip_file" submit

echo "Data copied and zip file created at: $(pwd)/$zip_file"