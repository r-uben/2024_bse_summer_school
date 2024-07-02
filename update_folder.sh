#!/bin/bash

# Define source and destination directories
SOURCE_DIR="../24DS02 - Harnessing Language Models Your Path to NLP Expert"
DESTINATION_DIR="sessions"

# Process source directory name
PROCESSED_SOURCE_DIR=$(echo "$SOURCE_DIR" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist."
    exit 1
fi

# Create the destination directory if it doesn't exist
mkdir -p "$DESTINATION_DIR"

# Copy all subfolders from SOURCE_DIR to destination
for dir in "$SOURCE_DIR"/*; do
    if [ -d "$dir" ]; then
        # Convert directory name to lowercase and remove spaces
        new_name=$(basename "$dir" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        cp -R "$dir" "$DESTINATION_DIR/$new_name"
        echo "Contents from '$(basename "$dir")' have been copied to '$DESTINATION_DIR/$new_name'"
    fi
done

echo "All subfolders from '$SOURCE_DIR' have been copied to the '$DESTINATION_DIR' directory with lowercase names and spaces removed"
