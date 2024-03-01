#!/bin/bash

# Check if arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_name> <file_path>"
    exit 1
fi

dataset_name="$1"
file_path="$2"

# Check if the file exists
if [ ! -f "$file_path" ]; then
    echo "File does not exist: $file_path"
    exit 1
fi

# Extract filename from path
filename=$(basename "$file_path")

# Encode the file in base64
encoded_data=$(base64 -w 0 "$file_path")

# Create a temporary file for the JSON payload
temp_json_payload=$(mktemp)

# Write the JSON payload to the temporary file
cat > "$temp_json_payload" <<EOF
{
  "filename": "$filename",
  "data": "$encoded_data"
}
EOF

# Send the data to the API using --data-binary to read from the file
curl -X 'POST' \
    "http://localhost:8000/datasets/$dataset_name/ingest" \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    --data-binary "@$temp_json_payload"

# Clean up: Remove the temporary file

rm "$temp_json_payload"
