#!/bin/bash
echo "Attempting to create swap file..."
# Try fallocate first, then dd
if fallocate -l 1G /swapfile; then
    echo "Created swapfile with fallocate"
elif dd if=/dev/zero of=/swapfile bs=1M count=1024; then
    echo "Created swapfile with dd"
else
    echo "Failed to create swapfile"
fi

chmod 600 /swapfile
if mkswap /swapfile; then
    if swapon /swapfile; then
        echo "Swap enabled successfully"
    else
        echo "Failed to enable swap (swapon failed - likely restricted permissions)"
    fi
else
    echo "Failed to mkswap"
fi

# Start application
echo "Starting Uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
