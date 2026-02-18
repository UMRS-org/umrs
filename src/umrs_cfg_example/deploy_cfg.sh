#!/bin/bash

# Define source directory relative to script location
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")  # Absolute path to script directory
UMRS_DIR="$SCRIPT_DIR/umrs"

# Define destination directory
CONFIG_DIR="$HOME/.config"

# Create ~/.config if it doesn't exist
mkdir -p "$CONFIG_DIR"

# Copy the whole umrs directory into ~/.config
cp -r "$UMRS_DIR" "$CONFIG_DIR"

echo "Copied $UMRS_DIR to $CONFIG_DIR"