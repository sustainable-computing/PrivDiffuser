#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Define Variables ---
# Use these variables to make the script easy to update
DOCKER_IMAGE_NAME="neilyxin/privdiffuser" # e.g., myuser/myproject
DOCKER_IMAGE_TAG="latest" # Or a version number like "1.0"

echo "Starting Docker image build process..."

# --- 2. Download Large Files from Git LFS ---
# NOTE: This assumes you have git-lfs installed in the local environment where you run the script.
echo "Downloading large files from Git LFS..."

# The core command to get the actual files.
# `git lfs pull` will download all LFS files that are part of the current branch.
git lfs pull

# --- 3. Build the Docker Image ---
echo "Building Docker image: ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"

# The `docker build` command uses the Dockerfile in the current directory.
# The `--tag` flag names and tags the image.
# The `.` at the end tells Docker to use the current directory as the build context.
docker build --tag "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" .

echo "Docker image built successfully!"