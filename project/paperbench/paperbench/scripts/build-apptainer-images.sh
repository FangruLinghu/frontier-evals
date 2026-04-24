#!/bin/bash
# Build Apptainer .sif images for PaperBench on HPC clusters.
#
# Option 1: Build from Docker Hub (if images are pushed)
#   apptainer build pb-reproducer.sif docker://pb-reproducer:latest
#
# Option 2: Build from local Docker daemon (run on a machine with Docker)
#   apptainer build pb-reproducer.sif docker-daemon://pb-reproducer:latest
#
# Option 3: Build from Dockerfile via intermediate Docker image
#   docker build --platform=linux/amd64 -f paperbench/reproducer.Dockerfile -t pb-reproducer .
#   apptainer build pb-reproducer.sif docker-daemon://pb-reproducer:latest

set -e

echo "Building pb-reproducer.sif from local Docker image..."
echo "Make sure Docker is running and pb-reproducer:latest exists."
echo ""
echo "If you haven't built the Docker image yet, run:"
echo "  docker build --platform=linux/amd64 -f paperbench/reproducer.Dockerfile -t pb-reproducer ."
echo ""

apptainer build pb-reproducer.sif docker-daemon://pb-reproducer:latest

echo "Done. Transfer pb-reproducer.sif to your HPC cluster."
