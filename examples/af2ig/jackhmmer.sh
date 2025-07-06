#!/bin/bash

set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)

basedir=/tmp/jackhmmer
query=$ROOT_DIR/assets/monomers/monomers.fasta
target=$HOME/alphafold/databases/small_bfd/bfd-first_non_consensus_sequences.fasta
target_on_ramdisk="$basedir"/"$(basename "$target")"
alignment="$ROOT_DIR/assets/monomers/af2ig/msas/small_bfd_hits.sto"

mkdir -p "$(dirname "$alignment")"

# Mount a tmpfs to speed up I/O
mkdir -p "$basedir"
sudo mount -t tmpfs -o size=64G tmpfs "$basedir"

# Copy the target database to the tmpfs
cp "$target" "$target_on_ramdisk"

# Run jackhmmer
time jackhmmer \
  -o /dev/null \
  -A "$alignment" \
  --noali \
  --F1 0.0005 \
  --F2 0.00005 \
  --F3 0.0000005 \
  --incE 0.0001 \
  -E 0.0001 \
  --cpu 8 -N 1 \
  "$query" \
  "$target_on_ramdisk"

# Clean up
rm -rf "$target_on_ramdisk"
sudo umount "$basedir"
rm -rf "$basedir"
