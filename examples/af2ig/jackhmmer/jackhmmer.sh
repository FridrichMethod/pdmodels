#!/bin/bash

basedir=/mnt/tmp
query=./monomer.fasta
target=/mnt/d/alphafold/small_bfd/bfd-first_non_consensus_sequences.fasta
target_on_ramdisk=$basedir/"$(basename $target)"

sudo mkdir -p $basedir
sudo mount -t tmpfs -o size=64G tmpfs $basedir
echo "mounted tmpfs"

sudo cp $target "$target_on_ramdisk"
echo "copied target fasta"

time jackhmmer \
  -o /dev/null \
  -A ./small_bfd_hits.sto \
  --noali \
  --F1 0.0005 \
  --F2 0.00005 \
  --F3 0.0000005 \
  --incE 0.0001 \
  -E 0.0001 \
  --cpu 8 -N 1 \
  $query \
  "$target_on_ramdisk"
echo "searched"

# sudo rm -rf $basedir/"$(basename $target)"
# echo "removed target fasta"

# sudo umount $basedir
# echo "unmounted tmpfs"

# sudo rm -rf $basedir
# echo "removed tmpfs"
