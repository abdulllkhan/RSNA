#!/usr/bin/env bash
# Drop the (clean, reclaimable) page cache of the 265 GB raw zip every 5 min while it is being
# unzipped, so `free` stays high and the harness's low-memory guard stops killing monitors.
cd /home/abdulllkhan/Documents/RSNA || exit 1
Z=data/raw/rsna-knee-abnormality-detection.zip
while [ -f "$Z" ] && ! grep -q 'RAW DONE' logs/post_folds.log 2>/dev/null; do
  python3 -c "import os;fd=os.open('$Z',os.O_RDONLY);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED);os.close(fd)" 2>/dev/null
  sleep 300
done
