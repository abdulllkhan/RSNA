#!/usr/bin/env bash
# One-shot: evict clean page cache for every file under the given dirs (own files; no root).
cd /home/abdulllkhan/Documents/RSNA || exit 1
find "$@" -type f \( -name '*.dcm' -o -name '*.jpg' -o -name '*.zip' \) -print0 2>/dev/null | python3 -c '
import os,sys
n=0
for p in sys.stdin.buffer.read().split(b"\0"):
    if not p: continue
    try:
        fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd); n+=1
    except Exception: pass
print("evicted",n,"files")'
