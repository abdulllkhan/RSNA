#!/usr/bin/env bash
# Every 5 min until the 384 training launches: evict clean page cache of files written in the
# last ~6 min under data/raw and data/cache* (own files; no root) so `free` stays high.
cd /home/abdulllkhan/Documents/RSNA || exit 1
while ! grep -qE 'HR FOLD0 LAUNCHED' logs/post_folds.log 2>/dev/null; do
  find data/raw data/cache384 data/cache512 -type f \( -name '*.dcm' -o -name '*.jpg' \) -mmin -6 -print0 2>/dev/null | python3 -c '
import os,sys
for p in sys.stdin.buffer.read().split(b"\0"):
    if not p: continue
    try:
        fd=os.open(p,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
    except Exception: pass'
  sleep 300
done
