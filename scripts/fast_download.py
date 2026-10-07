import sys
import os
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import urllib.error

def get_file_size(url):
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", "Mozilla/5.0")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            cl = resp.headers.get("Content-Length")
            if cl:
                return int(cl)
    except Exception:
        # Fallback to GET with range
        req = urllib.request.Request(url)
        req.add_header("Range", "bytes=0-0")
        with urllib.request.urlopen(req, timeout=15) as resp:
            cr = resp.headers.get("Content-Range")
            if cr and "/" in cr:
                return int(cr.split("/")[-1])
    raise RuntimeError("Could not determine Content-Length")

def download_chunk(url, chunk_idx, start, end, part_file, max_retries=5):
    expected_size = end - start + 1
    if part_file.exists() and part_file.stat().st_size == expected_size:
        return chunk_idx, expected_size, 0  # Already done

    headers = {
        "Range": f"bytes={start}-{end}",
        "User-Agent": "Mozilla/5.0"
    }

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) == expected_size:
                    tmp_file = part_file.with_suffix(".tmp")
                    with open(tmp_file, "wb") as f:
                        f.write(data)
                    tmp_file.replace(part_file)
                    return chunk_idx, expected_size, attempt
                else:
                    time.sleep(1)
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Chunk {chunk_idx} failed after {max_retries} retries: {e}")
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Chunk {chunk_idx} failed")

def main():
    parser = argparse.ArgumentParser(description="Multi-threaded chunk downloader")
    parser.add_argument("url", help="Download URL")
    parser.add_argument("output", help="Target output file path")
    parser.add_argument("--threads", type=int, default=16, help="Number of worker threads")
    parser.add_argument("--chunk_size_mb", type=int, default=16, help="Chunk size in MB")
    args = parser.parse_args()

    url = args.url
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = out_path.parent / f".{out_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {url} ...")
    total_size = get_file_size(url)
    print(f"Total file size: {total_size:,} bytes ({total_size / (1024**3):.2f} GiB)")

    chunk_size = args.chunk_size_mb * 1024 * 1024
    chunks = []
    start = 0
    idx = 0
    while start < total_size:
        end = min(start + chunk_size - 1, total_size - 1)
        part_file = parts_dir / f"chunk_{idx:05d}.part"
        chunks.append((idx, start, end, part_file))
        start = end + 1
        idx += 1

    total_chunks = len(chunks)
    print(f"Split into {total_chunks} chunks of {args.chunk_size_mb} MB using {args.threads} threads.")

    # Check already completed
    already_done = 0
    for c_idx, s, e, p_file in chunks:
        if p_file.exists() and p_file.stat().st_size == (e - s + 1):
            already_done += (e - s + 1)
    
    print(f"Resuming: {already_done / (1024**2):.1f} MB already downloaded ({already_done / total_size * 100:.1f}%)")

    downloaded = already_done
    start_time = time.time()
    last_print = start_time

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(download_chunk, url, c_idx, s, e, p_file): (c_idx, e - s + 1)
            for c_idx, s, e, p_file in chunks
        }

        for fut in as_completed(futures):
            c_idx, exp_size = futures[fut]
            try:
                _, size, retries = fut.result()
                downloaded += size
                now = time.time()
                if now - last_print >= 2.0 or downloaded >= total_size:
                    elapsed = now - start_time
                    spd = (downloaded - already_done) / max(elapsed, 0.001) / (1024**2)
                    pct = downloaded / total_size * 100
                    rem_sec = (total_size - downloaded) / max((spd * 1024 * 1024), 1)
                    print(f"Progress: {downloaded / (1024**2):.1f}/{total_size / (1024**2):.1f} MB ({pct:.1f}%) | "
                          f"Speed: {spd:.2f} MB/s | ETA: {int(rem_sec // 60)}m {int(rem_sec % 60)}s", flush=True)
                    last_print = now
            except Exception as e:
                print(f"Error on chunk {c_idx}: {e}", flush=True)
                raise

    print(f"\nAll {total_chunks} chunks downloaded successfully! Stitching final file...")
    temp_out = out_path.with_suffix(".stitching")
    with open(temp_out, "wb") as f_out:
        for c_idx, s, e, p_file in chunks:
            with open(p_file, "rb") as f_in:
                while True:
                    buf = f_in.read(4 * 1024 * 1024)
                    if not buf:
                        break
                    f_out.write(buf)

    if temp_out.stat().st_size != total_size:
        raise RuntimeError(f"Stitched file size mismatch: {temp_out.stat().st_size} vs {total_size}")

    if out_path.exists():
        out_path.unlink()
    temp_out.replace(out_path)
    print(f"Final file verified and saved to {out_path} ({out_path.stat().st_size:,} bytes).")

    # Clean up parts
    print("Cleaning up temporary parts...")
    for _, _, _, p_file in chunks:
        if p_file.exists():
            p_file.unlink()
    try:
        parts_dir.rmdir()
    except Exception:
        pass
    print("Done!")

if __name__ == "__main__":
    main()
