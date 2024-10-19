#!/usr/bin/env python3
import queue
import os
import urllib
import logging
from tqdm import tqdm
from pathlib import Path
from flask import jsonify
from factgenie.campaigns import CampaignMode


# https://maxhalford.github.io/blog/flask-sse-no-deps/
class MessageAnnouncer:
    def __init__(self):
        self.listeners = []

    def listen(self):
        self.listeners.append(queue.Queue(maxsize=5))
        return self.listeners[-1]

    def announce(self, msg):
        # We go in reverse order because we might have to delete an element, which will shift the
        # indices backward
        for i in reversed(range(len(self.listeners))):
            try:
                self.listeners[i].put_nowait(msg)
            except queue.Full:
                del self.listeners[i]


def format_sse(data: str, event=None) -> str:
    """Formats a string and an event name in order to follow the event stream convention.

    >>> format_sse(data=json.dumps({'abc': 123}), event='Jackson 5')
    'event: Jackson 5\\ndata: {"abc": 123}\\n\\n'

    """
    msg = f"data: {data}\n\n"
    if event is not None:
        msg = f"event: {event}\n{msg}"
    return msg


def success(message=None):
    resp = jsonify(success=True, message=message)
    return resp


def error(err):
    resp = jsonify(success=False, error=err)
    return resp


def get_mode_from_path(path):
    # if path begins with /llm_eval, return llm_eval
    # else if path begins with /llm_gen, return llm_gen
    if path.startswith("/llm_eval"):
        return CampaignMode.LLM_EVAL
    elif path.startswith("/llm_gen"):
        return CampaignMode.LLM_GEN


# source: https://github.com/lhotse-speech/lhotse/blob/bc2c0a294b1437b90d1581d4f214348d2f8bfc12/lhotse/utils.py#L465
def resumable_download(
    url,
    filename,
    force_download,
    completed_file_size=None,
    missing_ok=True,
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Check if the file exists and get its size
    file_exists = os.path.exists(filename)
    if file_exists:
        if force_download:
            logging.info(f"Removing existing file and downloading from scratch because force_download=True: {filename}")
            os.unlink(filename)
            file_size = 0
        else:
            file_size = os.path.getsize(filename)

        if completed_file_size and file_size == completed_file_size:
            return
    else:
        file_size = 0

    # Set the request headers to resume downloading
    # Also set user-agent header to stop picky servers from complaining with 403
    ua_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30",
    }

    headers = {
        "Range": "bytes={}-".format(file_size),
        **ua_headers,
    }

    # Create a request object with the URL and headers
    req = urllib.request.Request(url, headers=headers)

    # Open the file for writing in binary mode and seek to the end
    # r+b is needed in order to allow seeking at the beginning of a file
    # when downloading from scratch
    mode = "r+b" if file_exists else "wb"
    with open(filename, mode) as f:

        def _download(rq, size):
            f.seek(size, 0)
            # just in case some garbage was written to the file, truncate it
            f.truncate()

            # Open the URL and read the contents in chunks
            with urllib.request.urlopen(rq) as response:
                chunk_size = 1024
                total_size = int(response.headers.get("content-length", 0)) + size
                with tqdm(
                    total=total_size,
                    initial=size,
                    unit="B",
                    unit_scale=True,
                    desc=str(filename),
                ) as pbar:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))

        try:
            _download(req, file_size)
        except urllib.error.HTTPError as e:
            # "Request Range Not Satisfiable" means the requested range
            # starts after the file ends OR that the server does not support range requests.
            if e.code == 404 and missing_ok:
                logging.warning(f"{url} does not exist (error 404). Skipping this file.")
                if Path(filename).is_file():
                    os.remove(filename)
            elif e.code == 416:
                content_range = e.headers.get("Content-Range", None)
                if content_range is None:
                    # sometimes, the server actually supports range requests
                    # but does not return the Content-Range header with 416 code
                    # This is out of spec, but let us check twice for pragmatic reasons.
                    head_req = urllib.request.Request(url, method="HEAD")
                    head_res = urllib.request.urlopen(head_req)
                    if head_res.headers.get("Accept-Ranges", "none") != "none":
                        content_length = head_res.headers.get("Content-Length")
                        content_range = f"bytes */{content_length}"

                if content_range == f"bytes */{file_size}":
                    # If the content-range returned by server also matches the file size,
                    # then the file is already downloaded
                    logging.info(f"File already downloaded: {filename}")
                else:
                    logging.info("Server does not support range requests - attempting downloading from scratch")
                    _download(urllib.request.Request(url, headers=ua_headers), 0)
            else:
                raise e
