#!/usr/bin/env python3

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from lxml import etree
from slugify import slugify

# Set factgenie path
FACTGENIE_PATH = Path("/home/kasner/dgt-workshop/factgenie/factgenie")


def generate_db_csv_from_jsonl_files(campaign_dir):
    """Generate db.csv based on existing JSONL files in the campaign files directory."""
    files_dir = campaign_dir / "files"

    if not files_dir.exists():
        print(f"Warning: No files directory found at {files_dir}")
        return

    # Find all JSONL files in the files directory
    jsonl_files = list(files_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"Warning: No JSONL files found in {files_dir}")
        return

    # Collect all entries from JSONL files
    db_entries = []

    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        db_entries.append(
                            {
                                "dataset": data.get("dataset", ""),
                                "split": data.get("split", ""),
                                "example_idx": data.get("example_idx", 0),
                                "setup_id": data.get("setup_id", ""),
                            }
                        )
        except Exception as e:
            print(f"Warning: Could not process {jsonl_file}: {e}")
            continue

    if not db_entries:
        print("Warning: No valid entries found in JSONL files")
        return

    # Sort entries by dataset, split, setup_id, and example_idx for consistent ordering
    db_entries.sort(key=lambda x: (x["dataset"], x["split"], x["setup_id"], x["example_idx"]))

    # Write db.csv
    csv_file = campaign_dir / "db.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("dataset,split,example_idx,setup_id,batch_idx,annotator_group,annotator_id,status,start,end\n")
        for entry in db_entries:
            f.write(f"{entry['dataset']},{entry['split']},{entry['example_idx']},{entry['setup_id']},0,0,,finished,,\n")

    print(f"Generated {csv_file} with {len(db_entries)} entries from {len(jsonl_files)} JSONL files")


def extract_error_statistics(soup):
    """Extract error statistics from the HTML table to create annotation categories."""
    categories = []
    error_table = None

    # Define a color palette for up to 10 categories (will cycle if more)
    color_palette = [
        "#d62626",  # Red
        "#2563eb",  # Blue
        "#E6AB02",  # Yellow
        "#059669",  # Green
        "#7c3aed",  # Purple
        "#ea580c",  # Orange
        "#0891b2",  # Cyan
        "#be123c",  # Pink
        "#65a30d",  # Lime
        "#a21caf",  # Magenta
    ]

    # Find the error statistics table
    for table in soup.find_all("table"):
        headers = table.find_all("td")
        if any("Error type" in str(header) for header in headers):
            error_table = table
            break

    if not error_table:
        return categories

    rows = error_table.find_all("tr")[3:]  # Skip header rows

    for idx, row in enumerate(rows):
        cells = row.find_all("td")
        if len(cells) >= 2:
            error_type = cells[0].get_text(strip=True)
            error_code = cells[1].get_text(strip=True)

            if error_type and error_code:
                # Get color from palette (cycle if we have more than 10 categories)
                base_color = color_palette[idx % len(color_palette)]

                # Convert hex to lighter version for minor errors
                # Remove # and convert to RGB
                hex_color = base_color.lstrip("#")
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)

                # Make lighter by blending with white (increase RGB values)
                light_r = min(255, r + (255 - r) * 0.4)
                light_g = min(255, g + (255 - g) * 0.4)
                light_b = min(255, b + (255 - b) * 0.4)

                light_color = f"#{int(light_r):02x}{int(light_g):02x}{int(light_b):02x}"

                # Create minor version (lighter color)
                categories.append(
                    {"name": f"{error_code.lower()}-", "color": light_color, "description": f"{error_type} (Minor)"}
                )
                # Create major version (base color)
                categories.append(
                    {"name": f"{error_code.lower()}+", "color": base_color, "description": f"{error_type} (Major)"}
                )

    # Add "unknown" category for cases where error category is missing
    categories.append({"name": "unknown", "color": "#808080", "description": "Unknown error category"})

    return categories


def extract_segments_from_sdlxliff(sdlxliff_path):
    """Extract segments from SDLXLIFF file."""
    segments = {}

    with open(sdlxliff_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse XML with lxml
    try:
        root = etree.fromstring(content.encode("utf-8"))
    except etree.XMLSyntaxError:
        print(f"Warning: Could not parse SDLXLIFF file {sdlxliff_path}")
        return segments

    # Define namespaces
    namespaces = {"xliff": "urn:oasis:names:tc:xliff:document:1.2", "sdl": "http://sdl.com/FileTypes/SdlXliff/1.0"}

    # Find all translation units
    for trans_unit in root.findall(".//xliff:trans-unit", namespaces):
        # Extract seg-source and target (not source and target)
        seg_source_elem = trans_unit.find("xliff:seg-source", namespaces)
        target_elem = trans_unit.find("xliff:target", namespaces)

        if seg_source_elem is not None and target_elem is not None:
            # Find segments within seg-source and target
            source_segments = seg_source_elem.findall('.//xliff:mrk[@mtype="seg"]', namespaces)
            target_segments = target_elem.findall('.//xliff:mrk[@mtype="seg"]', namespaces)

            # Match source and target segments by mid
            for src_seg in source_segments:
                mid = src_seg.get("mid")
                if mid:
                    # Find corresponding target segment
                    target_text = ""
                    for tgt_seg in target_segments:
                        if tgt_seg.get("mid") == mid:
                            # Extract clean text from target segment
                            target_text = extract_clean_text_from_sdlxliff_segment(tgt_seg)
                            break

                    # Extract source text
                    source_text = extract_clean_text_from_sdlxliff_segment(src_seg)

                    segments[mid] = {
                        "segment_id": mid,
                        "source_segment": source_text,
                        "target_text": target_text,
                        "error_code": "",  # Will be filled from HTML if available
                        "target_revision_cell": None,  # Will be filled from HTML if available
                    }

    return segments


def extract_clean_text_from_sdlxliff_segment(segment_elem):
    """Extract clean text from an SDLXLIFF segment, removing revision marks but keeping the final text."""
    import copy

    # Clone the element to avoid modifying original
    seg_copy = copy.deepcopy(segment_elem)

    # Remove deleted text (x-sdl-deleted marks)
    for deleted in seg_copy.findall(
        './/xliff:mrk[@mtype="x-sdl-deleted"]', {"xliff": "urn:oasis:names:tc:xliff:document:1.2"}
    ):
        parent = deleted.getparent()
        if parent is not None:
            parent.remove(deleted)

    # Keep added text but remove the markup (x-sdl-added marks)
    for added in seg_copy.findall(
        './/xliff:mrk[@mtype="x-sdl-added"]', {"xliff": "urn:oasis:names:tc:xliff:document:1.2"}
    ):
        text = added.text or ""
        tail = added.tail or ""
        parent = added.getparent()
        if parent is not None:
            prev = added.getprevious()
            if prev is not None:
                prev.tail = (prev.tail or "") + text + tail
            else:
                parent.text = (parent.text or "") + text + tail
            parent.remove(added)

    # Remove comment marks but keep the text
    for comment in seg_copy.findall(
        './/xliff:mrk[@mtype="x-sdl-comment"]', {"xliff": "urn:oasis:names:tc:xliff:document:1.2"}
    ):
        text = comment.text or ""
        tail = comment.tail or ""
        parent = comment.getparent()
        if parent is not None:
            prev = comment.getprevious()
            if prev is not None:
                prev.tail = (prev.tail or "") + text + tail
            else:
                parent.text = (parent.text or "") + text + tail
            parent.remove(comment)

    # Extract final text
    return "".join(seg_copy.itertext()).strip()


def extract_segments(soup):
    """Extract segment information from the HTML."""
    segments = []

    # Find the specific segment table by looking for the one with "Segment Id" header
    segment_table = None
    for table in soup.find_all("table"):
        first_row = table.find("tr")
        if first_row:
            header_cells = first_row.find_all("td")
            if any("Segment Id" in cell.get_text() for cell in header_cells):
                segment_table = table
                break

    if not segment_table:
        return segments

    rows = segment_table.find_all("tr")[1:]  # Skip header row

    i = 0
    while i < len(rows):
        row = rows[i]
        cells = row.find_all("td")

        # Check if this is a main segment row (has segment_id in first cell)
        if len(cells) >= 3:
            segment_id_text = cells[0].get_text(strip=True)

            # Only process if we have a valid segment ID (numeric)
            if segment_id_text.isdigit():
                segment_id = segment_id_text
                source_segment = cells[1].get_text(strip=True)
                target_revision = cells[2]

                # Initialize error codes and comments lists
                error_codes = []
                comments = []

                # Check if there's an error code column in the current row (index 3)
                if len(cells) >= 4:
                    error_code_text = cells[3].get_text(strip=True)
                    if error_code_text:
                        error_codes.append(error_code_text)
                        # If there's a comment column (index 4), extract it
                        if len(cells) >= 5:
                            comments.append(cells[4].get_text(strip=True))
                        else:
                            comments.append("")

                # Look for additional error codes in subsequent rows
                next_row_idx = i + 1
                rows_to_skip = 0

                while next_row_idx < len(rows):
                    next_row = rows[next_row_idx]
                    next_cells = next_row.find_all("td")

                    # Check if next row has error code (first cell is not a number and not empty)
                    if (
                        len(next_cells) >= 1
                        and next_cells[0].get_text(strip=True)
                        and not next_cells[0].get_text(strip=True).isdigit()
                    ):
                        error_codes.append(next_cells[0].get_text(strip=True))
                        # If there's a comment in the next row
                        if len(next_cells) >= 2:
                            comments.append(next_cells[1].get_text(strip=True))
                        else:
                            comments.append("")
                        rows_to_skip += 1
                        next_row_idx += 1
                    else:
                        # Hit a segment row or empty row, stop looking
                        break

                # Process error codes to handle major/minor format
                processed_error_codes = []
                for error_code in error_codes:
                    if error_code:
                        # Check for major error pattern (uppercase + '+')
                        if error_code.endswith("+"):
                            # Convert to lowercase + '+'
                            processed_error_codes.append(error_code.lower())
                        # Check for minor error pattern (lowercase + '-')
                        elif error_code.endswith("-"):
                            # Keep as is
                            processed_error_codes.append(error_code)
                        else:
                            # If there's an error code but no severity indicator, use "unknown"
                            processed_error_codes.append("unknown")
                    else:
                        processed_error_codes.append("unknown")

                # If no error codes found but we have corrections, add "unknown"
                if not processed_error_codes:
                    target_str = str(target_revision)
                    if any(
                        marker in target_str
                        for marker in ["<strike>", "background-color:yellow", 'color="red"', 'color="purple"']
                    ):
                        processed_error_codes.append("unknown")
                        comments.append("")

                segments.append(
                    {
                        "segment_id": segment_id,
                        "source_segment": source_segment,
                        "target_revision": target_revision,
                        "error_codes": processed_error_codes,  # Now a list
                        "comments": comments,  # Now a list
                    }
                )

                # Skip the error code rows we processed
                i += 1 + rows_to_skip
            else:
                i += 1
        else:
            i += 1

    return segments


def merge_segments(html_segments, sdlxliff_segments):
    """Merge segments from HTML and SDLXLIFF, prioritizing HTML for annotations."""
    merged_segments = []
    html_segment_ids = {seg["segment_id"] for seg in html_segments}

    # Add all HTML segments first (these may have annotations)
    for html_seg in html_segments:
        merged_segments.append(html_seg)

    # Add SDLXLIFF segments that are not in HTML
    for seg_id, sdlxliff_seg in sdlxliff_segments.items():
        if seg_id not in html_segment_ids:
            # Convert SDLXLIFF segment to the format expected by the rest of the code
            # For plain text, we don't need BeautifulSoup
            merged_segments.append(
                {
                    "segment_id": seg_id,
                    "source_segment": sdlxliff_seg["source_segment"],
                    "target_revision": sdlxliff_seg["target_text"],  # Keep as plain text
                    "error_codes": [],  # No error codes for segments not in HTML
                    "comments": [],  # No comments for segments not in HTML
                }
            )

    # Sort by segment_id (numeric)
    merged_segments.sort(key=lambda x: int(x["segment_id"]))
    return merged_segments


def extract_original_target(target_revision_cell):
    """Extract the original target text before edits."""
    # Handle case where target_revision_cell is a string (from SDLXLIFF)
    if isinstance(target_revision_cell, str):
        return target_revision_cell

    # Only create BeautifulSoup for actual HTML content
    target_str = str(target_revision_cell)

    # Check if it's actually HTML content (contains HTML tags)
    if not ("<" in target_str and ">" in target_str):
        return target_str

    # Clone the cell to avoid modifying the original
    cell_copy = BeautifulSoup(target_str, "html.parser")

    # Remove purple underlined text (additions)
    for elem in cell_copy.find_all(["font", "span"], color="purple"):
        elem.decompose()
    for elem in cell_copy.find_all("u"):
        if elem.parent and elem.parent.get("color") == "purple":
            elem.decompose()

    # Keep red strikethrough text (deletions) but remove the formatting
    for elem in cell_copy.find_all("font", color="red"):
        if elem.find("strike"):
            strike_elem = elem.find("strike")
            strike_elem.unwrap()  # Remove <strike> but keep content
        elem.unwrap()  # Remove <font> but keep content

    # Remove yellow highlighting but keep the text
    for elem in cell_copy.find_all("span", style=lambda x: x and "background-color:yellow" in x):
        elem.unwrap()

    # Don't strip - preserve all whitespace
    return cell_copy.get_text()


def extract_annotations(target_revision_cell, error_codes, comments, categories):
    """Extract annotations from the target revision cell with precise error-highlight matching."""
    annotations = []

    if not error_codes:
        return annotations

    # Handle case where target_revision_cell is a string (from SDLXLIFF)
    if isinstance(target_revision_cell, str):
        return annotations

    # Check if it's actually HTML content (contains HTML tags)
    target_str = str(target_revision_cell)
    if not ("<" in target_str and ">" in target_str):
        return annotations

    # Get the original text (with deleted parts) for position calculation
    original_text = extract_original_target(target_revision_cell)

    # Create a copy to work with
    cell_soup = BeautifulSoup(target_str, "html.parser")

    # Find all yellow highlights with their positions
    yellow_highlights = []
    for span in cell_soup.find_all("span", style=lambda x: x and "background-color:yellow" in x):
        highlight_text = span.get_text()
        highlight_pos = original_text.find(highlight_text)
        if highlight_pos != -1:
            yellow_highlights.append(
                {"text": highlight_text, "pos": highlight_pos, "end_pos": highlight_pos + len(highlight_text)}
            )

    # Find all deletions with their positions
    deletions = []
    for font_elem in cell_soup.find_all("font", color="red"):
        strike_elem = font_elem.find("strike")
        if strike_elem:
            deleted_text = strike_elem.get_text()
            start_pos = original_text.find(deleted_text)

            if start_pos != -1:
                # Find adjacent added text by looking at siblings
                reason = ""

                # Check previous sibling
                prev_sibling = font_elem.previous_sibling
                if prev_sibling and prev_sibling.name == "font" and prev_sibling.get("color") == "purple":
                    u_elem = prev_sibling.find("u")
                    if u_elem:
                        reason = f"Add: {u_elem.get_text()}"

                # If no previous, check next sibling
                if not reason:
                    next_sibling = font_elem.next_sibling
                    if next_sibling and next_sibling.name == "font" and next_sibling.get("color") == "purple":
                        u_elem = next_sibling.find("u")
                        if u_elem:
                            reason = f"Add: {u_elem.get_text()}"

                deletions.append(
                    {"text": deleted_text, "pos": start_pos, "end_pos": start_pos + len(deleted_text), "reason": reason}
                )

    # Match deletions with yellow highlights and assign error codes
    for deletion in deletions:
        # Find the closest yellow highlight to this deletion
        closest_highlight_idx = None
        min_distance = float("inf")

        for i, highlight in enumerate(yellow_highlights):
            # Calculate distance between deletion and highlight
            # Use the minimum distance between their boundaries
            deletion_center = deletion["pos"] + len(deletion["text"]) // 2
            highlight_center = highlight["pos"] + len(highlight["text"]) // 2
            distance = abs(deletion_center - highlight_center)

            if distance < min_distance:
                min_distance = distance
                closest_highlight_idx = i

        # Determine error code and comment
        error_code = "unknown"
        comment = ""

        if closest_highlight_idx is not None and closest_highlight_idx < len(error_codes):
            error_code = error_codes[closest_highlight_idx]
            if closest_highlight_idx < len(comments):
                comment = comments[closest_highlight_idx]
        elif len(error_codes) > 0:
            # Use first error code if no good match found
            error_code = error_codes[0]
            if len(comments) > 0:
                comment = comments[0]

        # Build reason with comment
        reason = deletion["reason"]
        if comment:
            if reason:
                reason = f"{reason} – {comment}"
            else:
                reason = f"{comment}"

        # Find error type index
        error_type_idx = 0
        for i, category in enumerate(categories):
            if category["name"] == error_code:
                error_type_idx = i
                break

        annotations.append(
            {
                "reason": reason,
                "text": deletion["text"],
                "type": error_type_idx,
                "start": deletion["pos"],
            }
        )

    return annotations


def extract_doc_level_annotations(segments, categories):
    """Extract document-level annotations with character positions adjusted for concatenated text."""
    all_annotations = []
    current_position = 0

    for segment in segments:
        # Extract annotations for this segment
        segment_annotations = extract_annotations(
            segment["target_revision"], segment.get("error_codes", []), segment.get("comments", []), categories
        )

        # Adjust character positions for document-level
        for annotation in segment_annotations:
            annotation["start"] += current_position
            all_annotations.append(annotation)

        # Update position for next segment
        segment_text = extract_original_target(segment["target_revision"])
        current_position += len(segment_text) + 1  # +1 for newline separator

    return all_annotations


def extract_doc_level_annotations_for_chunk(segments, categories, start_segment_idx, end_segment_idx):
    """Extract document-level annotations for a chunk of segments with character positions adjusted for concatenated text."""
    all_annotations = []
    current_position = 0

    for i in range(start_segment_idx, end_segment_idx):
        segment = segments[i]
        # Extract annotations for this segment
        segment_annotations = extract_annotations(
            segment["target_revision"], segment.get("error_codes", []), segment.get("comments", []), categories
        )

        # Adjust character positions for document-level
        for annotation in segment_annotations:
            annotation["start"] += current_position
            all_annotations.append(annotation)

        # Update position for next segment
        segment_text = extract_original_target(segment["target_revision"])
        current_position += len(segment_text) + 1  # +1 for space separator

    return all_annotations


def parse_heading_filename(soup):
    """Parse dataset name and setup ID from the document numbers table."""
    # Find the table with "Assignment identification" header
    for table in soup.find_all("table"):
        h2_tag = table.find("h2")
        if h2_tag and "Assignment identification" in h2_tag.get_text():
            # Look for the row with "Document numbers"
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2 and "Document numbers" in cells[0].get_text():
                    filename = cells[1].get_text(strip=True)
                    # Remove .sdlxliff extension if present
                    original_filename = filename
                    filename = filename.replace(".sdlxliff", "")

                    parts = filename.split("-")
                    if len(parts) >= 8:
                        dataset_name = slugify("-".join(parts[:3])).lower()
                        setup_id = slugify("-".join(parts[6:8])).lower()
                        return dataset_name, setup_id, original_filename
                    break
            break

    return "UNKNOWN", "UNKNOWN", "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(description="Convert DGT HTML evaluation report to factgenie format")
    parser.add_argument("html_file", help="Path to the HTML evaluation report")
    parser.add_argument("--sdlxliff-file", help="Path to the SDLXLIFF file (optional)")
    parser.add_argument("--campaign-id", default="eval-feedback", help="Campaign ID for annotations")
    parser.add_argument("--split", default="sample", help="Dataset split name")
    parser.add_argument("--dataset-name", help="Override dataset name (default: extract from HTML heading)")
    parser.add_argument("--setup-id", help="Override setup ID (default: extract from HTML heading)")
    parser.add_argument(
        "--doc-level", action="store_true", help="Process as single document instead of individual segments"
    )
    parser.add_argument(
        "--segment-size", type=int, help="Number of segments to concatenate into each sample (overrides --doc-level)"
    )

    args = parser.parse_args()

    # Parse HTML file
    with open(args.html_file, "r", encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content, "html.parser")

    # Extract dataset name and setup ID
    if args.dataset_name and args.setup_id:
        dataset_name = args.dataset_name
        setup_id = args.setup_id
        # Still need to extract filename for input data
        _, _, filename = parse_heading_filename(soup)
    else:
        dataset_name, setup_id, filename = parse_heading_filename(soup)
        if args.dataset_name:
            dataset_name = args.dataset_name
        if args.setup_id:
            setup_id = args.setup_id

    # Extract error statistics for annotation categories
    categories = extract_error_statistics(soup)

    # Extract segments from HTML
    html_segments = extract_segments(soup)

    # Extract segments from SDLXLIFF if provided
    segments = html_segments
    if args.sdlxliff_file and os.path.exists(args.sdlxliff_file):
        print(f"Loading segments from SDLXLIFF file: {args.sdlxliff_file}")
        sdlxliff_segments = extract_segments_from_sdlxliff(args.sdlxliff_file)
        segments = merge_segments(html_segments, sdlxliff_segments)
        print(f"Merged {len(html_segments)} HTML segments with {len(sdlxliff_segments)} SDLXLIFF segments")
    elif args.sdlxliff_file:
        print(f"Warning: SDLXLIFF file {args.sdlxliff_file} not found, using only HTML segments")

    # Create necessary directories
    inputs_dir = FACTGENIE_PATH / "data" / "inputs" / dataset_name
    outputs_dir = FACTGENIE_PATH / "data" / "outputs" / dataset_name
    campaigns_dir = FACTGENIE_PATH / "campaigns"

    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    campaigns_dir.mkdir(parents=True, exist_ok=True)

    # Determine segment size
    if args.segment_size is not None:
        segment_size = args.segment_size
    elif args.doc_level:
        segment_size = len(segments)
    else:
        segment_size = 1

    if segment_size > 1:
        # Multi-segment processing: create chunks of segments
        num_chunks = (len(segments) + segment_size - 1) // segment_size  # Ceiling division

        # Generate input.jsonl with chunked source text
        input_file = inputs_dir / f"{args.split}.jsonl"
        with open(input_file, "w", encoding="utf-8") as f:
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * segment_size
                end_idx = min(start_idx + segment_size, len(segments))

                # Concatenate source segments for this chunk
                source_text = "<br>".join(segments[i]["source_segment"] for i in range(start_idx, end_idx))
                # Create a segment_id representing the range of segments in this chunk
                if start_idx == end_idx - 1:
                    segment_id = segments[start_idx]["segment_id"]
                else:
                    segment_id = f"{segments[start_idx]['segment_id']}-{segments[end_idx-1]['segment_id']}"

                input_data = {"segment_id": segment_id, "source_text": source_text, "filename": filename}
                f.write(json.dumps(input_data, ensure_ascii=False) + "\n")

        # Generate output.jsonl with chunked target text
        output_file = outputs_dir / f"{args.split}-{setup_id}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * segment_size
                end_idx = min(start_idx + segment_size, len(segments))

                # Concatenate target segments for this chunk
                target_text = "\n".join(
                    extract_original_target(segments[i]["target_revision"]) for i in range(start_idx, end_idx)
                )
                output_data = {
                    "dataset": dataset_name,
                    "split": args.split,
                    "setup_id": setup_id,
                    "example_idx": chunk_idx,
                    "output": target_text,
                }
                f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

        # Create campaign folder in factgenie campaigns directory
        campaign_dir = campaigns_dir / args.campaign_id
        campaign_dir.mkdir(exist_ok=True)

        # Generate metadata.json
        metadata = {
            "id": args.campaign_id,
            "mode": "crowdsourcing",
            "config": {
                "annotators_per_example": 1,
                "annotation_granularity": "characters",
                "annotation_overlap_allowed": False,
                "service": "local",
                "annotation_span_categories": categories,
            },
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(campaign_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Create files subdirectory and generate annotations.jsonl
        files_dir = campaign_dir / "files"
        files_dir.mkdir(exist_ok=True)

        with open(files_dir / f"{dataset_name}-{args.split}.jsonl", "w", encoding="utf-8") as f:
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * segment_size
                end_idx = min(start_idx + segment_size, len(segments))

                # Extract chunk-level annotations
                chunk_annotations = extract_doc_level_annotations_for_chunk(segments, categories, start_idx, end_idx)

                annotation_data = {
                    "dataset": dataset_name,
                    "split": args.split,
                    "setup_id": setup_id,
                    "example_idx": chunk_idx,
                    "annotations": chunk_annotations,
                    "metadata": {"annotator_group": 0},
                }
                f.write(json.dumps(annotation_data, ensure_ascii=False) + "\n")

        # Generate db.csv based on existing JSONL files
        generate_db_csv_from_jsonl_files(campaign_dir)

        print(f"Multi-segment conversion completed successfully!")
        print(f"Generated files:")
        print(f"  - {input_file}")
        print(f"  - {output_file}")
        print(f"  - {campaign_dir}/")
        print(f"Dataset: {dataset_name}, Setup: {setup_id}, Split: {args.split}")
        print(f"Processed {len(segments)} segments in {num_chunks} chunks of size {segment_size}")

    else:
        # Segment-level processing (original behavior)
        # Generate input.jsonl in factgenie inputs directory
        input_file = inputs_dir / f"{args.split}.jsonl"
        with open(input_file, "w", encoding="utf-8") as f:
            for segment in segments:
                input_data = {
                    "segment_id": segment["segment_id"],
                    "source_text": segment["source_segment"],
                    "filename": filename,
                }
                f.write(json.dumps(input_data, ensure_ascii=False) + "\n")

        # Generate output.jsonl in factgenie outputs directory
        output_file = outputs_dir / f"{args.split}-{setup_id}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments):
                original_target = extract_original_target(segment["target_revision"])
                output_data = {
                    "dataset": dataset_name,
                    "split": args.split,
                    "setup_id": setup_id,
                    "example_idx": i,
                    "output": original_target,
                }
                f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

        # Create campaign folder in factgenie campaigns directory
        campaign_dir = campaigns_dir / args.campaign_id
        campaign_dir.mkdir(exist_ok=True)

        # Generate metadata.json
        metadata = {
            "id": args.campaign_id,
            "mode": "crowdsourcing",
            "config": {
                "annotators_per_example": 1,
                "annotation_granularity": "characters",
                "annotation_overlap_allowed": False,
                "service": "local",
                "annotation_span_categories": categories,
            },
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(campaign_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Create files subdirectory and generate annotations.jsonl
        files_dir = campaign_dir / "files"
        files_dir.mkdir(exist_ok=True)

        with open(files_dir / f"{dataset_name}-{args.split}.jsonl", "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments):
                annotations = extract_annotations(
                    segment["target_revision"], segment.get("error_codes", []), segment.get("comments", []), categories
                )

                annotation_data = {
                    "dataset": dataset_name,
                    "split": args.split,
                    "setup_id": setup_id,
                    "example_idx": i,
                    "annotations": annotations,
                    "metadata": {"annotator_group": 0},
                }
                f.write(json.dumps(annotation_data, ensure_ascii=False) + "\n")

        # Generate db.csv based on existing JSONL files
        generate_db_csv_from_jsonl_files(campaign_dir)

        print(f"Conversion completed successfully!")
        print(f"Generated files:")
        print(f"  - {input_file}")
        print(f"  - {output_file}")
        print(f"  - {campaign_dir}/")
        print(f"Dataset: {dataset_name}, Setup: {setup_id}, Split: {args.split}")
        print(f"Processed {len(segments)} segments")


if __name__ == "__main__":
    main()
