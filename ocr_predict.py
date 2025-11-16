#!/usr/bin/env python3
"""
Batch inference of OCR necessity for PDFs using the model from construct_data.ipynb.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pymupdf  # PyMuPDF
from xgboost import XGBClassifier


def flatten_per_page_features(feature_dict_sample: dict, sample_to_k_page_features: int = 8) -> dict:
    """Flatten list-based per-page features into scalar columns."""
    flattened_features: dict[str, float] = {}
    doc_level_features = [
        "num_pages_successfully_sampled",
        "num_unique_image_xrefs",
        "num_junk_image_xrefs",
        "garbled_text_ratio",
        "is_form",
        "creator_or_producer_is_known_scanner",
        "class",
    ]

    for key in doc_level_features:
        if key in feature_dict_sample:
            flattened_features[key] = feature_dict_sample[key]

    page_level_features = [
        "page_level_unique_font_counts",
        "page_level_char_counts",
        "page_level_text_box_counts",
        "page_level_avg_text_box_lengths",
        "page_level_text_area_ratios",
        "page_level_hidden_char_counts",
        "page_level_hidden_text_box_counts",
        "page_level_hidden_avg_text_box_lengths",
        "page_level_hidden_text_area_ratios",
        "page_level_image_counts",
        "page_level_non_junk_image_counts",
        "page_level_bitmap_proportions",
        "page_level_max_merged_strip_areas",
        "page_level_drawing_strokes_count",
        "page_level_vector_graphics_obj_count",
    ]

    counts = feature_dict_sample.get("page_level_unique_font_counts", [])
    num_pages = len(counts)
    if num_pages == 0:
        return flattened_features

    page_indices = list(range(num_pages))
    if num_pages < sample_to_k_page_features:
        extra = np.random.choice(
            num_pages, sample_to_k_page_features - num_pages, replace=True).tolist()
        page_indices += extra

    for key in page_level_features:
        list_data = feature_dict_sample.get(key, [])
        if not list_data:
            continue
        for page_idx, ind in enumerate(page_indices):
            flattened_features[f"{key}_page{page_idx + 1}"] = list_data[ind]

    return flattened_features


class PDFFeatureExtractor:
    JUNK_IMAGE_THRESHOLD_RATIO = 0.5
    JUNK_IMAGE_MIN_PAGES_FOR_THRESHOLD = 3
    MERGE_MAX_OFFSET = 5
    MERGE_MAX_GAP = 2
    KNOWN_SCANNER_STRINGS = [
        "scanner",
        "scan",
        "epson",
        "hp scanjet",
        "canon",
        "fujitsu",
        "kodak",
        "brother",
        "xerox",
        "lexmark",
        "kmc",
        "kofax",
        "ricoh",
        "iris",
        "capturedocument",
        "paperport",
        "readiris",
        "simpleocr",
    ]

    def __init__(self, num_pages_to_sample: int = 5, num_chunks: int = 1):
        if not isinstance(num_pages_to_sample, int):
            raise ValueError("num_pages_to_sample must be an integer")
        self.num_pages_to_sample = num_pages_to_sample
        self.num_chunks = num_chunks

    def get_garbled_text_per_page(self, doc: pymupdf.Document) -> tuple[list[int], list[int]]:
        all_text = []
        garbled_text = []
        for page in doc:
            page_text = page.get_text(
                "text", flags=pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP
            )
            all_text.append(len(page_text))
            garbled_text.append(page_text.count(chr(0xFFFD)))
        return all_text, garbled_text

    def _get_sampled_page_indices(self, doc: pymupdf.Document) -> list[list[int]]:
        total_pages = len(doc)
        if total_pages == 0 or self.num_pages_to_sample <= 0:
            return []

        available_indices = list(range(total_pages))
        sampled_indices = []
        num_chunks = (
            len(available_indices) // self.num_pages_to_sample +
            1 if self.num_chunks == -1 else self.num_chunks
        )

        for _ in range(num_chunks):
            if not available_indices:
                break
            chunk_size = min(self.num_pages_to_sample, len(available_indices))
            chunk = random.sample(available_indices, chunk_size)
            for idx in chunk:
                available_indices.remove(idx)
            sampled_indices.append(sorted(chunk))
        return sampled_indices

    def _heuristic_merge_image_strips_on_page(
        self, single_page_image_list: list, page_width: float, page_height: float
    ) -> list:
        if not single_page_image_list:
            return []

        deduped = []
        dedup_bboxes = set()
        for img_data in single_page_image_list:
            bbox_tuple = (img_data[0], img_data[1], img_data[2], img_data[3])
            if bbox_tuple not in dedup_bboxes:
                dedup_bboxes.add(bbox_tuple)
                deduped.append(img_data)

        if not deduped:
            return []

        deduped.sort(key=lambda img: (img[1], img[0]))
        merged = [deduped[0]]

        for img_to_merge in deduped[1:]:
            x0_curr, y0_curr, x1_curr, y1_curr, imgid_curr = img_to_merge
            x0_last, y0_last, x1_last, y1_last, _ = merged[-1]
            img_curr_width = abs(x1_curr - x0_curr)
            img_curr_height = abs(y1_curr - y0_curr)
            full_width_curr = page_width > 0 and (
                img_curr_width >= page_width * 0.9)
            full_height_curr = page_height > 0 and (
                img_curr_height >= page_height * 0.9)

            can_merge = False
            if full_width_curr:
                aligned = (
                    abs(x0_last - x0_curr) <= self.MERGE_MAX_OFFSET
                    and abs(x1_last - x1_curr) <= self.MERGE_MAX_OFFSET
                    and abs(y0_curr - y1_last) <= self.MERGE_MAX_GAP
                )
                can_merge = aligned
            if not can_merge and full_height_curr:
                aligned = (
                    abs(y0_last - y0_curr) <= self.MERGE_MAX_OFFSET
                    and abs(y1_last - y1_curr) <= self.MERGE_MAX_OFFSET
                    and abs(x0_curr - x1_last) <= self.MERGE_MAX_GAP
                )
                can_merge = aligned

            if can_merge:
                merged[-1] = [
                    min(x0_curr, x0_last),
                    min(y0_curr, y0_last),
                    max(x1_curr, x1_last),
                    max(y1_curr, y1_last),
                    imgid_curr,
                ]
            else:
                merged.append(img_to_merge)
        return merged

    def _extract_document_level_stats_from_sampled_pages(
        self, doc: pymupdf.Document, sampled_page_indices: list[int]
    ) -> dict:
        doc_stats = {"junk_image_xrefs_list": []}
        if not sampled_page_indices:
            return doc_stats

        img_xrefs = []
        page_unique_xrefs = {}
        for page_idx in sampled_page_indices:
            try:
                page = doc.load_page(page_idx)
            except Exception:
                logging.warning("Failed to load page %s", page_idx)
                page_unique_xrefs[page_idx] = set()
                continue
            current_page_unique_xrefs = set()
            for img_def in page.get_images(full=False):
                xref = img_def[0]
                if xref == 0:
                    continue
                current_page_unique_xrefs.add(xref)
                img_xrefs.append(xref)
            page_unique_xrefs[page_idx] = current_page_unique_xrefs

        if not img_xrefs:
            return doc_stats

        doc_stats["num_unique_image_xrefs"] = len(set(img_xrefs))
        xref_counts = Counter()
        for page_xrefs in page_unique_xrefs.values():
            xref_counts.update(page_xrefs)

        num_sampled_pages = len(sampled_page_indices)
        min_page_occurrence_threshold = min(
            max(num_sampled_pages * self.JUNK_IMAGE_THRESHOLD_RATIO,
                float(self.JUNK_IMAGE_MIN_PAGES_FOR_THRESHOLD)),
            num_sampled_pages,
        )

        junk_xrefs = [
            xref
            for xref, count in xref_counts.items()
            if count >= min_page_occurrence_threshold and num_sampled_pages >= self.JUNK_IMAGE_MIN_PAGES_FOR_THRESHOLD
        ]
        doc_stats["num_junk_image_xrefs"] = len(junk_xrefs)
        doc_stats["junk_image_xrefs_list"] = junk_xrefs if num_sampled_pages >= self.JUNK_IMAGE_MIN_PAGES_FOR_THRESHOLD else []
        return doc_stats

    def _check_creator_producer_scanner(self, doc: pymupdf.Document) -> bool:
        metadata = doc.metadata
        creator = metadata.get("creator", "").lower()
        producer = metadata.get("producer", "").lower()
        return any(scanner in creator or scanner in producer for scanner in self.KNOWN_SCANNER_STRINGS)

    def extract_all_features(self, doc: pymupdf.Document) -> list[dict]:
        sampled_page_indices = self._get_sampled_page_indices(doc)
        return [self.compute_features_per_chunk(doc, indices) for indices in sampled_page_indices]

    def compute_features_per_chunk(self, doc: pymupdf.Document, sampled_page_indices: list[int]) -> dict:
        features = {
            "is_form": False,
            "creator_or_producer_is_known_scanner": False,
            "garbled_text_ratio": 0,
            "page_level_unique_font_counts": [],
            "page_level_char_counts": [],
            "page_level_text_box_counts": [],
            "page_level_avg_text_box_lengths": [],
            "page_level_text_area_ratios": [],
            "page_level_hidden_char_counts": [],
            "page_level_hidden_text_box_counts": [],
            "page_level_hidden_avg_text_box_lengths": [],
            "page_level_hidden_text_area_ratios": [],
            "page_level_image_counts": [],
            "page_level_non_junk_image_counts": [],
            "page_level_bitmap_proportions": [],
            "page_level_max_merged_strip_areas": [],
            "page_level_drawing_strokes_count": [],
            "page_level_vector_graphics_obj_count": [],
            "num_pages_successfully_sampled": 0,
            "num_pages_requested_for_sampling": len(sampled_page_indices),
            "sampled_page_indices": [],
        }

        if not sampled_page_indices:
            return features

        doc_level_stats = self._extract_document_level_stats_from_sampled_pages(
            doc, sampled_page_indices)
        junk_image_xrefs = set(
            doc_level_stats.get("junk_image_xrefs_list", []))
        features.update(
            {k: v for k, v in doc_level_stats.items() if not k.endswith("_list")})

        is_form_pdf = getattr(doc, "is_form_pdf", False)
        features["is_form"] = bool(
            is_form_pdf) if is_form_pdf is not None else False
        features["creator_or_producer_is_known_scanner"] = self._check_creator_producer_scanner(
            doc)

        all_text, garbled_text = self.get_garbled_text_per_page(doc)
        features["global_garbled_text_ratio"] = (
            sum(garbled_text) / sum(all_text) if sum(all_text) else 0
        )
        sampled_all_text = sum(all_text[i] for i in sampled_page_indices)
        sampled_garbled = sum(garbled_text[i] for i in sampled_page_indices)
        features["garbled_text_ratio"] = sampled_garbled / \
            sampled_all_text if sampled_all_text else 0

        for page_idx in sampled_page_indices:
            try:
                page = doc.load_page(page_idx)
            except Exception:
                logging.warning("Failed to load page %s", page_idx)
                continue

            features["sampled_page_indices"].append(page_idx)
            features["num_pages_successfully_sampled"] += 1
            page_rect = page.rect
            page_area_pts = float(page_rect.width * page_rect.height) or 1.0

            current_page_fonts = set()
            try:
                for fi in page.get_fonts(full=True):
                    if len(fi) > 3 and fi[3]:
                        current_page_fonts.add(fi[3])
            except Exception:
                logging.warning("Failed to read fonts for page %s", page_idx)
            features["page_level_unique_font_counts"].append(
                len(current_page_fonts))

            page_char_count = 0
            page_text_total_area = 0
            page_text_box_count = 0
            page_hidden_total_area = 0
            page_hidden_char_count = 0
            page_hidden_box_count = 0

            for trace in page.get_texttrace():
                chars = trace.get("chars", [])
                n_chars = len([c[0] for c in chars])
                bbox = trace.get("bbox", (0, 0, 0, 0))
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if trace.get("type") == 3 or trace.get("opacity", 1.0) == 0:
                    page_hidden_char_count += n_chars
                    page_hidden_total_area += area
                    page_hidden_box_count += 1
                else:
                    page_char_count += n_chars
                    page_text_total_area += area
                    page_text_box_count += 1

            features["page_level_char_counts"].append(page_char_count)
            features["page_level_text_box_counts"].append(page_text_box_count)
            features["page_level_avg_text_box_lengths"].append(
                page_text_total_area / page_text_box_count if page_text_box_count else 0.0
            )
            features["page_level_text_area_ratios"].append(
                page_text_total_area / page_area_pts)

            features["page_level_hidden_char_counts"].append(
                page_hidden_char_count)
            features["page_level_hidden_text_box_counts"].append(
                page_hidden_box_count)
            features["page_level_hidden_avg_text_box_lengths"].append(
                page_hidden_total_area / page_hidden_box_count if page_hidden_box_count else 0.0
            )
            features["page_level_hidden_text_area_ratios"].append(
                page_hidden_total_area / page_area_pts)

            page_total_image_instances = 0
            page_non_junk_image_instances = 0
            non_junk_rects = []
            try:
                for img_def in page.get_images(full=False):
                    xref = img_def[0]
                    if xref == 0:
                        continue
                    rects = page.get_image_rects(xref, transform=False)
                    page_total_image_instances += len(rects)
                    if xref not in junk_image_xrefs:
                        page_non_junk_image_instances += len(rects)
                        for rect in rects:
                            if rect.is_empty or rect.is_infinite:
                                continue
                            non_junk_rects.append(
                                [rect.x0, rect.y0, rect.x1, rect.y1, xref])
            except Exception:
                logging.warning("Failed to read images for page %s", page_idx)

            features["page_level_image_counts"].append(
                page_total_image_instances)
            features["page_level_non_junk_image_counts"].append(
                page_non_junk_image_instances)

            merged_strip_bboxes = self._heuristic_merge_image_strips_on_page(
                non_junk_rects, page_rect.width, page_rect.height
            )
            merged_strip_areas = [abs(b[2] - b[0]) * abs(b[3] - b[1])
                                  for b in merged_strip_bboxes]
            if merged_strip_areas:
                features["page_level_max_merged_strip_areas"].append(
                    max(merged_strip_areas) / page_area_pts)
                features["page_level_bitmap_proportions"].append(
                    sum(merged_strip_areas) / page_area_pts)
            else:
                features["page_level_max_merged_strip_areas"].append(0.0)
                features["page_level_bitmap_proportions"].append(0.0)

            drawings_stroke_count = 0
            vector_graphics_obj_count = 0
            try:
                drawings = page.get_cdrawings()
                vector_graphics_obj_count = len(drawings)
                for path in drawings:
                    for item in path.get("items", []):
                        if item[0] in ["l", "c", "q"]:
                            drawings_stroke_count += 1
                    if (path.get("rect") or path.get("quad")) and path.get("stroke_opacity", 1) > 0 and path.get("color"):
                        drawings_stroke_count += 1
            except Exception:
                logging.warning(
                    "Failed to read drawings for page %s", page_idx)
            features["page_level_drawing_strokes_count"].append(
                drawings_stroke_count)
            features["page_level_vector_graphics_obj_count"].append(
                vector_graphics_obj_count)
        return features

    def extract_features(self, doc: pymupdf.Document) -> list[dict]:
        return self.extract_all_features(doc)


def discover_pdf_paths(raw_inputs: Iterable[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for raw in raw_inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            logging.warning("Input %s does not exist; skipping", raw)
            continue
        if path.is_file():
            candidates = [path]
        else:
            candidates = path.rglob("*.pdf")
        for candidate in candidates:
            if candidate.suffix.lower() != ".pdf":
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def predict_for_pdf(
    pdf_path: Path,
    extractor: PDFFeatureExtractor,
    model: XGBClassifier,
    sample_to_k_page_features: int,
    threshold: float,
) -> dict:
    with pymupdf.open(pdf_path) as doc:
        num_pages = len(doc)
        is_encrypted = bool(doc.is_encrypted)
        needs_password = bool(doc.needs_pass)
        features_raw = extractor.extract_features(doc)
    if not features_raw:
        raise ValueError("No features were extracted from sampled pages")

    flattened = flatten_per_page_features(
        features_raw[0], sample_to_k_page_features)
    features_df = pd.DataFrame([flattened])
    ocr_prob = float(model.predict_proba(features_df)[0][1])
    return {
        "path": str(pdf_path),
        "ocr_probability": ocr_prob,
        "needs_ocr": ocr_prob >= threshold,
        "is_form": bool(flattened.get("is_form", False)),
        "garbled_text_ratio": float(flattened.get("garbled_text_ratio", 0.0)),
        "num_pages_successfully_sampled": int(flattened.get("num_pages_successfully_sampled", 0)),
        "num_pages": num_pages,
        "is_encrypted": is_encrypted,
        "needs_password": needs_password,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict whether PDFs need OCR using xgb_ocr_classifier/xgb_classifier.ubj."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="PDF files or directories to scan (directories are searched recursively).",
    )
    parser.add_argument(
        "--model-path",
        default="xgb_ocr_classifier/xgb_classifier.ubj",
        help="Path to the trained XGBoost model file.",
    )
    parser.add_argument(
        "--output",
        default="ocr_predictions.csv",
        help="CSV file to write predictions to.",
    )
    parser.add_argument(
        "--failures-output",
        default=None,
        help="Optional CSV path to record files that could not be processed.",
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=8,
        help="How many pages to sample inside each chunk when building features.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help="How many disjoint page chunks to sample per document.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold to flag a PDF as needing OCR.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for page sampling and numpy resampling.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the number of PDFs to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Emit a progress log every N documents.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show warnings/errors in the console.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    extractor = PDFFeatureExtractor(
        num_pages_to_sample=args.sample_pages, num_chunks=args.num_chunks)
    model = XGBClassifier()
    model.load_model(args.model_path)

    pdf_paths = list(discover_pdf_paths(args.inputs))
    if args.max_files:
        pdf_paths = pdf_paths[: args.max_files]
    logging.info("Found %d PDFs to score.", len(pdf_paths))

    results: list[dict] = []
    failures: list[dict] = []
    for idx, pdf_path in enumerate(pdf_paths, start=1):
        try:
            result = predict_for_pdf(
                pdf_path=pdf_path,
                extractor=extractor,
                model=model,
                sample_to_k_page_features=args.sample_pages,
                threshold=args.threshold,
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001 - we want to keep going
            logging.exception("Failed to score %s", pdf_path)
            failures.append({"path": str(pdf_path), "error": str(exc)})
        if idx % max(1, args.log_every) == 0:
            logging.info("Processed %d/%d PDFs", idx, len(pdf_paths))

    if results:
        pd.DataFrame(results).to_csv(args.output, index=False)
        logging.info("Wrote %d predictions to %s", len(results), args.output)

    if failures:
        fail_path = args.failures_output or Path(
            args.output).with_suffix(".failures.csv")
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        logging.info("Recorded %d failures to %s", len(failures), fail_path)


if __name__ == "__main__":
    main()
