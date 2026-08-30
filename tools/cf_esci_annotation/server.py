from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "cf_esci" / "human_validation_v1"
COMPLETED_DIR = DATA_DIR / "completed"
STATIC_DIR = Path(__file__).resolve().parent

QUERY_ANSWERS = (
    "core_intent_preserved", "single_requirement_change",
    "changed_requirement_correct", "requirement_family", "operation",
    "query_naturalness", "semantic_plausibility", "confidence",
    "old_value_normalized", "new_value_normalized",
)
DIRECTION_ANSWERS = ("candidate_direction", "direction_confidence")

MODE_CONFIG = {
    "ANNOTATOR_A_QUERY": ("annotator_A_query_template.csv", "annotator_A_query_annotations.csv", "example_id", QUERY_ANSWERS, "A", "FINAL"),
    "ANNOTATOR_B_QUERY": ("annotator_B_query_template.csv", "annotator_B_query_annotations.csv", "example_id", QUERY_ANSWERS, "B", "FINAL"),
    "ANNOTATOR_A_DIRECTION": ("annotator_A_direction_template.csv", "annotator_A_direction_annotations.csv", "direction_item_id", DIRECTION_ANSWERS, "A", "FINAL"),
    "ANNOTATOR_B_DIRECTION": ("annotator_B_direction_template.csv", "annotator_B_direction_annotations.csv", "direction_item_id", DIRECTION_ANSWERS, "B", "FINAL"),
    "CALIBRATION_A_QUERY": ("calibration_query_pairs.csv", "calibration_annotations_A.csv", "example_id", QUERY_ANSWERS, "A", "CALIBRATION_ONLY"),
    "CALIBRATION_B_QUERY": ("calibration_query_pairs.csv", "calibration_annotations_B.csv", "example_id", QUERY_ANSWERS, "B", "CALIBRATION_ONLY"),
    "CALIBRATION_A_DIRECTION": ("calibration_candidate_pairs.csv", "calibration_direction_annotations_A.csv", "direction_item_id", DIRECTION_ANSWERS, "A", "CALIBRATION_ONLY"),
    "CALIBRATION_B_DIRECTION": ("calibration_candidate_pairs.csv", "calibration_direction_annotations_B.csv", "direction_item_id", DIRECTION_ANSWERS, "B", "CALIBRATION_ONLY"),
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def config_for(mode: str):
    if mode not in MODE_CONFIG:
        raise ValueError("Unsupported annotation mode")
    return MODE_CONFIG[mode]


def load_mode(mode: str, data_dir: Path = DATA_DIR, completed_dir: Path = COMPLETED_DIR) -> dict:
    source_name, output_name, key, answer_fields, annotator, phase = config_for(mode)
    fields, source_rows = read_csv(data_dir / source_name)
    output_path = completed_dir / output_name
    if output_path.exists():
        saved_fields, saved_rows = read_csv(output_path)
        if saved_fields != fields or [r[key] for r in saved_rows] != [r[key] for r in source_rows]:
            raise ValueError("Saved annotation file is incompatible with the frozen source")
        immutable = [f for f in fields if f not in answer_fields and f != "annotator_id"]
        for source, saved in zip(source_rows, saved_rows):
            if any(source[f] != saved[f] for f in immutable):
                raise ValueError("Saved annotation file changes immutable source data")
        rows = saved_rows
    else:
        rows = source_rows
    for row in rows:
        if "annotator_id" in row:
            row["annotator_id"] = annotator
        row["_annotation_phase"] = phase
    return {"mode": mode, "key": key, "answer_fields": list(answer_fields), "rows": rows, "output": output_name, "phase": phase}


def atomic_write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def save_mode(mode: str, submitted: list[dict], data_dir: Path = DATA_DIR, completed_dir: Path = COMPLETED_DIR) -> dict:
    source_name, output_name, key, answer_fields, annotator, _phase = config_for(mode)
    fields, source_rows = read_csv(data_dir / source_name)
    if len(submitted) != len(source_rows) or [str(r.get(key, "")) for r in submitted] != [r[key] for r in source_rows]:
        raise ValueError("Submitted rows do not preserve frozen count and ordering")
    allowed = set(answer_fields)
    merged = []
    for source, answers in zip(source_rows, submitted):
        row = dict(source)
        if "annotator_id" in row: row["annotator_id"] = annotator
        for field in answer_fields:
            row[field] = str(answers.get(field, ""))
        unexpected = set(answers) - allowed - {key}
        if unexpected: raise ValueError("Submission contains non-answer fields")
        merged.append(row)
    path = completed_dir / output_name
    atomic_write_csv(path, fields, merged)
    completed = sum(all(row.get(field, "") for field in answer_fields) for row in merged)
    return {"saved": True, "path": str(path), "row_count": len(merged), "completed": completed}


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        relative = urlparse(path).path.lstrip("/") or "index.html"
        return str(STATIC_DIR / relative)

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/modes":
            return self.send_json({"modes": list(MODE_CONFIG)})
        if parsed.path == "/api/data":
            try: return self.send_json(load_mode(parse_qs(parsed.query).get("mode", [""])[0]))
            except (ValueError, OSError) as exc: return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/save": return self.send_error(HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length))
            return self.send_json(save_mode(payload["mode"], payload["rows"]))
        except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
            return self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main():
    parser = argparse.ArgumentParser(description="Local CF-ESCI human annotation interface")
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(); server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"CF-ESCI annotation interface: http://{args.host}:{args.port}")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == "__main__": main()
