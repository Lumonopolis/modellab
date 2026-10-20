"""The labelling page: one gold set, served on 127.0.0.1 to a person answering its questions.

Standard library only. The page (modellab/page/) asks the task's questions about each item
and posts answers here; each answer is checked against the task and appended to the set's
labels.jsonl as a new revision under a file lock. A stale page cannot replace a newer answer.

Only this machine can reach the server. It answers only requests addressed to it by name
(the Host header), so a web page elsewhere cannot reach it through DNS rebinding, and it
accepts answers only with a header that a page from another site cannot send. The page may
load nothing from elsewhere and may not be framed; media are served sandboxed, under names
made with a secret this server keeps, so an SVG or HTML file cannot act as the page.
"""

import datetime
import difflib
import fcntl
import hashlib
import hmac
import json
import mimetypes
import random
import re
import secrets
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from . import gold

PAGE = Path(__file__).parent / "page"
_RANGE = re.compile(r"bytes=(\d*)-(\d*)$")
PAGE_POLICY = ("default-src 'self'; img-src 'self'; media-src 'self'; object-src 'none'; base-uri 'none'; "
               "form-action 'none'; frame-ancestors 'none'")
MEDIA_POLICY = "sandbox; default-src 'none'"


class LabelServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, folder, labeller, port=0):
        self.folder = Path(folder)
        self.labeller = labeller
        self.task = gold.load_task(self.folder)
        self.items = gold.read_items(self.folder / "items.jsonl")
        self.by_id = {item["id"]: item for item in self.items}
        self.secret = secrets.token_bytes(32)  # media names cannot be guessed from their paths
        self.media = {}  # token -> file, for every file an item names
        for item in self.items:
            for value in item.values():
                if (path := gold.media_path(self.folder, value)) is not None:
                    self.media[self.token(path)] = path
        self.plugins = {}  # published path -> file, for plugin page parts only
        for kind in self.task["plugins"].get("kinds", {}).values():
            self.plugins[kind["page"]] = self.folder / kind["page"]
        for file in self.task["plugins"].get("displays", {}).values():
            self.plugins[file] = self.folder / file
        super().__init__(("127.0.0.1", port), Handler)
        self.hosts = {f"127.0.0.1:{self.server_port}", f"localhost:{self.server_port}"}

    # What the page asks for

    def task_view(self):
        questions = [{key: value for key, value in question.items() if key != "plugin_check"}
                     for question in self.task["questions"]]
        instructions = None
        if self.task["instructions"]:
            instructions = (self.folder / self.task["instructions"]).read_text()
        return {**self.task, "questions": questions, "instructions": instructions, "labeller": self.labeller,
                "plugins": sorted(self.plugins)}

    def queue(self):
        """This labeller's items, in the task's order, each with the latest answer given here."""
        newest = gold.latest(gold.read_labels(self.folder / "labels.jsonl"))
        queue = self.task["queue"]
        wanted = queue["labellers"]
        items = []
        for item in self.items:
            mine = newest.get((item["id"], self.labeller))
            others = sum(1 for (item_id, who), record in newest.items()
                         if item_id == item["id"] and who != self.labeller and not record.get("skip"))
            if mine is None and others >= wanted:
                continue  # enough other people answered it already
            items.append(self.item_view(item, mine))
        order = queue["order"]
        if order == "random":
            random.Random(f"{queue['seed']}:{self.labeller}").shuffle(items)
        elif order.startswith("field:"):
            field = order.split(":", 1)[1]
            items.sort(key=lambda view: (str(self.by_id[view["id"]].get(field, "")), view["id"]))
        return items

    def item_view(self, item, record):
        blind = self.task["queue"]["blind"]
        fields = {}
        for pane in self.task["panes"]:
            if blind and pane["target"]:
                continue  # never sent, so it cannot be seen before answering
            value = item.get(pane["field"])
            path = gold.media_path(self.folder, value)
            if path is not None:
                fields[pane["field"]] = {"url": f"/media/{self.token(path)}{path.suffix.lower()}"}
            elif pane["show"] in ("image", "video", "audio"):
                fields[pane["field"]] = None  # only files the set names; never an address from elsewhere
            else:
                fields[pane["field"]] = value
            if pane["show"] == "diff":
                fields[pane["field"]] = {"diff": _diff(item.get(pane["against"]), value)}
        shuffle = self.task["queue"]["shuffle"]
        order = [pane["field"] for pane in self.task["panes"] if not (blind and pane["target"])]
        if shuffle:
            mixed = [field for field in shuffle if field in order]  # never a target that blind mode withholds
            random.Random(f"{item['id']}:{self.labeller}").shuffle(mixed)
            slots = iter(mixed)
            order = [next(slots) if field in shuffle else field for field in order]
        return {"id": item["id"], "fields": fields, "order": order,
                "revision": record["revision"] if record else 0,
                "answer": None if record is None else {"skip": True} if record.get("skip") else record["answers"]}

    def token(self, path):
        return hmac.new(self.secret, str(Path(path).resolve()).encode(), hashlib.sha256).hexdigest()[:32]

    def save(self, item_id, body):
        """Append an answer. (status, reply)."""
        if item_id not in self.by_id:
            return HTTPStatus.NOT_FOUND, {"error": f"There is no item {item_id!r}."}
        if not isinstance(body, dict) or not isinstance(body.get("revision"), int):
            return HTTPStatus.BAD_REQUEST, {"error": "The answer must name the revision it replaces."}
        record = {"item": item_id, "labeller": self.labeller}
        if body.get("skip") is True:
            record["skip"] = True
        else:
            answers = body.get("answers")
            if not isinstance(answers, dict):
                return HTTPStatus.BAD_REQUEST, {"error": "The answer must hold answers or a skip."}
            if problems := gold.check_answers(self.task, answers, self.by_id[item_id]):
                return HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "The answer does not fit the task: " + "; ".join(problems)}
            record["answers"] = answers
        if isinstance(body.get("order"), list):
            record["order"] = [str(field) for field in body["order"]]
        path = self.folder / "labels.jsonl"
        with open(path, "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            current = gold.latest(gold.read_labels(path)).get((item_id, self.labeller))
            revision = current["revision"] if current else 0
            if body["revision"] != revision:
                return HTTPStatus.CONFLICT, {"error": "This item was answered elsewhere since this page loaded it.",
                                             "revision": revision}
            record = {**record, "revision": revision + 1,
                      "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        return HTTPStatus.OK, {"revision": record["revision"]}


class Handler(BaseHTTPRequestHandler):
    server: LabelServer

    def log_message(self, format, *args):  # quiet; errors reach the page
        pass

    def allowed(self):
        if self.headers.get("Host") not in self.server.hosts:
            self.reply(HTTPStatus.FORBIDDEN, {"error": "This server answers only on 127.0.0.1."})
            return False
        return True

    def do_GET(self):
        if self.allowed():
            self.answering(self.get)

    def do_POST(self):
        if self.allowed():
            self.answering(self.post)

    def answering(self, handle):
        """Run a handler; a failure is answered with its reason, and logged, not dropped."""
        try:
            handle()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser left, as it does when a video seeks elsewhere
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            self.reply(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"The server failed: {type(error).__name__}: {error}"})

    def get(self):
        path = unquote(self.path.split("?", 1)[0])
        if path in ("/", "/index.html"):
            return self.file(PAGE / "label.html")
        if path.startswith("/page/") and "/" not in path[6:] and ".." not in path[6:] and (PAGE / path[6:]).is_file():
            return self.file(PAGE / path[6:])
        if path.startswith("/plugins/") and path[9:] in self.server.plugins:
            return self.file(self.server.plugins[path[9:]])
        if path.startswith("/media/"):
            token = path[7:].split(".", 1)[0]
            if token in self.server.media:
                return self.file(self.server.media[token], ranged=True)
        if path == "/api/task":
            return self.reply(HTTPStatus.OK, self.server.task_view())
        if path == "/api/items":
            return self.reply(HTTPStatus.OK, {"items": self.server.queue()})
        self.reply(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def post(self):
        if self.headers.get("X-ModelLab") != "1" or self.headers.get("Content-Type") != "application/json":
            return self.reply(HTTPStatus.FORBIDDEN, {"error": "Answers come only from the ModelLab page."})
        match = re.fullmatch(r"/api/answer/([^/]+)", self.path)
        if not match:
            return self.reply(HTTPStatus.NOT_FOUND, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except (ValueError, json.JSONDecodeError):
            return self.reply(HTTPStatus.BAD_REQUEST, {"error": "The answer is not JSON."})
        status, reply = self.server.save(match.group(1), body)
        self.reply(status, reply)

    def reply(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.guard_headers(PAGE_POLICY)
        self.end_headers()
        self.wfile.write(body)

    def guard_headers(self, policy):
        self.send_header("Content-Security-Policy", policy)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")

    def file(self, path, ranged=False):
        size = path.stat().st_size
        start, end = 0, size - 1
        match = _RANGE.match(self.headers.get("Range", "")) if ranged else None
        if match and (match.group(1) or match.group(2)):
            if match.group(1):
                start, end = int(match.group(1)), int(match.group(2)) if match.group(2) else size - 1
            else:
                start = max(0, size - int(match.group(2)))
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if match and (match.group(1) or match.group(2)) else HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if match and (match.group(1) or match.group(2)):
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "private, max-age=3600" if ranged else "no-store")  # media only; a plugin may change
        self.guard_headers(PAGE_POLICY if ranged is False else MEDIA_POLICY)
        self.end_headers()
        with open(path, "rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0 and (chunk := handle.read(min(2**20, remaining))):
                self.wfile.write(chunk)
                remaining -= len(chunk)


def _diff(before, after):
    """Lines as [["=", line] | ["-", line] | ["+", line]], from the `against` field to this one."""
    old, new = str(before or "").splitlines(), str(after or "").splitlines()
    lines = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if tag == "equal":
            lines += [["=", line] for line in old[i1:i2]]
        else:
            lines += [["-", line] for line in old[i1:i2]] + [["+", line] for line in new[j1:j2]]
    return lines


def serve(folder, labeller, port=0, ready=None):
    """Serve until interrupted. `ready(url)` is called once the server listens."""
    server = LabelServer(folder, labeller, port)
    url = f"http://127.0.0.1:{server.server_port}/"
    if ready:
        ready(url)
    try:
        server.serve_forever()
    finally:
        server.server_close()
