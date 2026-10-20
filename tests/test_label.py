"""The labelling server: what it sends, what it accepts, and what it refuses. The page itself
is checked by hand in a browser, on desktop and phone."""

import http.client
import json
import subprocess
import sys
import threading

import pytest

from conftest import CLI
from modellab import gold, label

TASK = '''
title = "Pairs"

[[pane]]
field = "a"
show = "image"

[[pane]]
field = "b"
show = "image"

[[pane]]
field = "truth"
show = "text"
target = true

[[question]]
id = "better"
kind = "rank"
panes = ["a", "b"]

[[question]]
id = "sure"
kind = "scale"
range = [1, 3]

[queue]
order = "random"
blind = true
shuffle = ["a", "b"]
labellers = 1
'''


@pytest.fixture
def served(tmp_path):
    folder = tmp_path / "set"
    (folder / "src").mkdir(parents=True)
    (folder / "task.toml").write_text(TASK)
    (folder / "src" / "clip.bin").write_bytes(bytes(range(256)) * 4)
    items = [{"id": f"i{n}", "a": {"file": "src/clip.bin"}, "b": {"file": "src/clip.bin"}, "truth": "left"}
             for n in range(6)]
    (folder / "items.jsonl").write_text("".join(json.dumps(item) + "\n" for item in items))
    servers = []

    def start(labeller="ana", task=None):
        if task:
            (folder / "task.toml").write_text(task)
        server = label.LabelServer(folder, labeller)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server
    yield folder, start
    for server in servers:
        server.shutdown()
        server.server_close()


def call(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
    sent = {"Host": f"127.0.0.1:{server.server_port}", **(headers or {})}
    if body is not None:
        sent = {"Content-Type": "application/json", "X-ModelLab": "1", **sent}
    connection.request(method, path, json.dumps(body) if body is not None else None, sent)
    response = connection.getresponse()
    data = response.read()
    return response.status, response.headers, data


def answer(server, item, revision, answers):
    status, _, data = call(server, "POST", f"/api/answer/{item}", {"revision": revision, "answers": answers})
    return status, json.loads(data)


def test_blind_mode_never_sends_the_target_and_shuffles_the_rest(served):
    _, start = served
    server = start()
    status, _, data = call(server, "GET", "/api/items")
    items = json.loads(data)["items"]
    assert status == 200 and len(items) == 6
    assert all("truth" not in item["fields"] and set(item["order"]) == {"a", "b"} for item in items)
    assert len({tuple(item["order"]) for item in items}) == 2  # both orders occur across six items
    again = json.loads(call(server, "GET", "/api/items")[2])["items"]
    assert [(i["id"], i["order"]) for i in again] == [(i["id"], i["order"]) for i in items]  # stable per labeller


def test_answers_are_checked_versioned_and_never_overwrite_a_newer_one(served):
    folder, start = served
    server = start()
    assert answer(server, "i0", 0, {"better": [["a"], ["b"]], "sure": 2}) == (200, {"revision": 1})
    status, reply = answer(server, "i0", 0, {"better": [["b"], ["a"]], "sure": 3})
    assert status == 409 and reply["revision"] == 1 and "answered elsewhere" in reply["error"]
    assert answer(server, "i0", 1, {"better": [["b"], ["a"]], "sure": 3}) == (200, {"revision": 2})
    status, reply = answer(server, "i1", 0, {"better": [["a", "b"]], "sure": 7})
    assert status == 422 and "ties are not allowed" in reply["error"] and "from 1 to 3" in reply["error"]
    status, _, _ = call(server, "POST", "/api/answer/i2", {"revision": 0, "skip": True})
    assert status == 200
    assert call(server, "POST", "/api/answer/nope", {"revision": 0, "skip": True})[0] == 404
    records = gold.read_labels(folder / "labels.jsonl")
    assert [(r["item"], r["revision"]) for r in records] == [("i0", 1), ("i0", 2), ("i2", 1)]
    assert gold.check(folder) == []


def test_enough_answers_from_others_take_an_item_out_of_the_queue(served):
    _, start = served
    ana = start("ana")
    answer(ana, "i0", 0, {"better": [["a"], ["b"]], "sure": 1})
    ben = start("ben")
    ids = [item["id"] for item in json.loads(call(ben, "GET", "/api/items")[2])["items"]]
    assert "i0" not in ids and len(ids) == 5  # labellers = 1: ana's answer is enough
    mine = json.loads(call(ana, "GET", "/api/items")[2])["items"]
    assert next(item for item in mine if item["id"] == "i0")["revision"] == 1  # ana still sees her own


def test_only_this_machine_and_this_page_are_answered(served):
    _, start = served
    server = start()
    assert call(server, "GET", "/api/items", headers={"Host": "attacker.example"})[0] == 403
    status, _, _ = call(server, "POST", "/api/answer/i0", None, headers={"Content-Type": "application/json"})
    assert status == 403
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
    connection.request("POST", "/api/answer/i0", '{"revision": 0, "skip": true}',
                       {"Host": f"127.0.0.1:{server.server_port}", "Content-Type": "application/json"})
    assert connection.getresponse().status == 403  # no X-ModelLab header: a form on another site
    for path in ("/page/../gold.py", "/page/..%2Fgold.py", "/plugins/task.toml", "/media/../items.jsonl"):
        assert call(server, "GET", path)[0] == 404, path


def test_media_is_served_with_ranges_for_video(served):
    _, start = served
    server = start()
    url = json.loads(call(server, "GET", "/api/items")[2])["items"][0]["fields"]["a"]["url"]
    status, headers, data = call(server, "GET", url, headers={"Range": "bytes=10-19"})
    assert status == 206 and data == bytes(range(10, 20)) and headers["Content-Range"] == "bytes 10-19/1024"
    assert call(server, "GET", url, headers={"Range": "bytes=2000-"})[0] == 416
    status, _, data = call(server, "GET", url)
    assert status == 200 and len(data) == 1024


def test_the_page_and_task_are_served(served):
    _, start = served
    server = start()
    status, headers, page = call(server, "GET", "/")
    assert status == 200 and b"label.js" in page and headers["Content-Type"] == "text/html"
    task = json.loads(call(server, "GET", "/api/task")[2])
    assert task["labeller"] == "ana" and task["plugins"] == [] and task["questions"][0]["kind"] == "rank"


def test_marks_are_saved_only_when_they_match_the_item(served):
    folder, start = served
    rows = [json.loads(line) for line in (folder / "items.jsonl").read_text().splitlines()]
    for row in rows:
        row["note"] = "Rain on the café 🌧 window"
    (folder / "items.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    server = start(task=TASK + '\n[[pane]]\nfield = "note"\nshow = "text"\n'
                               '[[question]]\nid = "mark"\nkind = "span"\npane = "note"\ntags = ["weather"]\n')
    rain = {"start": 0, "end": 4, "text": "Rain", "tag": "weather"}
    cloud = {"start": 17, "end": 18, "text": "🌧", "tag": "weather"}  # code points, not UTF-16 units
    base = {"better": [["a"], ["b"]], "sure": 2}
    assert answer(server, "i0", 0, {**base, "mark": {"note": [rain, cloud]}}) == (200, {"revision": 1})
    status, reply = answer(server, "i1", 0, {**base, "mark": {"note": [{**cloud, "start": 18, "end": 19}]}})
    assert status == 422 and "is not the text at 18..19" in reply["error"]
    assert gold.read_labels(folder / "labels.jsonl")[-1]["answers"]["mark"] == {"note": [rain, cloud]}


def test_the_label_command_refuses_versions_and_odd_names(served):
    folder, _ = served
    frozen = subprocess.run([sys.executable, CLI, "label", f"{folder}@abcdef12", "--no-open"],
                            capture_output=True, text=True)
    assert frozen.returncode != 0 and "does not change" in frozen.stderr
    odd = subprocess.run([sys.executable, CLI, "label", str(folder), "--labeller", "a b", "--no-open"],
                         capture_output=True, text=True)
    assert odd.returncode != 0 and "--labeller" in odd.stderr


def test_the_page_and_media_carry_protections(served):
    _, start = served
    server = start()
    _, page_headers, _ = call(server, "GET", "/")
    assert "frame-ancestors 'none'" in page_headers["Content-Security-Policy"]
    assert "default-src 'self'" in page_headers["Content-Security-Policy"]
    assert page_headers["X-Frame-Options"] == "DENY" and page_headers["X-Content-Type-Options"] == "nosniff"
    _, api_headers, _ = call(server, "GET", "/api/items")
    assert api_headers["X-Content-Type-Options"] == "nosniff"
    url = json.loads(call(server, "GET", "/api/items")[2])["items"][0]["fields"]["a"]["url"]
    _, media_headers, _ = call(server, "GET", url)
    assert media_headers["Content-Security-Policy"].startswith("sandbox") and media_headers["X-Content-Type-Options"] == "nosniff"
    other = start("ben")
    assert json.loads(call(other, "GET", "/api/items")[2])["items"][0]["fields"]["a"]["url"] != url  # secret per server


def test_an_image_pane_shows_only_files_the_set_names(served):
    folder, start = served
    rows = [json.loads(line) for line in (folder / "items.jsonl").read_text().splitlines()]
    rows[0]["a"] = {"url": "https://example.com/tracker.png"}
    (folder / "items.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    server = start()
    first = next(item for item in json.loads(call(server, "GET", "/api/items")[2])["items"] if item["id"] == "i0")
    assert first["fields"]["a"] is None


def test_a_skip_does_not_count_as_an_answer(served):
    _, start = served
    ana = start("ana")
    assert call(ana, "POST", "/api/answer/i0", {"revision": 0, "skip": True})[0] == 200
    ben = start("ben")
    assert "i0" in [item["id"] for item in json.loads(call(ben, "GET", "/api/items")[2])["items"]]


def test_a_server_failure_is_answered_not_dropped(served):
    folder, start = served
    server = start()
    (folder / "labels.jsonl").write_text("{broken\n")
    status, _, data = call(server, "GET", "/api/items")
    assert status == 500 and "The server failed" in json.loads(data)["error"]


def test_only_one_of_two_saves_of_the_same_revision_wins(served):
    folder, start = served
    server = start()
    results = []

    def save(sure):
        results.append(answer(server, "i3", 0, {"better": [["a"], ["b"]], "sure": sure})[0])
    threads = [threading.Thread(target=save, args=(n,)) for n in (1, 2, 3, 1, 2, 3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == [200, 409, 409, 409, 409, 409]
    assert [r["revision"] for r in gold.read_labels(folder / "labels.jsonl")] == [1]


def test_an_answer_to_a_question_not_asked_is_refused(served):
    _, start = served
    server = start(task=TASK + '\n[[question]]\nid = "why"\nkind = "text"\nwhen = {sure = [1]}\n')
    status, reply = answer(server, "i4", 0, {"better": [["a"], ["b"]], "sure": 3, "why": "unsure"})
    assert status == 422 and "why: not asked" in reply["error"]


def test_a_plugin_is_served_by_its_encoded_path(served):
    folder, start = served
    (folder / "my plugins").mkdir()
    (folder / "my plugins" / "mood kind.js").write_text("window.ModelLab.kinds.mood = {render() {}};\n")
    server = start(task=TASK + '\n[[question]]\nid = "mood"\nkind = "mood"\nrequired = false\n'
                               '[plugins]\nkinds = {mood = {page = "my plugins/mood kind.js"}}\n')
    assert json.loads(call(server, "GET", "/api/task")[2])["plugins"] == ["my plugins/mood kind.js"]
    status, _, body = call(server, "GET", "/plugins/my%20plugins/mood%20kind.js")
    assert status == 200 and b"ModelLab.kinds.mood" in body
