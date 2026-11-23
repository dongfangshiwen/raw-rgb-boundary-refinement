"""Show the manuscript values after the archived count-based replays have passed."""
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

BACKBONES = ("mask2former", "segformer")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def interval(result):
    lo, hi = result["ci95"]
    return f'{result["pp"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'


def table_4(work):
    s4 = work / "S4"
    original = read(s4 / "four-decoder-fog/results.json")["targets"]
    fitted = read(s4 / "kernel-ablation/refit-results.json")["targets"]
    pairs = ("fixed_four_minus_bilinear", "fixed_four_minus_gaussian_jbu",
             "source_fit_minus_gaussian_jbu", "source_fit_minus_fixed_four")
    rows = [
        [interval(original[b]["regions"]["boundary"]["paired"][pair]) for b in BACKBONES]
        for pair in pairs
    ]
    for arm in ("colour_only", "spatial_only", "joint"):
        rows.append([interval(fitted[b]["regions"]["boundary"]["compared_to_fixed_pp"][arm])
                     for b in BACKBONES])
    return rows


def table_5(work):
    operators = read(work / "S1/idd/operator-comparisons.json")["metrics"]
    regions = ("boundary", "full")
    reference = operators["jbu"]
    rows = [[f'{reference[region]["ours"]:.3f}' for region in regions]]
    for method in ("jbu", "dysample", "sapa_context"):
        row = []
        for region in regions:
            metric = operators[method][region]
            lo, hi = metric["ci95"]
            row.extend((f'{metric["reference"]:.3f}', f'{metric["delta"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'))
        rows.append(row)
    return rows


def table_6(work):
    targets = read(work / "S2/semantic-results.json")["bddval"]
    rows = []
    for b in BACKBONES:
        for arm in ("paper_fit", "bilinear_bounded4"):
            pair = targets[b]["all"]["contrasts"][arm + " minus released"]
            rows.append([interval(pair[region]) for region in ("boundary", "full")])
    return rows


def table_10(work):
    panels = read(work / "S4/records/results.json")["panels"]
    pairs = (("boundary", "paper - released"), ("full", "paper - released"),
             ("boundary", "paper - fixed"), ("boundary", "by_scale - shared"),
             ("boundary", "by_scale - shuffled_scale"))
    rows = []
    for region, pair in pairs:
        rows.append([interval(panels[b]["new_bdd"]["scales"]["768"][region]["contrasts"][pair])
                     for b in BACKBONES])
    return rows


def table_11(work):
    records = read(work / "S4/records/stf-results.json")["results"]
    rows = []
    for region in ("near12", "all_points"):
        for b in BACKBONES:
            endpoint = records[b]["all"]["endpoints"][region]
            scores = endpoint["scores"]
            rows.append([f'{scores["released"]:.3f}', f'{scores["paper"]:.3f}',
                         interval(endpoint["contrasts"]["paper - released"])])
    return rows


def table_17(work):
    path = work / "S1/source-diagnostics/Original supplementary archive.zip"
    with ZipFile(path) as archive:
        results = json.loads(archive.read("data/joint/joint-test.json"))
    rows = []
    for cohort in ("MUSES_t70", "MUSES_t40"):
        photo = results["photos"][cohort]["scores"]
        full = results["full"][cohort]["scores"]["curve/fixed"]
        boundary = results["edge"][cohort]
        delta = boundary["effects"]["curve-fixed_minus_mean"]
        lo, hi = delta["ci95"]
        rows.append([f'{photo["psnr"]["curve"]:.2f}', f'{photo["deltaE"]["curve"]:.2f}',
                     f'{full:.2f}', f'{boundary["scores"]["curve/fixed"]:.2f}',
                     f'{delta["diff"]:+.3f}', f'[{lo:.3f}, {hi:.3f}]'])
    return rows


RECORDED_TABLES = {4: table_4, 5: table_5, 6: table_6, 10: table_10, 11: table_11, 17: table_17}


def show_verified_cells(work, published, selected):
    with published.open(encoding="utf-8-sig", newline="") as stream:
        cells = list(csv.DictReader(stream))
    for number in selected:
        if number not in RECORDED_TABLES:
            continue
        values = RECORDED_TABLES[number](work)
        cells_for_table = [c for c in cells if int(c["Table"]) == number]
        expected_count = sum(len(row) for row in values)
        if len(cells_for_table) != expected_count:
            raise ValueError(f"Wrong number of published values for Table {number}")
        positions = {(int(c["Data row"]), int(c["Numeric column"])): c for c in cells_for_table}
        print(f"\nTable {number}: {expected_count} numerical cells checked at publication precision")
        for i, row in enumerate(values, 1):
            line = []
            for j, value in enumerate(row, 1):
                cell = positions[i, j]
                if value != cell["Published value"].replace("\N{MINUS SIGN}", "-"):
                    raise ValueError(f"Table {number}, {cell['Row label']}, {cell['Column label']}: "
                                     f"replay {value} differs from {cell['Published value']}")
                line.append(f"{cell['Column label']} {value}")
            print("  " + positions[i, 1]["Row label"] + " | " + " | ".join(line))


def verify_bdd_selection(s2):
    inventory = read(s2 / "bdd-screen-inventory.json")["rows"]
    candidates = read(s2 / "bdd-screen-candidates.json")["indices"]
    adjudication = read(s2 / "bdd-screen-decisions.json")
    selected = read(s2 / "bddval-selection.json")
    fog = read(s2 / "fogreserve-manifest.json")
    official = read(s2 / "bddval-manifest.json")
    for key, name in (("decision", "bdd-screen-decisions.json"),
                      ("inventory", "bdd-screen-inventory.json"),
                      ("protocol", "bdd-fog-screen-protocol.json")):
        if hashlib.sha256((s2 / name).read_bytes()).hexdigest() != fog[key]:
            raise ValueError(f"Screen record checksum differs for {name}")
    if hashlib.sha256((s2 / "bddval-selection.json").read_bytes()).hexdigest() != official["selection"]:
        raise ValueError("The validation manifest cites a different selection record")
    ids = [r["id"] for r in inventory]
    prefix = lambda name: name.split("-")[0]
    candidate_ids = {ids[i] for i in candidates}
    accepted = sorted(ids[int(i)] for i, v in adjudication["decisions"].items() if v["accept"])
    natives = set(adjudication["native_views_inspected"])
    if not (len(set(ids)) == len(ids) == 1000 and set(candidates) == set(map(int, adjudication["decisions"]))
            and len(candidate_ids) == 32 and len(natives) == 5 and natives <= set(candidates)
            and len(accepted) == 4 and set(accepted) <= {ids[i] for i in natives}
            and accepted == sorted(r["id"] for r in fog["rows"])
            and sorted(candidate_ids) == selected["excluded_possible_fog"]):
        raise ValueError("The candidate decisions differ from the source manifests")
    excluded_prefixes = {prefix(key) for key in candidate_ids}
    blocked = excluded_prefixes | set(selected["previous_prefixes"])
    replay = []
    for key in sorted(ids, key=lambda key: hashlib.sha256(("axis101-locked/" + key).encode()).digest()):
        capture = prefix(key)
        if capture in blocked:
            continue
        replay.append(key)
        blocked.add(capture)
        if len(replay) == 128:
            break
    if replay != selected["IDs"] or replay != [r["id"] for r in official["rows"]]:
        raise ValueError("The reconstructed validation draw differs from the 128 published IDs")
    print(f"Table 6 selection: {len(ids)} RGB thumbnails, {len(candidates)} candidates, "
          f"{len(excluded_prefixes)} candidate capture prefixes excluded. "
          f"All {len(replay)} validation IDs replayed in order.")
