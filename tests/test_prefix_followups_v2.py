from experiments.prefix_followups_v2 import build_followups


def test_targeted_selection_controls_and_dedup_keep_the_original_query_arrays():
    def row(identifier, method, k, probes, recall, latency, **extra):
        return dict(setting_id=identifier, method=method, router="/router", clusters=1024,
                    top_k=k, probes=probes, recall=recall, p50_ms=latency,
                    complete=True, qualified=True, repetitions=1, **extra)

    rows = [row("a-low", "scan", 1, 2, .5, 1),
            row("a-high", "scan", 1, 8, .9, 4),
            row("a-all", "scan", 1, 1024, 1, 10),
            row("b-win", "branch", 1, 8, .8, 2),
            row("b-same-point", "branch", 1, 8, .8, 2),
            row("c-dominated", "prefix", 1, 8, .7, 3),
            row("c-small-gain", "prefix", 1, 1024, 1, 9.8),
            row("b-budget", "branch", 1, 1024, .99, 6,
                node_budget=512, mean_bitplane_words=0),
            row("a-ten", "scan", 10, 8, 1, 1),
            row("a-hundred", "scan", 100, 8, 1, 1)]
    rows.append(dict(row("unqualified", "prefix", 1, 8, .999, .01), qualified=False))
    jobs = [dict(job_id=f"original-{method}", phase="main", pool="/fixed-pool",
                 method=method, router="/router", clusters=1024, dimensions=256,
                 documents=100, query_rows=[0, 1], max_prefix_bits=32,
                 ram_budget_bytes=1000000, variants=[])
            for method in ("scan", "branch", "prefix")]
    layouts = [dict(path="/router", clusters=1024, probes={"1": [2, 8, 1024]})]
    config = dict(top_ks=[1, 10, 100], recall_targets=[.5, .9], repetitions=1)
    calls = []

    def resolve(layout, k, target):
        calls.append((layout["path"], k, target))
        return dict(probes=3 if target == .8 else 512, rank_cutoff=3,
                    routing_recall=target)

    followups, evidence = build_followups(rows, jobs, layouts, config, resolve)
    selected = {item["setting_id"] for item in evidence["apparent_advantages"]}
    assert selected == {"b-win", "b-same-point", "b-budget"}
    assert len(calls) == 2  # Equal targets share one routing calculation.
    variants = [(job["method"], variant) for job in followups for variant in job["variants"]]
    assert {(method, v["top_k"], v["probes"]) for method, v in variants if method == "scan"} == {
        ("scan", 1, 3), ("scan", 1, 512)}
    assert {(v["top_k"], v["probes"]) for method, v in variants if method == "prefix"} == {
        (1, 2), (1, 8), (10, 8), (100, 8)}
    assert all(v["start_depth"] == v["candidate_target"] == 0
               for method, v in variants if method == "prefix")
    control = next(v for method, v in variants if method == "scan" and v["probes"] == 512)
    assert {reason["kind"] for reason in control["reasons"]} == {
        "match_alternative_recall", "zero_split_budget_512"}
    assert all(job["pool"] == "/fixed-pool" and job["query_rows"] == [0, 1] for job in followups)
    assert len({v["setting_id"] for _, v in variants}) == len(variants) == 6
    assert all(v["setting_id"].startswith("followup-") for _, v in variants)
