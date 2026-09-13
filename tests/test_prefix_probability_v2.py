"""The probability study uses real split work and removes duplicate budgets."""

from experiments import prefix_probability_v2 as probability
import pytest


def test_selection_fallback_and_budget_deduplication_keep_queries_and_router():
    def setting(identifier, k, recall, latency, budget, split_words=10):
        return dict(setting_id=identifier, method="branch", top_k=k, recall=recall,
                    p50_ms=latency, node_budget=budget, mean_bitplane_words=split_words,
                    complete=True, qualified=True)

    rows = [setting("zero-splits", 1, .99, .01, 128, 0),
            setting("k1-best", 1, .96, .2, 128),
            setting("k1-slower", 1, .98, .3, 256),
            setting("k100-fallback", 100, .85, .4, 256)]
    jobs = [dict(job_id="source", method="branch", router="saved-router", clusters=64,
                 pool="saved-pool", query_rows=list(range(1000)), dimensions=32,
                 variants=[dict(setting_id=row["setting_id"], top_k=row["top_k"],
                                node_budget=row["node_budget"], probes=12, leaf_size=8,
                                exploration=0, seed=7, prefer_deeper_ties=True, repetitions=1)
                           for row in rows])]
    schedule, evidence = probability.build_jobs(rows, jobs, id_prefix="probability-test")
    assert [(row["top_k"], row["source_setting_id"], row["target"])
            for row in evidence["selected"]] == [(1, "k1-best", .95), (100, "k100-fallback", .8)]
    variants = [variant for job in schedule for variant in job["variants"]]
    assert len(variants) == 27  # Nine at K=1, eighteen at K=100; budget128 is not repeated.
    assert len({row["setting_id"] for row in variants}) == 27
    assert {row["exploration"] for row in variants} == {0, .1, .2}
    assert {row["seed"] for row in variants} == {7, 23, 42}
    assert all(job["query_rows"] == list(range(1000)) and job["router"] == "saved-router"
               and job["pool"] == "saved-pool" for job in schedule)
    assert all(row["prefer_deeper_ties"] and row["leaf_size"] == 8 and row["probes"] == 12 for row in variants)
    empty, omitted = probability.build_jobs(rows[:1], jobs, id_prefix="omitted")
    assert not empty
    assert [row["top_k"] for row in omitted["omitted"]] == [1, 100]


def test_plot_groups_require_three_qualified_seeds_on_the_selected_layout():
    selection = dict(probabilities=[0, .1, .2], seeds=[7, 23, 42], selected=[dict(
        top_k=1, source_setting_id="base", router="fixed-router", probes=12,
        leaf_size=8, node_budgets=[128, 512], prefer_deeper_ties=True)])

    def row(p, seed, recall, milliseconds, **changes):
        return dict(dict(method="branch", top_k=1, source_setting_id="base", router="fixed-router",
                         probes=12, leaf_size=8, node_budget=512, prefer_deeper_ties=True,
                         exploration=p, seed=seed, recall=recall, p50_ms=milliseconds,
                         complete=True, qualified=True), **changes)

    rows = [row(0, 7, .8, 1), row(0, 23, .9, 100), row(0, 42, 1, 2),
            row(.1, 7, .9, 1), row(.1, 23, .9, 2),
            row(.1, 42, .99, .01, router="different-router"),
            row(.2, 7, .9, 1), row(.2, 23, .9, 2), row(.2, 42, .99, .01, qualified=False)]
    groups = probability.group_probability_rows(rows, selection)
    assert len(groups) == 6  # Missing budgets remain in the summary.
    complete = next(group for group in groups if group["node_budget"] == 512 and group["probability"] == 0)
    assert complete["plotted"] is True and complete["missing_seeds"] == []
    assert complete["recall_percent_mean"] == pytest.approx(90)
    assert (complete["recall_percent_min"], complete["recall_percent_max"]) == (80, 100)
    assert (complete["query_ms_median"], complete["query_ms_min"], complete["query_ms_max"]) == (2, 1, 100)
    missing = next(group for group in groups if group["node_budget"] == 512 and group["probability"] == .1)
    assert missing["plotted"] is False and missing["missing_seeds"] == [42]
    assert missing["query_ms_median"] is None
    failed = next(group for group in groups if group["node_budget"] == 512 and group["probability"] == .2)
    assert failed["plotted"] is False and failed["unqualified_seeds"] == [42]
    assert all(group["missing_seeds"] == [7, 23, 42] and not group["plotted"]
               for group in groups if group["node_budget"] == 128)
