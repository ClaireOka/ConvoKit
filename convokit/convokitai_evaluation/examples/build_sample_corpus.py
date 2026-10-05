"""
Build the sample corpus used by the examples: the completed cohorts of one simulated
Track 1 experiment (one mediator, two agent participants, one cohort per stance pair) for each
of three mediators, converted with convokitai-simulation's export_to_corpus and saved to
``sample_corpus/``.

    python build_sample_corpus.py

Needs the TrAuSt-util simulation results (SIM) and the convokitai-simulation package.
"""

import json
import os

from export_to_corpus import export_to_corpus

SIM = "/Users/caiyang/Documents/Research Projects/Deliberate Lab/GitHub/TrAuSt-util/simulation_results"
SOURCES = {  # mediator label -> (submission file, folder whose runs.json lists its experiments)
    "greeting": ("greeting_mediator.yaml", f"{SIM}/baseline_mediators/track 1/clean"),
    "paraphrase": (
        "paraphrase_mediator.yaml",
        f"{SIM}/baseline_mediators/track 1/clean",
    ),
    "constructive": (
        "[default] constructive mediator.yaml",
        f"{SIM}/final_pool_batches/batch1/Track 1/clean",
    ),
}
OUT = os.path.dirname(os.path.abspath(__file__))


def load_one(mediator, folder):
    """The export of the first experiment run by `mediator`, trimmed to its completed cohorts."""
    runs = json.load(open(f"{folder}/runs.json"))
    experiment = next(
        e
        for r in runs
        if r["mediator"] == mediator
        for e in r["experiments"]
        if e.get("completed_cohorts")
    )
    eid, cids = experiment["experiment_id"], experiment["completed_cohorts"]
    export = json.load(open(f"{folder}/{eid}/export.json"))
    export["cohortMap"] = {cid: export["cohortMap"][cid] for cid in cids}
    topic = json.load(open(f"{folder}/{eid}/meta.json"))["topic"]
    return export, topic


def statement(export):
    """The debate statement shown with the chat, without its 'Statement: "..."' wrapping."""
    text = export["stageMap"]["chat-round-1"]["descriptions"]["primaryText"].strip()
    return text.removeprefix("Statement:").strip().strip('"')


exports, extra = [], {}
for label, (mediator, folder) in SOURCES.items():
    export, topic = load_one(mediator, folder)
    exports.append(export)
    for cid, cohort in export["cohortMap"].items():
        answers = cohort["dataMap"]["pre-survey-1"]["participantAnswerMap"]
        extra[cid] = {
            "mediator": label,
            "topic": topic,
            "statement": statement(export),
            "pre_survey": {
                sid: a["pre_q1_a.5"]["answer"] for sid, a in answers.items()
            },
        }

corpus = export_to_corpus(exports)
for convo in corpus.iter_conversations():
    for key, value in extra[convo.id].items():
        convo.meta[key] = value
corpus.dump("sample_corpus", base_path=OUT, force_version=1)
corpus.print_summary_stats()
