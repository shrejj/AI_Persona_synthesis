"""
persona_census_matching.py

Matches literature-derived RAG personas to NASS census counties using a
shared affinity-label vocabulary, blocked by age and farm size first.

Renumbered slightly from the original outline (which had two "Step 4"s):

  Step 1  Label every RAG row with affinity labels from LABEL_CODEBOOK.
  Step 2  Assign each RAG row to one of the 6 fixed age brackets, and to a
          farm-size bucket (LOW/MEDIUM/HIGH) derived from the census farm
          size distribution -- not a fixed category, see derive_farm_size_scale().
  Step 3  Load the census-item -> RAG-column mapping (from
          Census89_RAG_Matching.xlsx) and collapse each category's mapped
          census items into one z-score per county.
  Step 4  Within each (age bracket, farm size bucket) block, cluster RAG rows
          by Jaccard similarity of their affinity label sets. For each
          cluster, collapse the DEMO columns (no census correspondence) into
          one cell, and ask the LLM to synthesize a short description and a
          barrier-to-adoption statement from the clustered rows' text.
  Step 5  Median-split each census-mapped category's per-county z-score into
          HIGH/LOW, the same vocabulary used for the RAG labels.
  Step 6  Match each persona (cluster) to counties by fraction of matching
          defining categories, keep everything above MIN_MATCH_FRACTION, and
          build the per-persona table (matched counties x all census columns).

CENSUS-SIDE CAVEATS baked into this script, worth remembering when reading
output:
  - "environment", "price_risk", and "carbon" have NO census mapping
    (they're marked DEMO_Column in the mapping workbook). They still get
    RAG-side affinity labels, but they can never be a "defining column" for
    county matching -- only regen_ag, farming_ops, fertilizer, finances, and
    insurance can be.
  - Farm size bucketing compares a survey respondent's self-reported farm
    size (free text) against a COUNTY-LEVEL aggregate acreage score. That's
    a real unit mismatch (individual vs. county aggregate) -- treat the farm
    size block as a rough sort, not a precise match, until/unless you get a
    farm-size distribution rather than a county total.
  - This script is self-contained (no imports from your other pipeline
    files), same reasoning your own scripts already use: importing a module
    with GPU-touching top-level code (embeddings, vectorstores) initializes
    it a second time and can fight your vLLM server for memory.

Run this on the SAME compute node where `vllm serve` is running, with your
venv activated.
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser

# ---------------------------------------------------------------------------
# CONFIG -- edit these
# ---------------------------------------------------------------------------
RAG_INPUT_XLSX = "/scratch/adriaens_root/adriaens1/shrejj/results_RAG6.xlsx"
CENSUS_MAPPING_XLSX = "/scratch/adriaens_root/adriaens1/shrejj/Census_data/Census89_RAG_Matching.xlsx"
CENSUS_NORMALIZED_XLSX = "/scratch/adriaens_root/adriaens1/shrejj/Census_data/trial98_analysis_normalized.xlsx"
CENSUS_NORMALIZED_SHEET = "Sheet2_normalized"
CENSUS_RAW_XLSX = "/scratch/adriaens_root/adriaens1/shrejj/Census_data/trial98_analysis_reshaped.xlsx"  # optional: path to a wide, NON-normalized version of the same
# table (same row/column layout as CENSUS_NORMALIZED_XLSX) so the per-persona
# tables can show real acreage/dollar values instead of z-scores. If None,
# the z-scores themselves are shown.

OUTPUT_XLSX = "/scratch/adriaens_root/adriaens1/shrejj/persona_census_match_3.xlsx"
DEBUG_LOG = "/scratch/adriaens_root/adriaens1/shrejj/persona_census_match_debug3.log"
# Step 1 labels are written here as soon as each row finishes, so a crash
# anywhere later in the pipeline (or even mid-labeling) doesn't cost you the
# LLM calls already made. Re-running main() picks this up automatically and
# skips rows already labeled.
LABELS_CHECKPOINT_JSON = "/scratch/adriaens_root/adriaens1/shrejj/row_labels_checkpoint.json"
VLLM_BASE_URL = "http://localhost:8000/v1"

# Step 2: fixed age brackets (from NASS Domain Category labels)
AGE_BRACKETS = [
    "PRODUCERS, AGE 25 TO 34",
    "PRODUCERS, AGE 35 TO 44",
    "PRODUCERS, AGE 45 TO 54",
    "PRODUCERS, AGE 55 TO 64",
    "PRODUCERS, AGE 65 TO 74",
    "PRODUCERS, AGE GE 75",
]

FARM_SIZE_BUCKETS = ["LOW", "MEDIUM", "HIGH"]

# Future hook: filter counties/RAG rows by state/county before blocking.
# Left as None (no filter) until you're ready to wire it in.
STATE_FILTER: Optional[List[str]] = None

# Step 4: Jaccard-distance cutoff for merging RAG rows into one persona
# cluster within a block. 0 = only identical label sets merge (like the old
# exact-match grouping). Higher = looser merging. Tune against real output.
CLUSTER_DISTANCE_THRESHOLD = 0.5

# Step 6: a county must match at least this fraction of a persona's defining
# categories (regen_ag/farming_ops/fertilizer/finances/insurance only) to be
# considered a match at all.
MIN_MATCH_FRACTION = 0.6

NA_TOKENS = {"", "na", "n/a", "nan", "none", "not specified", "not available"}


def is_missing(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    return str(val).strip().lower() in NA_TOKENS


# ---------------------------------------------------------------------------
# RAG schema
# ---------------------------------------------------------------------------
VIEW_COLUMNS = [
    "View on Regenerative Agriculture (Cover crop, No-Till)",
    "Views on farming operations and practices",
    "View on Fertilizer Use",
    "Farm Finances Perspective",
    "Views on Insurance",
    "Views on Price Risk Mgmt",
    "Views on Carbon Markets",
]

ID_COLUMNS = ["Source File", "Unique ID", "Study Sample Size", "Pub Year"]

# Only these categories have a census mapping and can be "defining columns"
# for county matching (see CENSUS-SIDE CAVEATS above).
RAG_COLUMN_TO_CATEGORY = {
    "View on Regenerative Agriculture (Cover crop, No-Till)": "regen_ag",
    "Views on farming operations and practices": "farming_ops",
    "View on Fertilizer Use": "fertilizer",
    "Farm Finances Perspective": "finances",
    "Views on Insurance": "insurance",
    "Views on environment": "environment"
}
CENSUS_MAPPED_CATEGORIES = list(RAG_COLUMN_TO_CATEGORY.values())

# Blocking columns (not affinity categories): age and farm size
BLOCKING_RAG_COLUMNS = {
    "Avg. Age": "age",
    "Farm Size": "farm_size",
}

SOURCE_INFO_COLUMN = "Farmer Informant"
EDUCATION_COLUMN = "Avg Education"
STATE_COLUMN = "State / County"

LABEL_CODEBOOK: Dict[str, str] = {
    "regen_ag_high_affinity": "Respondent is likely to adopt or continue regenerative practices such as cover cropping or no-till.",
    "regen_ag_low_affinity": "Respondent is unlikely to adopt regenerative practices such as cover cropping or no-till.",
    "farming_ops_high_affinity": "Respondent is likely to change their farming operations or practices for new technology.",
    "farming_ops_low_affinity": "Respondent is unlikely to change their farming operations or practices for new technology.",
    "fertilizer_high_affinity": "Respondent is likely to increase their fertilizer use or maintain high fertilizer use.",
    "fertilizer_low_affinity": "Respondent is likely to reduce fertilizer use or maintain very low use.",
    "finances_high_affinity": "Respondent is likely to change practices to improve farm finances.",
    "finances_low_affinity": "Respondent is unlikely to change practices to improve farm finances.",
    "insurance_high_affinity": "Respondent is likely to adopt or expand use of insurance products.",
    "insurance_low_affinity": "Respondent is unlikely to adopt or expand insurance use.",
    "environment_high_affinity": "Respondent is likely to adapt practices in response to environmental risk or climate change.",
    "environment_low_affinity": "Respondent is unlikely to adapt practices in response to environmental risk or climate change.",
    "price_risk_high_affinity": "Respondent is likely to change practices for a new price risk management strategy.",
    "price_risk_low_affinity": "Respondent is unlikely to adopt a new price risk management strategy.",
    "carbon_high_affinity": "Respondent is likely to participate in carbon markets.",
    "carbon_low_affinity": "Respondent is unlikely to participate in carbon markets.",
}
ALL_LABELS = list(LABEL_CODEBOOK.keys())


def label_category(label: str) -> str:
    """'regen_ag_high_affinity' -> 'regen_ag'. Relies on the fixed
    '_high_affinity' / '_low_affinity' suffix convention above."""
    return label.replace("_high_affinity", "").replace("_low_affinity", "")


def label_direction(label: str) -> str:
    return "HIGH" if label.endswith("_high_affinity") else "LOW"


def format_codebook(order: List[str]) -> str:
    return "\n".join(f"- {l}: {LABEL_CODEBOOK[l]}" for l in order)


# ---------------------------------------------------------------------------
# STEP 1: label every RAG row
# ---------------------------------------------------------------------------
class RowLabels(BaseModel):
    labels: List[str] = Field(default_factory=list, description="0 or more labels from the codebook.")

    class Config:
        populate_by_name = True


LABELING_RULES = (
    "You are coding one grower's stated views against a fixed persona codebook.\n"
    "Rule 1: assign every label clearly supported by the respondent's own words; "
    "do not guess from N/A or from context outside the text.\n"
    "Rule 2: a respondent can have more than one label if genuinely supported.\n"
    "Rule 3: if nothing in the text clearly supports any label, return an empty list.\n"
    "Rule 4: base labels only on stated attitudes/behaviour, never on demographic facts.\n\n"
    "LABEL CODEBOOK:\n{codebook}"
)


def row_text_for_labeling(row: pd.Series) -> str:
    lines = []
    for col in VIEW_COLUMNS:
        if col in row and not is_missing(row[col]):
            lines.append(f"{col}: {row[col]}")
    return "\n".join(lines) if lines else "(no view data reported)"


def build_labeling_chain(llm: ChatOpenAI):
    base_parser = PydanticOutputParser(pydantic_object=RowLabels)
    parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
    prompt = ChatPromptTemplate.from_template(
        "{rules}\n\n{format_instructions}\n\nRespondent text:\n{row_text}"
    )
    return prompt | llm | parser, base_parser.get_format_instructions()


def save_labels_checkpoint(df: pd.DataFrame, row_labels: Dict[int, List[str]], path: str) -> None:
    """Keyed by Unique ID rather than the pandas index, so the checkpoint
    still lines up correctly if the RAG file gets reloaded with a different
    row order later."""
    payload = {}
    for idx, labels in row_labels.items():
        uid = str(df.loc[idx, "Unique ID"]) if "Unique ID" in df.columns else str(idx)
        payload[uid] = labels
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_labels_checkpoint(df: pd.DataFrame, path: str) -> Dict[int, List[str]]:
    """Returns {} if no checkpoint exists yet, or if none of its Unique IDs
    match the current df (e.g. a stale checkpoint from a different file)."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        payload = json.load(f)
    row_labels: Dict[int, List[str]] = {}
    if "Unique ID" in df.columns:
        uid_to_idx = {str(v): idx for idx, v in df["Unique ID"].items()}
        for uid, labels in payload.items():
            if uid in uid_to_idx:
                row_labels[uid_to_idx[uid]] = labels
    else:
        for uid, labels in payload.items():
            try:
                row_labels[int(uid)] = labels
            except ValueError:
                continue
    return row_labels


def label_all_rows(df: pd.DataFrame, llm: ChatOpenAI,
                    checkpoint_path: Optional[str] = None) -> Dict[int, List[str]]:
    chain, fmt = build_labeling_chain(llm)
    rules = LABELING_RULES.format(codebook=format_codebook(ALL_LABELS))

    out: Dict[int, List[str]] = {}
    if checkpoint_path:
        out.update(load_labels_checkpoint(df, checkpoint_path))
        if out:
            print(f"Resuming from checkpoint: {len(out)} row(s) already labeled, skipping those.")

    for idx, row in df.iterrows():
        if idx in out:
            continue
        row_id = str(row.get("Unique ID", idx))
        text = row_text_for_labeling(row)
        try:
            result: RowLabels = chain.invoke({
                "rules": rules, "format_instructions": fmt, "row_text": text,
            })
            out[idx] = [l for l in result.labels if l in ALL_LABELS]
        except Exception as e:
            with open(DEBUG_LOG, "a") as f:
                f.write(f"\n\n=== {row_id} labeling FAILED ===\n{e}\n")
            out[idx] = []
        print(f"  {row_id}: {out[idx]}")
        if checkpoint_path:
            # Written after every row on purpose: a crash later in this
            # function, or anywhere downstream in the pipeline, still leaves
            # every row labeled so far safely on disk.
            save_labels_checkpoint(df, out, checkpoint_path)
    return out


# ---------------------------------------------------------------------------
# STEP 2: age bracket + farm size bucket assignment
# ---------------------------------------------------------------------------
class BracketAssignment(BaseModel):
    bracket: str = Field(description="Exactly one value from the allowed list, or 'UNKNOWN'.")

    class Config:
        populate_by_name = True


def build_bracket_chain(llm: ChatOpenAI):
    base_parser = PydanticOutputParser(pydantic_object=BracketAssignment)
    parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
    prompt = ChatPromptTemplate.from_template(
        "{rules}\n\n{format_instructions}\n\nAllowed values:\n{allowed}\n\nText:\n{text}"
    )
    return prompt | llm | parser, base_parser.get_format_instructions()


def assign_age_brackets(df: pd.DataFrame, llm: ChatOpenAI) -> Dict[int, str]:
    chain, fmt = build_bracket_chain(llm)
    rules = (
        "Assign this respondent group's age description to the single closest age "
        "bracket below. If the text spans multiple brackets, pick the one that best "
        "represents the center of the described range. If age is not reported or "
        "cannot be inferred, return 'UNKNOWN'."
    )
    out: Dict[int, str] = {}
    for idx, row in df.iterrows():
        text = str(row.get("Avg. Age", "")) if "Avg. Age" in row else ""
        if is_missing(text):
            out[idx] = "UNKNOWN"
            continue
        try:
            result: BracketAssignment = chain.invoke({
                "rules": rules, "format_instructions": fmt,
                "allowed": "\n".join(AGE_BRACKETS + ["UNKNOWN"]), "text": text,
            })
            out[idx] = result.bracket if result.bracket in AGE_BRACKETS else "UNKNOWN"
        except Exception as e:
            with open(DEBUG_LOG, "a") as f:
                f.write(f"\n\n=== row {idx} age bracket FAILED ===\n{e}\n")
            out[idx] = "UNKNOWN"
    return out


def assign_farm_size_buckets(
    df: pd.DataFrame, llm: ChatOpenAI, low_cut_desc: str, high_cut_desc: str
) -> Dict[int, str]:
    """low_cut_desc/high_cut_desc are plain-language descriptions of the
    census-derived cutoffs (see derive_farm_size_scale), e.g. 'roughly
    under 120 acres' / 'roughly over 340 acres', so the LLM has a concrete
    anchor instead of guessing what LOW/MEDIUM/HIGH means in isolation.
    Reminder: this compares an individual respondent's self-reported farm
    size to a county-aggregate scale -- a real unit mismatch, see caveats
    at the top of this file."""
    chain, fmt = build_bracket_chain(llm)
    rules = (
        f"Assign this respondent group's farm size description to LOW, MEDIUM, or HIGH.\n"
        f"LOW is {low_cut_desc}. HIGH is {high_cut_desc}. MEDIUM is between the two.\n"
        "If farm size is not reported or cannot be inferred, return 'UNKNOWN'."
    )
    out: Dict[int, str] = {}
    for idx, row in df.iterrows():
        text = str(row.get("Farm Size", "")) if "Farm Size" in row else ""
        if is_missing(text):
            out[idx] = "UNKNOWN"
            continue
        try:
            result: BracketAssignment = chain.invoke({
                "rules": rules, "format_instructions": fmt,
                "allowed": "\n".join(FARM_SIZE_BUCKETS + ["UNKNOWN"]), "text": text,
            })
            out[idx] = result.bracket if result.bracket in FARM_SIZE_BUCKETS else "UNKNOWN"
        except Exception as e:
            with open(DEBUG_LOG, "a") as f:
                f.write(f"\n\n=== row {idx} farm size bucket FAILED ===\n{e}\n")
            out[idx] = "UNKNOWN"
    return out


# ---------------------------------------------------------------------------
# STEP 3: census-item -> RAG-column mapping, and per-county category scores
# ---------------------------------------------------------------------------
def load_census_mapping(path: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """Parses Census89_RAG_Matching.xlsx. Header row = whichever row has the
    literal label 'columns' in its first cell (row 1 in the current file,
    but this doesn't assume a fixed row number, since a blank title row
    above the headers has already shifted this once). Rows below the header
    row = census 'Data Item' strings feeding that column, or a single
    'DEMO_Column' marker if there's no census correspondence.
    Returns (mapping: {rag_column: [data_item, ...]}, demo_columns: [rag_column, ...])."""
    raw = pd.read_excel(path, header=None)

    header_row = None
    for r in range(len(raw)):
        first_cell = raw.iloc[r, 0]
        if isinstance(first_cell, str) and first_cell.strip().lower() == "columns":
            header_row = r
            break
    if header_row is None:
        raise ValueError(
            f"Could not find the header row in {path} (expected a cell reading "
            f"'columns' in column A). Check the file layout hasn't changed again."
        )

    mapping: Dict[str, List[str]] = {}
    demo_columns: List[str] = []
    for col in raw.columns[1:]:
        header = raw.iloc[header_row, col]
        if pd.isna(header):
            continue
        header = str(header).strip()
        items = [str(v).strip() for v in raw.iloc[header_row + 1:, col].tolist() if pd.notna(v)]
        if len(items) == 1 and items[0].upper() == "DEMO_COLUMN":
            demo_columns.append(header)
        else:
            mapping[header] = items
    return mapping, demo_columns


def load_census_wide(path: str, sheet: str) -> pd.DataFrame:
    """Loads a census table: rows = census Data Items (index), columns =
    counties. Falls back to the first sheet in the workbook if `sheet`
    isn't found, since a raw (non-normalized) version of this file may not
    use the same sheet name as the normalized one."""
    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except ValueError:
        print(f"WARNING: sheet '{sheet}' not found in {path}, using the first sheet instead.")
        df = pd.read_excel(path, sheet_name=0)
    df = df.rename(columns={df.columns[0]: "Data Item"}).set_index("Data Item")
    return df


def build_category_scores(census_wide: pd.DataFrame, mapping: Dict[str, List[str]],
                           category_keys: Dict[str, str]) -> pd.DataFrame:
    """category_keys maps RAG column name -> category key (e.g. 'regen_ag').
    Returns a DataFrame: index = county, columns = category key, values =
    mean z-score across that category's mapped census items for that county.
    Missing items are reported once (not per county) so you can catch a
    naming mismatch between the mapping sheet and the census file early."""
    # Case/whitespace-tolerant lookup: 'Farm Size', 'farm size', ' Farm Size '
    # all resolve the same way, so a header formatting difference between
    # this script's constants and your Excel file doesn't silently drop a
    # whole category.
    normalized_category_keys = {k.strip().lower(): v for k, v in category_keys.items()}

    unmatched_mapping_headers = [h for h in mapping if h.strip().lower() not in normalized_category_keys]
    if unmatched_mapping_headers:
        print(f"NOTE: these mapping-sheet columns have no matching category and will be ignored: {unmatched_mapping_headers}")

    matched_categories = {normalized_category_keys[h.strip().lower()] for h in mapping
                           if h.strip().lower() in normalized_category_keys}
    missing_categories = set(category_keys.values()) - matched_categories
    if missing_categories:
        raise ValueError(
            f"These expected categories were not found in the mapping sheet's column "
            f"headers, so they can't be scored: {sorted(missing_categories)}. "
            f"Mapping sheet headers found: {list(mapping.keys())}. "
            f"Check for a spelling/case/whitespace mismatch against RAG_COLUMN_TO_CATEGORY "
            f"/ BLOCKING_RAG_COLUMNS in this script."
        )

    warned = set()
    rows = []
    for county in census_wide.columns:
        row = {}
        for rag_col, items in mapping.items():
            cat = normalized_category_keys.get(rag_col.strip().lower())
            if cat is None:
                continue
            found = [it for it in items if it in census_wide.index]
            missing = [it for it in items if it not in census_wide.index]
            for m in missing:
                if m not in warned:
                    print(f"WARNING: census item not found in normalized table, skipping: {m}")
                    warned.add(m)
            if found:
                row[cat] = census_wide.loc[found, county].astype(float).mean()
            else:
                row[cat] = np.nan
        row["_county"] = county
        rows.append(row)
    out = pd.DataFrame(rows).set_index("_county")
    return out


def validate_category_scores(category_scores: pd.DataFrame, category_keys: Dict[str, str],
                              mapping: Dict[str, List[str]], demo_columns: List[str]) -> None:
    """Fails fast with a specific message instead of a bare KeyError later,
    and tells you exactly which mapping-sheet header is missing or mis-typed."""
    expected_categories = set(category_keys.values())
    found_categories = set(category_scores.columns)
    missing = expected_categories - found_categories
    if not missing:
        return

    missing_rag_cols = [rc for rc, cat in category_keys.items() if cat in missing]
    lines = [f"category_scores is missing expected column(s): {sorted(missing)}"]
    for rag_col in missing_rag_cols:
        if rag_col in demo_columns:
            lines.append(f"  - '{rag_col}' IS in the mapping sheet but is marked DEMO_Column "
                          f"(no census items), so it can't produce a score.")
        elif rag_col in mapping:
            lines.append(f"  - '{rag_col}' is in the mapping sheet, but none of its census items "
                          f"were found in the normalized census table (all warned above).")
        else:
            lines.append(f"  - '{rag_col}' was not found as a header in the mapping sheet at all. "
                          f"Headers present: {sorted(list(mapping.keys()) + demo_columns)}")
    raise ValueError("\n".join(lines))


def derive_farm_size_scale(category_scores: pd.DataFrame) -> Tuple[float, float]:
    """Terciles of the county farm_size score -> (low_cut, high_cut)."""
    vals = category_scores["farm_size"].dropna()
    low_cut = float(vals.quantile(1 / 3))
    high_cut = float(vals.quantile(2 / 3))
    return low_cut, high_cut


def label_counties_high_low(category_scores: pd.DataFrame,
                             categories: List[str]) -> pd.DataFrame:
    """Median split per category, same HIGH/LOW vocabulary as the RAG labels."""
    out = pd.DataFrame(index=category_scores.index)
    for cat in categories:
        median = category_scores[cat].median()
        out[cat] = np.where(category_scores[cat] >= median, "HIGH", "LOW")
        out.loc[category_scores[cat].isna(), cat] = "UNKNOWN"
    return out


# ---------------------------------------------------------------------------
# STEP 4: cluster RAG rows within each (age, farm size) block
# ---------------------------------------------------------------------------
def label_vector(labels: List[str]) -> np.ndarray:
    return np.array([1 if l in labels else 0 for l in ALL_LABELS], dtype=bool)


def cluster_block(indices: List[int], row_labels: Dict[int, List[str]],
                   threshold: float) -> Dict[int, int]:
    """Returns {row_index: cluster_id} for one block. Jaccard distance on
    binary label vectors, hierarchical clustering, flat cut at `threshold`."""
    if len(indices) == 1:
        return {indices[0]: 1}
    vectors = np.stack([label_vector(row_labels.get(i, [])) for i in indices])
    if vectors.sum() == 0:
        # every row in this block has no labels at all -- one cluster
        return {i: 1 for i in indices}
    dist = pdist(vectors, metric="jaccard")
    dist = np.nan_to_num(dist, nan=1.0)  # two all-zero rows -> jaccard is undefined; treat as identical
    Z = linkage(dist, method="average")
    cluster_ids = fcluster(Z, t=threshold, criterion="distance")
    return dict(zip(indices, cluster_ids.tolist()))


class PersonaSynthesis(BaseModel):
    short_description: str = Field(description="A few words to one short phrase naming this persona's defining trait.")
    barrier_to_adoption: str = Field(description="1-2 sentences on what this group's own words cite as the barrier(s) to adoption based on obstacles described")
    source_of_information: str = Field(description="One aggregated, comma-separated statement of every unique source of information reported across these respondents.")
    education_summary: str = Field(description="One aggregated, comma-separated statement of every unique education level reported across these respondents.")

    class Config:
        populate_by_name = True

    class Config:
        populate_by_name = True


SYNTHESIS_RULES = (
    "Below are the stated views of several respondents who were grouped together because "
    "they share a similar behavioral/attitude profile. Write:\n"
    "1. short_description: a short phrase describing this persona's defining trait "
    "(e.g. 'challenger new tech adopter', 'debt-averse traditionalist').\n"
    "2. barrier_to_adoption: what these respondents' own words cite as reasons they "
    "haven't adopted or wouldn't adopt certain practices. If no barrier is stated, say so.\n"
    "3. source_of_information: collapse the raw 'Source of Information' values listed below into "
    "one aggregated, comma-separated statement of every unique source mentioned "
    "(e.g. 'Source of information: University extension, fertilizer vendors'). "
    "If none is reported, say 'Not reported'.\n"
    "4. education_summary: collapse the raw 'Education' values listed below into one aggregated, "
    "comma-separated statement of every unique education level mentioned "
    "(e.g. 'Education: 2-4 years of College education, diploma'). "
    "If none is reported, say 'Not reported'.\n"
    "Do not invent anything not implied by the text. Do not mention 'the group' or 'rows'; "
    "write as an outside analyst describing this persona."
)


def build_synthesis_chain(llm: ChatOpenAI):
    base_parser = PydanticOutputParser(pydantic_object=PersonaSynthesis)
    parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
    prompt = ChatPromptTemplate.from_template(
        "{rules}\n\n{format_instructions}\n\nRespondent statements:\n{group_text}"
    )
    return prompt | llm | parser, base_parser.get_format_instructions()


def collapse_demo_columns(rows: pd.DataFrame, demo_columns: List[str]) -> str:
    """Concatenates non-missing demo-column text across the cluster's rows
    into one cell. Swap for an LLM summarization call if the concatenation
    gets too long to be useful."""
    present_cols = [c for c in demo_columns if c in rows.columns]
    parts = []
    for col in present_cols:
        vals = [str(v).strip() for v in rows[col].tolist() if not is_missing(v)]
        deduped = list(dict.fromkeys(vals))
        if deduped:
            parts.append(f"{col}: " + " | ".join(deduped))
    return "\n".join(parts) if parts else "N/A"

def collect_column_values(rows: pd.DataFrame, column: str) -> str:
    """Dedupes and comma-joins every non-missing raw value for one column
    across a cluster's rows. Handed to the synthesis LLM as raw material
    to aggregate (Source of Information / Education), not pre-summarized
    here, since the LLM call is where SYNTHESIS_RULES asks for it."""
    if column not in rows.columns:
        return "Not reported"
    vals = [str(v).strip() for v in rows[column].tolist() if not is_missing(v)]
    deduped = list(dict.fromkeys(vals))
    return ", ".join(deduped) if deduped else "Not reported"


def build_personas(df: pd.DataFrame, row_labels: Dict[int, List[str]],
                    age_bracket: Dict[int, str], farm_size_bucket: Dict[int, str],
                    demo_columns: List[str], llm: ChatOpenAI) -> pd.DataFrame:
    df = df.copy()
    df["_age_bracket"] = df.index.map(lambda i: age_bracket.get(i, "UNKNOWN"))
    df["_farm_size_bucket"] = df.index.map(lambda i: farm_size_bucket.get(i, "UNKNOWN"))

    synth_chain, synth_fmt = build_synthesis_chain(llm)

    records = []
    persona_counter = 1
    for (age_b, size_b), block_df in df.groupby(["_age_bracket", "_farm_size_bucket"]):
        cluster_ids = cluster_block(block_df.index.tolist(), row_labels, CLUSTER_DISTANCE_THRESHOLD)
        block_df = block_df.copy()
        block_df["_cluster"] = block_df.index.map(cluster_ids)

        for cluster_id, cluster_rows in block_df.groupby("_cluster"):
            labels_union = sorted(set(l for i in cluster_rows.index for l in row_labels.get(i, [])))
            demo_text = collapse_demo_columns(cluster_rows, demo_columns)
            source_info_raw = collect_column_values(cluster_rows, SOURCE_INFO_COLUMN)
            education_raw = collect_column_values(cluster_rows, EDUCATION_COLUMN)
            statements = "\n\n".join(
                row_text_for_labeling(row) for _, row in cluster_rows.iterrows()
            )
            group_text = (
                statements
                + "\n\nDemographic context:\n" + demo_text
                + "\n\nRaw Source of Information values to aggregate:\n" + source_info_raw
                + "\n\nRaw Education values to aggregate:\n" + education_raw
            )

            try:
                synth: PersonaSynthesis = synth_chain.invoke({
                    "rules": SYNTHESIS_RULES,
                    "format_instructions": synth_fmt,
                    "group_text": group_text,
                })
                short_desc, barrier = synth.short_description, synth.barrier_to_adoption
                source_info_summary, education_summary = synth.source_of_information, synth.education_summary
            except Exception as e:
                with open(DEBUG_LOG, "a") as f:
                    f.write(f"\n\n=== persona synthesis FAILED (age={age_b}, size={size_b}, cluster={cluster_id}) ===\n{e}\n")
                short_desc, barrier = "SYNTHESIS_ERROR", "SYNTHESIS_ERROR"
                source_info_summary, education_summary = "SYNTHESIS_ERROR", "SYNTHESIS_ERROR"

            uids = list(dict.fromkeys(
                str(v).strip() for v in cluster_rows.get("Unique ID", pd.Series(dtype=str)).tolist()
                if not is_missing(v)
            ))
            states = list(dict.fromkeys(
                str(v).strip() for v in cluster_rows.get(STATE_COLUMN, pd.Series(dtype=str)).tolist()
                if not is_missing(v)
            ))

            pid = f"P{persona_counter:03d}"
            persona_counter += 1
            records.append({
                "Persona ID": pid,
                "Age Bracket": age_b,
                "Farm Size Bucket": size_b,
                "Labels": ", ".join(labels_union) if labels_union else "(none)",
                "N Source Rows": len(cluster_rows),
                "Unique IDs": ", ".join(uids) if uids else "(none)",
                "States": ", ".join(states) if states else "(none)",
                "Demo Context": demo_text,
                "Source of Information": source_info_summary,
                "Education": education_summary,
                "Short Description": short_desc,
                "Barrier to Adoption": barrier,
                "_row_indices": list(cluster_rows.index),
            })
            print(f"Built {pid} [{age_b} / {size_b}] labels={labels_union} n={len(cluster_rows)}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# STEP 6: match personas to counties
# ---------------------------------------------------------------------------
def persona_defining_categories(labels_str: str) -> Dict[str, str]:
    """{'regen_ag': 'HIGH', 'insurance': 'LOW', ...} -- only categories that
    have a census mapping (see CENSUS_MAPPED_CATEGORIES)."""
    if labels_str in ("(none)", ""):
        return {}
    out = {}
    for label in [l.strip() for l in labels_str.split(",")]:
        cat = label_category(label)
        if cat in CENSUS_MAPPED_CATEGORIES:
            out[cat] = label_direction(label)
    return out


def match_personas_to_counties(
    persona_df: pd.DataFrame, county_labels: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    summary_rows = []
    per_persona_tables: Dict[str, pd.DataFrame] = {}

    for _, prow in persona_df.iterrows():
        pid = prow["Persona ID"]
        defining = persona_defining_categories(prow["Labels"])

        if not defining:
            summary_rows.append({
                "Persona ID": pid, "Matched Counties": "N/A (no census-comparable labels)",
                "Best Match County": "N/A", "Best Match Score": None, "N Matched Counties": 0,
            })
            per_persona_tables[pid] = pd.DataFrame()
            continue

        scores = {}
        for county, row in county_labels.iterrows():
            hits = sum(1 for cat, direction in defining.items()
                       if row.get(cat) == direction)
            scores[county] = hits / len(defining)

        scores_series = pd.Series(scores).sort_values(ascending=False)
        matched = scores_series[scores_series >= MIN_MATCH_FRACTION]

        best_county = scores_series.index[0] if len(scores_series) else None
        best_score = float(scores_series.iloc[0]) if len(scores_series) else None

        summary_rows.append({
            "Persona ID": pid,
            "Matched Counties": ", ".join(matched.index.tolist()) if len(matched) else "(none above threshold)",
            "Best Match County": best_county,
            "Best Match Score": best_score,
            "N Matched Counties": len(matched),
        })

        table_rows = []
        for county in matched.index:
            row = {"County": county, "Match Score": scores[county]}
            for cat in defining:
                row[f"{cat}_label"] = county_labels.loc[county, cat]
            table_rows.append(row)
        per_persona_tables[pid] = pd.DataFrame(table_rows)

        print(f"{pid}: {len(matched)} matched counties, best={best_county} ({best_score:.2f})" if best_score is not None else f"{pid}: no counties")

    return pd.DataFrame(summary_rows), per_persona_tables


def attach_full_census_values(per_persona_tables: Dict[str, pd.DataFrame],
                               census_values: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Joins every matched county's full set of census rows (all items, not
    just the defining categories) onto its per-persona table, transposed so
    counties are rows and census items are columns. Uses CENSUS_RAW_XLSX if
    configured, otherwise the normalized z-scores."""
    out = {}
    for pid, table in per_persona_tables.items():
        if table.empty:
            out[pid] = table
            continue
        counties = table["County"].tolist()
        values_t = census_values[counties].T
        values_t.index.name = "County"
        merged = table.merge(values_t.reset_index(), on="County", how="left")
        out[pid] = merged
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    llm = ChatOpenAI(base_url=VLLM_BASE_URL, api_key="not-needed",
                      model="qwen_persona", max_tokens=32768)

    print(f"Loading RAG rows from {RAG_INPUT_XLSX}")
    df = pd.read_excel(RAG_INPUT_XLSX)
    print(f"Loaded {len(df)} rows")

    print("Step 1: labeling RAG rows...")
    row_labels = label_all_rows(df, llm, checkpoint_path=LABELS_CHECKPOINT_JSON)

    print(f"Loading census mapping from {CENSUS_MAPPING_XLSX}")
    mapping, demo_columns = load_census_mapping(CENSUS_MAPPING_XLSX)
    category_keys = {**RAG_COLUMN_TO_CATEGORY, **BLOCKING_RAG_COLUMNS}

    print(f"Loading normalized census table from {CENSUS_NORMALIZED_XLSX}")
    census_wide = load_census_wide(CENSUS_NORMALIZED_XLSX, CENSUS_NORMALIZED_SHEET)

    print("Step 3: building per-county category scores...")
    category_scores = build_category_scores(census_wide, mapping, category_keys)
    validate_category_scores(category_scores, category_keys, mapping, demo_columns)

    low_cut, high_cut = derive_farm_size_scale(category_scores)
    low_desc = f"roughly the bottom third of counties on aggregate farm acreage (z-score below {low_cut:.2f})"
    high_desc = f"roughly the top third of counties on aggregate farm acreage (z-score above {high_cut:.2f})"
    print(f"Farm size scale derived from census: low<{low_cut:.2f}, high>{high_cut:.2f}")

    print("Step 2: assigning age brackets and farm size buckets to RAG rows...")
    age_bracket = assign_age_brackets(df, llm)
    farm_size_bucket = assign_farm_size_buckets(df, llm, low_desc, high_desc)

    print("Step 4: clustering RAG rows within each block and synthesizing personas...")
    persona_df = build_personas(df, row_labels, age_bracket, farm_size_bucket, demo_columns, llm)

    print("Step 5: labeling counties HIGH/LOW per census-mapped category...")
    county_labels = label_counties_high_low(category_scores, CENSUS_MAPPED_CATEGORIES)

    print("Step 6: matching personas to counties...")
    summary_df, per_persona_tables = match_personas_to_counties(persona_df, county_labels)

    # If CENSUS_RAW_XLSX resolves to a real file, its values are shown in the
    # per-persona tables instead of the normalized z-scores; the sheet-name
    # fallback in load_census_wide handles it even if its sheet isn't named
    # the same as CENSUS_NORMALIZED_SHEET.
    census_values_for_display = (
        load_census_wide(CENSUS_RAW_XLSX, CENSUS_NORMALIZED_SHEET) if CENSUS_RAW_XLSX else census_wide
    )
    per_persona_tables = attach_full_census_values(per_persona_tables, census_values_for_display)

    print(f"Writing output to {OUTPUT_XLSX}")
    persona_export = persona_df.drop(columns=["_row_indices"])
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        persona_export.to_excel(writer, sheet_name="Persona_Summary", index=False)
        summary_df.to_excel(writer, sheet_name="Match_Summary", index=False)
        county_labels.to_excel(writer, sheet_name="County_Category_Labels")
        for pid, table in per_persona_tables.items():
            sheet = f"Persona_{pid}"[:31]
            table.to_excel(writer, sheet_name=sheet, index=False)

    print("Done.")


if __name__ == "__main__":
    main()