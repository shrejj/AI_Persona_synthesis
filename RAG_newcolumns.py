"""
extract_papers_table_langchain.py

Reads every PDF in a folder, asks your locally-served Qwen model to pull
out structured fields from each one, and writes the results to an Excel table.

Run this on the SAME compute node where `vllm serve` is running (or one that
can reach it over the network), with your venv activated.

Install deps first:
    pip install pdfplumber langchain langchain-openai pandas openpyxl pydantic
"""

import os
import glob
import time
from typing import List

import torch
import pandas as pd
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# ---- CONFIG: edit these ----
PDF_FOLDER = "/scratch/adriaens_root/adriaens1/shrejj/Included"
OUTPUT_XLS = "/scratch/adriaens_root/adriaens1/shrejj/results_RAG6.xlsx"
CHECKPOINT_CSV = "/scratch/adriaens_root/adriaens1/shrejj/results_RAG6.csv"
DEBUG_LOG = "/scratch/adriaens_root/adriaens1/shrejj/debug_raw_responses6.log"
VLLM_BASE_URL = "http://localhost:8000/v1"

# Cap on characters pulled from each PDF before chunking/embedding.
# 160,000,000 (the old value) is effectively "no cap" and lets huge PDFs
# blow up chunk count, and therefore embedding time, per document.
MAX_PDF_CHARS = 300000


# ---- SCHEMA ----
# This `alias=` is exactly the column name that ends up in the
# spreadsheet, and it's also what the model is told to produce.
class PaperRow(BaseModel):
    unique_id: str = Field(alias="Unique ID")
    study_sample_size: str = Field(alias="Study Sample Size")
    pub_year: str = Field(alias="Pub Year")
    crop_types_grown: str = Field(alias="Crop Types Grown")
    farm_size: str = Field(alias="Farm Size")
    state_county: str = Field(alias="State / County")
    avg_age: str = Field(alias="Avg. Age")
    avg_nationality: str = Field(alias="Avg. Nationality")
    avg_ethnicity_gender: str = Field(alias="Avg Ethnicity,Gender, race, culture")
    avg_education: str = Field(alias="Avg Education")
    info_source: str = Field(alias="Farmer Informant")
    view_regen_ag: str = Field(alias="View on Regenerative Agriculture (Cover crop, No-Till)")
    view_farming_ops: str = Field(alias="Views on farming operations and practices")
    view_fertilizer: str = Field(alias="View on Fertilizer Use")
    farm_finances: str = Field(alias="Farm Finances Perspective")
    view_insurance: str = Field(alias="Views on Insurance")
    view_environment: str = Field(alias="Views on environment")
    view_price_risk: str = Field(alias="Views on Price Risk Mgmt")
    view_carbon: str = Field(alias="Views on Carbon Markets")

    class Config:
        populate_by_name = True


class PaperRows(BaseModel):
    rows: List[PaperRow]


# ---- RAG SETUP ----
# vLLM already reserves ~95% of memory on both GPUs for the LLM itself
# (--gpu-memory-utilization 0.95 --tensor-parallel-size 2), so a second process trying to grab a CUDA context for the embedding model can throw "CUDA-capable device(s) is/are busy or unavailable". Try GPU, fall back
# Try GPU, fall back to CPU cleanly instead of crashing the run.
EMBED_DEVICE = "cpu"
if torch.cuda.is_available():
    try:
        torch.zeros(1).cuda()
        EMBED_DEVICE = "cuda"
    except Exception as e:
        print(f"CUDA available but unusable for embeddings ({e}). Falling back to CPU.")
print(f"Embedding device: {EMBED_DEVICE}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": EMBED_DEVICE},
)



def build_retriever(text, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)

    # vector (semantic) retriever
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # keyword (exact-match) retriever
    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = 4

    # combine both, weighting semantic search slightly higher
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )
    return ensemble_retriever

FIELD_QUERIES = {
    "sample size": "survey sample size",
    "Age": "respondent age, average age, older, younger",
    "Nationality": "American, Mexican, european",
    "ethnicity": "latin, white, black, asian, native american, culture",
    "education": "farmer education level, graduate degree, high school, college, vocational, specialzation, degree, diploma, certificate, training",
    "Farmer Informant": "Information sources that farmers being studied rely on for farm operations like fertilizer, crop rotation, subsidy information, finances and technology such as University consulting, extension services, consultants, financial advisors, crop advisors, fertilizer suppliers, salesmen, government conservation agencies, neighbors, peer groups, farmer unions, magazines, academic publications",
    "farming_practices": "crop types grown, farm size, regenerative agriculture, no-till, cover crops, crop rotation, tilling, sowing, reaping, strategy, machinery, labor, livestock, land management",
    "finance_and_risk": "debt-asset ratio, agriculture markets, sales price, cost price, profit, loss, farm finances, insurance, price risk management, carbon markets, crop sales markets, return on investment, forecasting, predicting, budgeting, profits, losses, revenue, expenses, additional income sources",
    "environment": "views on environment, global warming, drought, flood, contamination, pollution, water quality, soil quality, soil erosion, soil health, soil fertility, soil degradation, soil conservation, soil management",
    "decision-making": "behavioural changes due to an observation, adoption, shift in practices, change in operations, change in finances, change in crop rotation, change in technology, buying, selling, leasing, investing",
}

def build_constrained_context(retriever):
    seen = set()
    context_parts = []
    for query in FIELD_QUERIES.values():
        for doc in retriever.invoke(query):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                context_parts.append(doc.page_content)
    return "\n\n".join(context_parts)

# ---- MODEL + PARSER SETUP (created once, reused for every PDF) ----
llm = ChatOpenAI(
    base_url=VLLM_BASE_URL,
    api_key="not-needed",
    model="qwen_persona",
    max_tokens=32768,
    temperature=0,
)

base_parser = PydanticOutputParser(pydantic_object=PaperRows)
parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

prompt = ChatPromptTemplate.from_template(
    "{rules}\n\n{format_instructions}\n\nPaper text:\n{paper_text}"
)

chain = prompt | llm | parser

FORMAT_INSTRUCTIONS = base_parser.get_format_instructions()

RULES_TEXT = ( #updated as of 8.31
    "Rule 1: 1-20 entries per source. One entry per unique decision-making behavior.\n"
    "Rule 2: Do not add information absent/not present from the source. All cells in a row must refer to the same group.\n"
    "Rule 3a: Each row is a group or individual with a distinct decision-making behavior in regen-ag, environment, fertilizer, financial, insurance, price-risk, or carbon-market perspectives, directly sourced from the text. \n"
    "Rule 3b: If the literature has no information, the cell must be \"N/A\" only. Statements saying Data unavailable should be rewritten as N/A\n"
    "Rule 4: Study sample size is the total or subset reported for that specific observation.\n"
    "Rule 5: Pub Year, Crop Types Grown, Farm Size, State / County, Avg. Age, and Avg. Nationality are objective and can never be \"N/A\".\n"
    "Rule 6: Every non-\"N/A\" view is a personified statement for each unique response in the literature.\n"
    "Rule 7: Highlight any transitions in crops, practices, or finances as examples in the perspective/decision-making columns. Format: \"example: ...\"\n"
    "Rule 8: Minimum 8 words for: view on regenerative agriculture (cover crops, no till), views on fertilizer use, farm finances perspective, views on insurance, views on environment, views on price risk management, views on carbon markets.\n"
    "Rule 9: Remove grower names and identifying information. Use Unique IDs: one consistent identifier per paper, with unique trailing initials per row.\n"
    "Rule 10: For Farmer Informant, exclude the citation source. Include only the source farmers receive their operations/finance recommendations from.\n"
)


def extract_text(pdf_path, max_chars=MAX_PDF_CHARS):
    """Same as RAG_enhanced_extract.extract_text -- duplicated here so this
    script has no import-time dependency on that file."""
    import pymupdf4llm
    text = pymupdf4llm.to_markdown(pdf_path)
    return text[:max_chars]


def process_pdf(path):
    """Run one PDF through the chain and return a list of dict rows."""
    t0 = time.time()
    raw_text = extract_text(path)
    t1 = time.time()
    retriever = build_retriever(raw_text)
    t2 = time.time()
    text = build_constrained_context(retriever)
    t3 = time.time()

    print(
        f"    extract_text: {t1 - t0:.1f}s | build_retriever: {t2 - t1:.1f}s "
        f"| build_context: {t3 - t2:.1f}s | chars_extracted: {len(raw_text)} "
        f"| chars_in_context: {len(text)}",
        flush=True,
    )

    try:
        result: PaperRows = chain.invoke({
            "rules": RULES_TEXT,
            "format_instructions": FORMAT_INSTRUCTIONS,
            "paper_text": text,
        })
    except Exception as e:
        # Log the failure and log it in a way you can inspect later,
        # then return a single PARSE_ERROR row instead of crashing the run.
        with open(DEBUG_LOG, "a") as f:
            f.write(f"\n\n=== {os.path.basename(path)} FAILED ===\n{e}\n")
        error_row = {field.alias: "PARSE_ERROR" for field in PaperRow.model_fields.values()}
        return [error_row]
    finally:
        t4 = time.time()
        print(f"    llm chain.invoke: {t4 - t3:.1f}s", flush=True)

    with open(DEBUG_LOG, "a") as f:
        f.write(f"\n\n=== {os.path.basename(path)} OK ===\n{result.model_dump_json(by_alias=True, indent=2)}\n")

    return [row.dict(by_alias=True) for row in result.rows]


def main():
    pdf_files = sorted(glob.glob(os.path.join(PDF_FOLDER, "*.pdf")))
    print(f"Found {len(pdf_files)} PDFs in {PDF_FOLDER}")

    # Resume support: skip files already recorded in the checkpoint CSV,
    # so a job that gets killed by SLURM's time limit doesn't have to
    # restart from PDF #1.
    already_done = set()
    if os.path.exists(CHECKPOINT_CSV):
        prior = pd.read_csv(CHECKPOINT_CSV)
        already_done = set(prior["Source File"].unique())
        print(f"Resuming: {len(already_done)} files already in checkpoint, skipping them.")

    rows = []
    for i, path in enumerate(pdf_files, start=1):
        name = os.path.basename(path)
        if name in already_done:
            print(f"[{i}/{len(pdf_files)}] Skipping {name} (already done)")
            continue

        print(f"[{i}/{len(pdf_files)}] Processing {name}")
        t_start = time.time()
        parsed_rows = process_pdf(path)
        print(f"    total: {time.time() - t_start:.1f}s", flush=True)

        for row in parsed_rows:
            row["Source File"] = name
            rows.append(row)

        # Write this PDF's rows to the checkpoint immediately, so progress
        # survives a timeout instead of only being saved at the very end.
        pd.DataFrame(parsed_rows).assign(**{"Source File": name}).to_csv(
            CHECKPOINT_CSV,
            mode="a",
            header=not os.path.exists(CHECKPOINT_CSV),
            index=False,
        )

    # Rebuild the final Excel from the full checkpoint (covers this run
    # plus any prior resumed runs), not just rows collected in this process.
    df = pd.read_csv(CHECKPOINT_CSV) if os.path.exists(CHECKPOINT_CSV) else pd.DataFrame(rows)
    cols = ["Source File"] + [c for c in df.columns if c != "Source File"]
    df = df[cols]
    df.to_excel(OUTPUT_XLS, index=False)
    print(f"\nDone. Table saved to {OUTPUT_XLS}")


if __name__ == "__main__":
    main()