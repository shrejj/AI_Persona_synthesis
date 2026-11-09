# Repository Guidelines

This repository contains Python scripts for RAG-based persona matching to USDA NASS census counties and PDF paper table extraction.

## Project Structure

- Census_cluster_grouping.py — Matches RAG personas to census counties using affinity labels, clustered by age and farm size.
- RAG_newcolumns.py — Extracts structured fields from PDFs using a locally-served Qwen model and writes results to Excel.
- .venv/ — Python virtual environment (do not commit).

## Development Commands

Activate the virtual environment before running scripts:

`powershell
.venv\Scripts\Activate.ps1
`

Install dependencies:

`powershell
pip install pdfplumber langchain langchain-openai pandas openpyxl pydantic
`

Run a script:

`powershell
python Census_cluster_grouping.py
python RAG_newcolumns.py
`

## Coding Style & Naming Conventions

- Use 4-space indentation.
- Module names use snake_case.py.
- Functions and variables use snake_case; classes use PascalCase.
- Keep docstrings at the top of each module describing purpose and steps.
- No formatter or linter is currently configured; follow the existing style.

## Testing Guidelines

No test framework or test directory is present. Verify changes by running the scripts directly and inspecting the Excel/console output.

## Commit & Pull Request Guidelines

- Write concise, imperative commit messages (e.g., "Add farm-size bucketing logic").
- Reference linked issues in commit bodies when applicable.
- Pull requests should include a clear description of changes and any affected output files.

## Agent-Specific Instructions

- Do not commit files inside .venv/.
- Run scripts from the repository root with the venv activated.
- Update this AGENTS.md if you add new scripts or dependencies.
