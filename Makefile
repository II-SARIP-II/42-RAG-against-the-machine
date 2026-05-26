FLK 	:= flake8
MYPY 	:= mypy
FLAGS	:= --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

install:
	uv add pydantic fire langchain-text-splitters bm25s PyStemmer numpy tqdm ollama dspy

run:
	@uv run python -m src index

index:
	@uv run python -m src index

search:
	@uv run python -m src search "What activation formats does the fused batched MoE layer return in vLLM?"

search-dataset:
	@uv run python -m src search_dataset datasets_public/public/AnsweredQuestions/dataset_code_public.json

answer:
	@uv run python -m src answer "What activation formats does the fused batched MoE layer return in vLLM?"

answer-dataset:
	@uv run python -m src answer_dataset data/output/search_results/dataset_code_public.json

evaluate:
	@uv run python -m src evaluate


redicCache:
	export UV_CACHE_DIR="~/goinfre/Ragcache"

debug:
	@uv run python -m pdb src/__main__.py

clean:
	rm -rf __pycache__ .mypy_cache .python-version .vscode
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	$(FLK) . --extend-exclude .venv,vllm-0.10.1
	$(MYPY) . $(FLAGS)

lint-strict:
	$(FLK) . --extend-exclude .venv,vllm-0.10.1
	$(MYPY) . $(FLAGS) --strict

venv:
	uv venv

init:
	uv init

.PHONY: install, run, debug, clean, lint, lint-strict, venv, init, index, search, search_dataset, answer, answer_dataset, evaluate
