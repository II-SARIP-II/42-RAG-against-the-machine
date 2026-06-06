FLK 	:= flake8
MYPY 	:= mypy
FLAGS	:= --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
DEP 	:= pydantic fire langchain-text-splitters bm25s PyStemmer numpy tqdm dspy chromadb flake8 mypy

export UV_CACHE_DIR := $(shell echo /goinfre/pgougne/Ragcache)

install:
	uv python pin 3.11
	uv add $(DEP)

run:
	@uv run python -m src index

index:
	@uv run python -m src index --chroma=True

search:
	@uv run python -m src search --k=5 --prompt="What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --chroma=False --save_directory="data/output/search_results" --expansion=False

search-dataset:
	@uv run python -m src search_dataset --k=5 --dataset_path=datasets_public/public/AnsweredQuestions/dataset_docs_public.json --chroma=True --expansion=False

answer:
	@uv run python -m src answer "What activation formats does the fused batched MoE layer return in vLLM?"

answer-dataset:
	@uv run python -m src answer_dataset --k=5 --prompts_file=data/output/search_results/dataset_code_public.json

evaluate:
	@uv run python -m src evaluate --dataset_path=data/output/search_results/dataset_docs_public.json --max_context_length=2000 --answer_path=datasets_public/public/AnsweredQuestions/dataset_docs_public.json --k=5

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

.PHONY: install run debug clean fclean re lint lint-strict venv init index search search-dataset answer answer-dataset evaluate
