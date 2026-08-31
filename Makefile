# Entry points for every stage. PYTHON selects the interpreter.
PYTHON ?= python

.PHONY: help test preprocess sweep arms merge crossyear c5 figures figure1 all clean

help:
	@echo "test        run the leakage and event-dating tests (needs data/raw)"
	@echo "preprocess  build the processed tensors from data/raw"
	@echo "sweep       the truncated sweep, 3,080 fits"
	@echo "arms        ROCKET, catch22, TEASER and the majority floor"
	@echo "merge       collect array-job shards into results/"
	@echo "crossyear   the 2018 transfer evaluation"
	@echo "c5          the released full-season model under truncation"
	@echo "figures     rebuild the five figures that need only results/"
	@echo "figure1     rebuild Figure 1, which also needs data/raw"
	@echo "all         every stage, end to end"

test:
	$(PYTHON) -m pytest tests/ -q

preprocess:
	$(PYTHON) scripts/build_processed.py

sweep:
	$(PYTHON) -c "from sgl import config, classify; classify.sweep(config.load(), 'results/sweep_c1.parquet')"

arms:
	$(PYTHON) scripts/simple_baseline.py
	$(PYTHON) scripts/run_majority.py
	@echo "ROCKET, catch22 and TEASER are array jobs; see slurm/README.md,"
	@echo "then run: make merge"

merge:
	$(PYTHON) scripts/merge_shards.py

crossyear:
	$(PYTHON) scripts/crossyear.py

c5:
	$(PYTHON) scripts/run_c5.py

# Five of the six figures need only results/; Figure 1 also needs data/raw.
figures:
	$(PYTHON) -c "import sys; sys.path.insert(0,'figures'); \
	from code import paperfigs, fig_pipeline; \
	paperfigs.leadtime(); paperfigs.errormodes(); paperfigs.incumbent(); \
	paperfigs.generalisation(); fig_pipeline.build()"

figure1:
	$(PYTHON) -c "import sys; sys.path.insert(0,'figures'); \
	from code import fig_classes; fig_classes.classes()"

all: preprocess sweep arms crossyear c5 figures figure1

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
