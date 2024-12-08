# Variables
PYTHON = /usr/local/bin/python3.10
PIP = $(PYTHON) -m pip
NOTEBOOK = final_project_code.ipynb
OUTPUT_NOTEBOOK = final_output.ipynb
DEPENDENCIES = pandas numpy scikit-learn keras matplotlib sentence-transformers tqdm seaborn wordcloud jupyter

# Default target
.PHONY: all
all: install run-notebook

# Install dependencies
.PHONY: install
install:
	$(PIP) install $(DEPENDENCIES)

# Run the Jupyter Notebook
.PHONY: run-notebook
run-notebook:
	jupyter nbconvert --to notebook --execute $(NOTEBOOK) --output $(OUTPUT_NOTEBOOK)
