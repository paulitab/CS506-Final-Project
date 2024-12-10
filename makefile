# Makefile for running the final_project_code.ipynb notebook

.PHONY: install run clean

# Install necessary dependencies
install:
	pip install pandas numpy matplotlib seaborn wordcloud tqdm \
		sentence-transformers scikit-learn notebook tf-keras plotly pytest

# Run the Jupyter Notebook
run:
	jupyter nbconvert --to notebook --execute final_project_code.ipynb --output final_project_output.ipynb

# Run the test file
test:
	pytest -s test_final_project.py

# Clean up generated files
clean:
	rm -f final_project_output.ipynb
