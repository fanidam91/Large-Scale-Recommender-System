import json
import os

def py_to_ipynb():
    py_path = os.path.join("notebooks", "databricks_recommender_pipeline.py")
    ipynb_path = os.path.join("notebooks", "databricks_recommender_pipeline.ipynb")
    
    if not os.path.exists(py_path):
        print(f"Error: Source file {py_path} not found.")
        return

    print(f"Reading {py_path}...")
    with open(py_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by Databricks cell separator
    cells_raw = content.split("# COMMAND ----------")
    
    cells = []
    for cell_raw in cells_raw:
        cell_raw = cell_raw.strip()
        if not cell_raw:
            continue
            
        lines = cell_raw.split("\n")
        
        # Check if it is a markdown cell
        is_markdown = False
        markdown_lines = []
        code_lines = []
        
        for line in lines:
            if line.startswith("# MAGIC %md") or line.startswith("# MAGIC"):
                is_markdown = True
                # Extract markdown text: strip leading comment and MAGIC annotations
                clean_line = line.replace("# MAGIC %md", "").replace("# MAGIC", "").strip()
                markdown_lines.append(clean_line)
            else:
                code_lines.append(line)
        
        if is_markdown:
            # Reconstruct markdown block
            # If line is empty, represent it, else keep it
            source_lines = [line + "\n" for line in markdown_lines]
            # Strip trailing newline from the last line to look clean in JSON
            if source_lines:
                source_lines[-1] = source_lines[-1].rstrip("\n")
                
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": source_lines
            })
        else:
            # Code cell
            # Remove leading databricks tags if any
            source_lines = [line + "\n" for line in code_lines]
            if source_lines:
                source_lines[-1] = source_lines[-1].rstrip("\n")
                
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source_lines
            })

    # Notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    print(f"Writing Jupyter notebook to {ipynb_path}...")
    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
        
    print("Conversion successful!")

if __name__ == "__main__":
    py_to_ipynb()
