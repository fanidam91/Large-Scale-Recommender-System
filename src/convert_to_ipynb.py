import json
import os

def convert_file(py_filename, ipynb_filename):
    py_path = os.path.join("notebooks", py_filename)
    ipynb_path = os.path.join("notebooks", ipynb_filename)
    
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
                clean_line = line.replace("# MAGIC %md", "").replace("# MAGIC", "").strip()
                markdown_lines.append(clean_line)
            else:
                code_lines.append(line)
        
        if is_markdown:
            source_lines = [line + "\n" for line in markdown_lines]
            if source_lines:
                source_lines[-1] = source_lines[-1].rstrip("\n")
                
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": source_lines
            })
        else:
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
        
    print(f"Successfully converted {py_filename} -> {ipynb_filename}")

def py_to_ipynb():
    # Convert standard version
    convert_file("databricks_recommender_pipeline_source.py", "databricks_recommender_pipeline.ipynb")
    # Convert Unity Catalog Volume version
    convert_file("databricks_recommender_pipeline_volume_source.py", "databricks_recommender_pipeline_volume.ipynb")

if __name__ == "__main__":
    py_to_ipynb()
