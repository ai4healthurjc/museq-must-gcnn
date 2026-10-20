# Data Processing Pipeline

This part of the repository contains the scripts and notebooks used to build 
graph-based representations of both **static** and **temporal** ICU data.  
It is organized in two main folders: `static_data` and `temporal_data`.  
Each follows a two-step workflow: (1) graph estimation and (2) graph representation.

---

## Workflow

1. **Graph Estimation (step1)**  
   - Use the provided scripts (`hgd.py`, `graph_estimation.ipynb`) to estimate 
     patient similarity graphs based on static or temporal features.  
   - Outputs are stored in the `estimatedGraphs/` folder.

2. **Graph Representation (step2)**  
   - The resulting graphs are transformed into structured representations suitable 
     for downstream machine learning.  
   - Notebooks such as `graphRepresentation.ipynb` and 
     `graphRepresentation_as_STG.ipynb` show how to build these inputs.

---

