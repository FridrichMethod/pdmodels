# Protein Design Models

```python
pdmodels/
│
├── __init__.py
│
├── globals.py  # global variables
├── types.py  # type definitions
├── basemodels.py  # base classes for all models
│
├── af2ig.py  # AlphaFold 2 and AlphaFold-Multimer implementation with initial guess of MSA and templates
├── esmfold.py  # ESMFold implementation
│
├── esm2.py  # ESM2 with batch single sequence scoring function
├── esmif.py  # ESM-IF batch version
├── mpnn.py  # ProteinMPNN and LigandMPNN batch version
│
└── utils.py  # utility functions
```