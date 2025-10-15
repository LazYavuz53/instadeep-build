## Layout of the SageMaker ModelBuild Project Template

TODO UPDATE ET THE END

```
├── codebuild-buildspec.yml      # Definition used by AWS CodeBuild to run CI jobs
├── CONTRIBUTING.md              # Contribution guidelines for collaborators
├── data_analysis/               # Utilities and cleaning scripts for exploratory analysis
├── developer.md                 # Developer-focused setup and workflow notes
├── img/                         # Images referenced in documentation and presentations
├── pipelines/                   # Pipeline package and entry points
│   ├── InstaDeepMHCIPresentation/
│   │   ├── evaluate.py          # Model evaluation step definition
│   │   ├── pipeline.py          # Orchestrates the full SageMaker pipeline
│   │   ├── preprocess.py        # Data preprocessing logic used by the pipeline
│   │   └── __init__.py
│   ├── __init__.py
│   ├── __version__.py
│   ├── _utils.py
│   ├── get_pipeline_definition.py
│   ├── run_pipeline.py
│   └── train.py
├── requirements.txt             # Python dependencies for local development and testing
├── sagemaker-pipelines-project.ipynb  # Notebook describing the SageMaker project setup
├── setup.cfg                    # Package metadata and configuration for tooling
├── setup.py                     # Package installation entry point
├── tests/
│   └── test_pipelines.py        # Unit tests for the pipeline package
├── tox.ini                      # Tox environments for running tests and linters
└── training_notebook/           # Assets for the interactive model training notebook
    ├── full_training.ipynb
    ├── clean/
    ├── models/
    └── outputs/
```

TODO