# Interactive Workflow Demo

This mini page visualizes the data automation pipeline implemented in this repository. The demo can be hosted on GitHub Pages or any static file server.

## Usage

1. Open `index.html` in your browser or deploy the `interactive-workflow-demo/` folder to GitHub Pages.
2. Click **Start Demo** to see each processing stage appear in sequence.

The animation mirrors the real workflow:

1. **Upload & Preprocess** – datasets are loaded and cleaned by `DataPreprocessor`.
2. **Semantic Enrichment** – `SemanticEnricher` joins public data and tags the upload.
3. **Model Training** – `AnalyzerSelector` trains and evaluates models.
4. **Generate Dashboard** – `OutputGenerator` saves results and calls `orchestrate_dashboard`.
5. **Trigger Actions** – `AgentTrigger` executes notifications and other recipes.
