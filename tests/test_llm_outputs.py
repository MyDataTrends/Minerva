import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.llm_analyzer import analyze_dataset
from preprocessing.llm_summarizer import generate_summary


def test_analyze_dataset_output(monkeypatch):
    monkeypatch.setattr(
        'preprocessing.llm_analyzer.analyze_dataset_with_llm',
        lambda df: 'analysis text'
    )
    result = analyze_dataset(None)
    assert result == {'summary': 'analysis text', 'artifacts': {}}


def test_generate_summary_output(monkeypatch):
    monkeypatch.setattr(
        'preprocessing.llm_summarizer.load_mistral_model',
        lambda path: None
    )
    monkeypatch.setattr(
        'preprocessing.llm_summarizer.run_mistral_inference',
        lambda model, input, max_tokens, temperature: 'summary text'
    )
    out = generate_summary({}, {}, 'prompt')
    assert out['summary'] == 'summary text'
    assert out['artifacts']['data_stats'] == {}
    assert out['artifacts']['model_results'] == {}

