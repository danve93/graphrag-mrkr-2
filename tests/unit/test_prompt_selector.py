import asyncio
import json
from pathlib import Path

from rag.nodes.prompt_selector import PromptSelector


def test_prompt_selector_prefers_step_back_and_instructions(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text(
        json.dumps(
            {
                "default": {
                    "retrieval_strategy": "balanced",
                    "generation_template": "Answer:\n{context}\nQ:{query}",
                    "format_instructions": "Default format",
                },
                "installation": {
                    "retrieval_strategy": "step_back",
                    "generation_template": "Install guide: {query}\n{context}",
                    "format_instructions": "Bullet steps",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("rag.nodes.prompt_selector.settings.enable_category_prompt_instructions", True, raising=False)

    selector = PromptSelector(prompts_file=str(prompts_path))
    strategy = selector.get_retrieval_strategy(["installation", "other"])

    assert strategy == "step_back"

    prompt = asyncio.run(
        selector.select_generation_prompt(
            query="How do I install Neo4j?",
            categories=["installation"],
            context="ctx",
            conversation_history=[{"role": "user", "content": "prev"}],
        )
    )

    assert "Install guide" in prompt
    assert "Format instructions: Bullet steps" in prompt
    assert "Previous conversation:" in prompt


def test_prompt_selector_fallback_to_default_when_missing(monkeypatch, tmp_path):
    # Missing prompts file should use default template
    missing_path = tmp_path / "does_not_exist.json"
    monkeypatch.setattr("rag.nodes.prompt_selector.settings.enable_category_prompt_instructions", False, raising=False)

    selector = PromptSelector(prompts_file=str(missing_path))
    prompt = asyncio.run(selector.select_generation_prompt(query="Q?", categories=None, context="CTX"))

    assert "Context" in prompt
