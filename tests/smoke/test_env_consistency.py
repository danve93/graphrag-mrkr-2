import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_env_example_contains_keys():
    env_example = read_file(ROOT / ".env.example")
    assert "OPENAI_MODEL" in env_example, "OPENAI_MODEL missing from .env.example"
    assert "NEO4J_URI" in env_example, "NEO4J_URI missing from .env.example"
    assert "EMBEDDING_MODEL" in env_example, "EMBEDDING_MODEL missing from .env.example"


def test_setup_sh_generates_matching_openai_model():
    setup = read_file(ROOT / "setup.sh")
    env_example = read_file(ROOT / ".env.example")

    # Find first OPENAI_MODEL assignment in .env.example
    m1 = re.search(r"^OPENAI_MODEL=(\S+)", env_example, re.M)
    assert m1, "OPENAI_MODEL not found in .env.example"
    env_model = m1.group(1).strip()

    # Find first OPENAI_MODEL assignment in setup.sh generated block (if present)
    m2 = re.search(r"OPENAI_MODEL=(\S+)", setup)
    if m2:
        setup_model = m2.group(1).strip()
        assert setup_model == env_model, (
            f"OPENAI_MODEL mismatch: setup.sh -> {setup_model}, .env.example -> {env_model}"
        )
    else:
        # If setup.sh does not include OPENAI_MODEL (e.g., external template), it's OK
        assert True


def test_no_obvious_hardcoded_demo_passwords():
    # Ensure repo doesn't contain the demo plaintext 'neo4jpassword' used previously
    found = False
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in {".md", ".sh", ".env", ".txt"}:
            txt = p.read_text(errors="ignore")
            if "neo4jpassword" in txt:
                found = True
                break
    assert not found, "Found hardcoded 'neo4jpassword' in repository files"
