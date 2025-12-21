
import os
import sys

# Ensure we can import from core
sys.path.append(os.getcwd())

from config.settings import settings
from core.llm import LLMManager
from core.embeddings import EmbeddingManager
import google.generativeai as genai

def load_env_file(filepath=".env"):
    """Manually load .env file into os.environ"""
    if not os.path.exists(filepath):
        return
    print(f"Loading environment from {filepath}")
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Basic quote handling
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                os.environ[key] = value

def test_gemini_integration():
    print("--- Verifying Gemini Integration ---")
    
    # Load .env explicitly
    load_env_file()
    
    # Reload settings to pick up new env vars if needed
    from config.settings import settings
    # Pydantic BaseSettings might cache the environment at import time.
    # We might need to manually set the fields on the singleton if it was already imported.
    # But let's check settings first.
    
    # Check API Key from settings (loaded from .env)
    api_key = settings.gemini_api_key
    if not api_key:
        # Fallback check
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in settings or environment.")
        return
    
    # Valid key found
    print(f"✅ API Key found in settings (ends with ...{str(api_key)[-4:]})")
    
    # Configure genai explicitly to list models
    genai.configure(api_key=api_key)
    
    print("\n--- Available Models ---")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f" - {m.name}")
    except Exception as e:
        print(f"❌ Failed to list models: {e}")

    # Override settings for this test
    settings.llm_provider = "gemini"
    # Inject key explicitly into settings since it might have been missed during init
    settings.gemini_api_key = api_key
    
    # Try to detect a valid model if default fails
    settings.gemini_model = "gemini-3-flash-preview" 
    
    # Test LLM
    print(f"\nTesting LLM Generation ({settings.gemini_model})...")
    try:
        llm = LLMManager()
        # Ensure it initialized with Gemini
        if llm.provider != "gemini":
            print(f"⚠️ Warning: LLMManager provider is {llm.provider}, expected gemini. Forcing...")
            llm.provider = "gemini"
            llm._init_gemini_client()
            
        response = llm.generate_response("Hello, say 'Gemini is working' if you can hear me.")
        print(f"✅ LLM Response: {response}")
    except Exception as e:
        print(f"❌ LLM Error: {e}")

    # Test Embeddings
    print("\nTesting Embedding Generation (models/text-embedding-004)...")
    try:
        # EmbeddingManager reads settings at init. We need to patch the singleton or creating a new instance might not be enough if it uses global settings that we just modified?
        # settings object is a singleton instance of Settings class, so modifying it above should work for new instances.
        
        embedder = EmbeddingManager()
        # Force provider if settings didn't take (EmbeddingManager logic might be sticky if we rely on cached properties, but it's a simple init)
        if embedder.provider != "gemini":
             print(f"⚠️ Warning: EmbeddingManager provider is {embedder.provider}, expected gemini. Forcing...")
             embedder.provider = "gemini"
             embedder.model = settings.gemini_embedding_model
             embedder._init_gemini_client()

        vector = embedder._generate_gemini_embedding("This is a test sentence for embedding.")
        
        if vector and len(vector) > 0:
            print(f"✅ Embedding generated. Dimension: {len(vector)}")
            print(f"   Sample: {vector[:3]}...")
        else:
            print("❌ Embedding generated but empty.")
            
    except Exception as e:
        print(f"❌ Embedding Error: {e}")

if __name__ == "__main__":
    try:
        test_gemini_integration()
    except KeyboardInterrupt:
        print("\nTest interrupted.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
