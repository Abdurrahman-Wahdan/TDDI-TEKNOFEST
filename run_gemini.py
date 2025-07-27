from config.env_service import get_env
from llm_providers.google.gemini_provider import get_gemini_provider
from llm_providers.interfaces.base_llm import LLMConfig, LLMProviderType

# Test getting API key with env_service
gemini_key = get_env("GEMINI_API_KEY", is_sensitive=True)
print(f"Got API key: {'✅' if gemini_key else '❌'}")

# Test Gemini 2.5 Flash with thinking disabled
provider = get_gemini_provider(api_key=gemini_key)

config = LLMConfig(
    provider=LLMProviderType.GEMINI,
    model_name="gemma-3-27b-it",
    temperature=0.7,
    max_token=100
)

model = provider.get_langchain_model(config)

# Test inference
response = model.invoke("selam")
print(f"Response: {response.content}")