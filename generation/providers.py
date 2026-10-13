"""Provider adapters; direct google-genai remains the compatibility baseline."""

import json


class LangChainGeminiProvider:
    """Optional current LangChain adapter using Pydantic structured output."""

    def __init__(self, api_key, model_name, temperature=0.2):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, prompt, schema):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("LangChain Gemini integration is not installed") from exc
        model = ChatGoogleGenerativeAI(
            model=self.model_name, google_api_key=self.api_key, temperature=self.temperature,
        )
        result = model.with_structured_output(schema).invoke(prompt)
        return result.model_dump() if hasattr(result, "model_dump") else result


class DirectGeminiJsonProvider:
    """Compatibility adapter for the existing google-genai JSON response path."""

    def __init__(self, client, model_name, config):
        self.client = client
        self.model_name = model_name
        self.config = config

    def generate(self, prompt, schema=None):
        response = self.client.models.generate_content(
            model=self.model_name, contents=prompt, config=self.config,
        )
        return json.loads(response.text or "")
