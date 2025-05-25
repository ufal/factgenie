import json
import logging
import os
import random
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelAPI:
    def __init__(self, config: dict, api_kwargs: dict = {}):
        # Importing LiteLLM is currently quite slow: https://github.com/BerriAI/litellm/issues/7605
        import litellm

        self.config = config
        self.api_kwargs = api_kwargs

        self.validate_environment()

    def get_model_service_name(self):
        # Get the model service name from the config
        model_service = self.config["model"]

        # Add the service prefix to the model service name
        if self._service_prefix():
            model_service = self._service_prefix() + model_service

        return model_service

    def validate_environment(self):
        import litellm

        model_service = self.get_model_service_name()
        response = litellm.validate_environment(model=model_service)

        if not response["keys_in_environment"]:
            raise ValueError(
                f"Required API variables not found for the model {model_service}. Please add the following keys to the system environment or factgenie config: {response['missing_keys']}"
            )

    def call_model_once(self, messages, model_service, prompt_strat_kwargs):
        import litellm

        response = litellm.completion(
            model=model_service,
            messages=messages,
            api_base=self._api_url(),
            **prompt_strat_kwargs,  # E.g. structured output format.
            **self.api_kwargs,  # E.g. credentials.
            **self.config.get("model_args", {}),  # E.g. temperature, max_tokens, etc.
        )
        return response

    def get_model_response_with_retries(self, messages, prompt_strat_kwargs={}):
        import litellm

        """Handle rate limits and overload errors with exponential backoff and retry logic."""
        max_retries = 15
        initial_retry_delay = 2  # seconds

        # Get the model service name
        model_service = self.get_model_service_name()
        logger.info(f"Waiting for {model_service}.")

        for attempt in range(max_retries):
            try:
                response = self.call_model_once(messages, model_service, prompt_strat_kwargs=prompt_strat_kwargs)
                return response

            except (litellm.exceptions.RateLimitError, litellm.exceptions.InternalServerError) as e:
                # Check if InternalServerError is specifically an "Overloaded" error
                is_overloaded = isinstance(e, litellm.exceptions.InternalServerError) and "Overloaded" in str(e)
                # Check if we've reached max retries
                if attempt == max_retries - 1:
                    error_type = "Rate limit" if isinstance(e, litellm.exceptions.RateLimitError) else "Server overload"
                    logger.error(f"{error_type} exceeded after {max_retries} attempts. Giving up.")
                    raise e

                # Only retry for rate limits or overloaded errors
                if not (isinstance(e, litellm.exceptions.RateLimitError) or is_overloaded):
                    logger.error(f"Non-retryable InternalServerError: {str(e)}")
                    raise e

                # Calculate exponential backoff with jitter
                retry_delay = initial_retry_delay * (2**attempt) + (random.uniform(0, 1))
                error_type = "Rate limit" if isinstance(e, litellm.exceptions.RateLimitError) else "Server overload"
                logger.warning(
                    f"{error_type} hit. Retrying in {retry_delay:.2f} seconds (attempt {attempt+1}/{max_retries})..."
                )
                time.sleep(retry_delay)

            except Exception as e:
                # For other exceptions, don't retry
                logger.error(f"Error calling API: {str(e)}")
                raise e

    def _api_url(self):
        # by default we ignore the API URL
        # override for local services that actually require the API URL (such as Ollama)
        return None

    def _service_prefix(self):
        raise NotImplementedError(
            "Override this method in the subclass to call the appropriate API. See LiteLLM documentation: https://docs.litellm.ai/docs/providers."
        )


class OpenAIAPI(ModelAPI):
    # https://docs.litellm.ai/docs/providers/openai
    def __init__(self, config, api_kwargs: dict = {}):
        super().__init__(config, api_kwargs)

    def _service_prefix(self):
        # OpenAI models do not seem to require a prefix: https://docs.litellm.ai/docs/providers/openai
        return ""


class OllamaAPI(ModelAPI):
    # https://docs.litellm.ai/docs/providers/ollama
    def __init__(self, config, api_kwargs: dict = {}):
        super().__init__(config, api_kwargs)

    def _service_prefix(self):
        # we want to call the `chat` endpoint: https://docs.litellm.ai/docs/providers/ollama#using-ollama-apichat
        return "ollama_chat/"

    def _api_url(self):
        # local server URL
        api_url = self.config.get("api_url", None)
        api_url = api_url.rstrip("/")

        if api_url.endswith("/generate") or api_url.endswith("/chat") or api_url.endswith("/api"):
            raise ValueError(f"The API URL {api_url} is not valid. Use only the base URL, e.g. http://localhost:11434.")

        return api_url

    def validate_environment(self):
        # Ollama would require setting OLLAMA_API_BASE, but we set the API URL in the config
        pass


class VllmAPI(ModelAPI):
    # https://docs.litellm.ai/docs/providers/vllm
    def __init__(self, config, api_kwargs: dict = {}):
        super().__init__(config, api_kwargs)

    def _service_prefix(self):
        return "hosted_vllm/"

    def _api_url(self):
        # local server URL
        return self.config.get("api_url", None)


class AnthropicAPI(ModelAPI):
    # https://docs.litellm.ai/docs/providers/anthropic
    def __init__(self, config, api_kwargs: dict = {}):
        super().__init__(config, api_kwargs)

    def _service_prefix(self):
        return "anthropic/"


class GeminiAPI(ModelAPI):
    # https://docs.litellm.ai/docs/providers/gemini
    def __init__(self, config, api_kwargs: dict = {}):
        super().__init__(config, api_kwargs)

    def _service_prefix(self):
        return "gemini/"


class VertexAIAPI(ModelAPI):
    # https://docs.litellm.ai/docs/providers/vertex
    def __init__(self, config, api_kwargs: dict = {}):
        api_kwargs["vertex_credentials"] = self.load_google_credentials()
        super().__init__(config, api_kwargs)

    def load_google_credentials(self):
        json_file_path = os.environ.get("VERTEXAI_JSON_FULL_PATH")

        if json_file_path:
            if not os.path.exists(json_file_path):
                raise ValueError(
                    "File not found in VERTEXAI_JSON_FULL_PATH. For more details, see https://docs.litellm.ai/docs/providers/vertex"
                )

            # Load the JSON file
            with open(json_file_path, "r") as file:
                vertex_credentials = json.load(file)

            # Convert to JSON string
            return json.dumps(vertex_credentials)

    def _service_prefix(self):
        return "vertex_ai/"
