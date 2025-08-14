import os
from typing import Any

import mlrun
import openai
from mlrun.serving.v2_serving import V2ModelServer


class OpenAILLMModelServer(V2ModelServer):
    """
    MLRun V2ModelServer for OpenAI LLM chat completion.

    This server is used to generate responses from an OpenAI model, with a system prompt and chat history.
    It supports preprocessing of the input request to format chat history for model monitoring, and then
    invokes the OpenAI chat completion API to generate a response.

    :param context: MLRun context.
    :param name: Name of the function.
    :param model_path: Path to the model (placeholder - not used).
    :param model_name: OpenAI model name (e.g., 'gpt-4o-mini').
    :param system_prompt: System prompt for the LLM.
    :param temperature: Sampling temperature for LLM output.
    :param input_key: Key for input messages in the request (default 'inputs').
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        name: str = None,
        model_path: str = None,
        model_name: str = None,
        system_prompt: str = None,
        temperature: int = 0,
        input_key: str = "inputs",
        **kwargs,
    ):
        super().__init__(name=name, context=context, model_path=model_path, **kwargs)
        self.model = None
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.input_key: str = input_key

    def load(self):
        print("Establishing connection to OpenAI")
        api_key = mlrun.get_secret_or_env("OPENAI_API_KEY")
        base_url = mlrun.get_secret_or_env("OPENAI_BASE_URL")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def preprocess(self, request: dict, operation) -> dict:
        """
        Preprocesses the event body before validation and action.

        This method formats the chat history for a single user input, which is used for model monitoring.
        It concatenates the messages into a single string, with each message on a new line in the format
        "<role>: <content>", and replaces the original messages in the request with this formatted string.

        :param request: The incoming request dictionary containing the chat messages.
        :param operation: The operation to be performed (not used in this method).

        :returns: The modified request dictionary with formatted messages.
        """
        messages = request[self.input_key]

        # format chat history for single user input (used for model monitoring)
        formatted_messages = "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages]
        )
        request[self.input_key] = [formatted_messages]

        return request

    def predict(self, request: dict[str, Any]):
        """
        Generates a prediction using the chat completion API based on the provided request.

        :param request: A dictionary containing the input data for the prediction.
                        The input text is expected to be in the list at the key specified
                        by `self.input_key`.

        :returns: A list containing the generated response as a single string.
        """
        result = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": request[self.input_key][0]},
            ],
        )

        return [result.choices[0].message.content]
