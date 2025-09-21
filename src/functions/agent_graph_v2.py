"""
agent_graph_v2.py
==================

Implements the core serving graph and model servers for the Banking Agent Demo application. This module defines custom MLRun model servers and utility classes for:

- Input guardrails (toxicity and topic detection)
- Sentiment analysis
- Churn prediction
- Context building for LLM prompts
- Banking agent orchestration with LLM and retrieval-augmented generation

These components are orchestrated in a serving graph (see 03_application_deployment.ipynb) to process user queries, enforce safety, analyze sentiment and churn propensity, and generate context-aware responses using LLMs and vector search.
"""

import os
from typing import Any

import evaluate
import jmespath
import mlrun
import openai
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import create_retriever_tool
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from mlrun.serving.routers import ParallelRun
from mlrun.serving.v2_serving import V2ModelServer
from storey.transformations import Choice
from transformers import AutoTokenizer, RobertaForSequenceClassification, pipeline


class OpenAILLMModelServer(V2ModelServer):
    """
    MLRun V2ModelServer for OpenAI LLM chat completion.

    Used in the serving graph to generate responses from an OpenAI model, with a system prompt and chat history.

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

    def predict(self, request: dict[str, Any]):
        """
        Generate a model prediction based on the provided request dictionary.

        This method formats the chat history from the request, sends it to the model for completion,
        and returns the generated response.

        :param request: A dictionary containing the chat messages under the key specified by `self.input_key`.

        :returns: A list containing the model's response as a single string.
        """
        messages = request[self.input_key]

        # format chat history for single user input (used for model monitoring)
        formatted_messages = "\n".join(
            [f"{m['role']}: {m['content']}" for m in messages]
        )
        request[self.input_key] = [formatted_messages]

        result = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": formatted_messages},
            ],
        )

        return [result.choices[0].message.content]


class GuardrailsChoice(Choice):
    """
    Choice router for input guardrails.

    Selects the next step in the serving graph based on the outputs of guardrail checks
    (e.g., toxicity, topic).

    :param mapping: Mapping of boolean string ('True'/'False') to output step names
                    (e.g., {'True': 'accept', 'False': 'reject'}).
    """

    def __init__(self, mapping: dict):
        super().__init__()
        self.mapping = mapping

    def select_outlets(self, event) -> list[str]:
        """
        Selects the appropriate outlet(s) based on the outputs of guardrails in the event.

        Iterates through the guardrails' outputs in the event and sets a flag to "False" if
        any guardrail output is "False".

        :param event: The event dictionary containing guardrails' outputs.

        :returns: A list with the selected outlet(s) based on the guardrails' evaluation.
        """
        flag = "True"
        for guardrail, output in event["guardrails_output"].items():
            if str(output["outputs"][0]) == "False":
                flag = "False"
        return [self.mapping[flag]]


def accept(event):
    """
    Accept handler for valid input.

    Returns the event unchanged if all guardrails pass.

    :param event: The event dictionary.

    :returns: The event unchanged if all guardrails pass.
    """
    print("ACCEPT")
    return event


def reject(event):
    """
    Reject handler for invalid input.

    Returns a standard rejection message if any guardrail fails.

    :param event: The event dictionary.

    :returns: The event with a standard rejection message if any guardrail fails.
    """
    print("REJECT")
    event["outputs"] = [
        "As a banking agent, I am not allowed to talk on this subject. Is there anything else I can help with?"
    ]
    return event


def responder(event):
    """
    Final responder handler.

    Returns the event as the final output of the serving graph.

    :param event: The event dictionary.

    :returns: The event as the final output of the serving graph.
    """
    return event


class ToxicityClassifierModelServer(V2ModelServer):
    """
    MLRun V2ModelServer for toxicity detection.

    Uses the 'toxicity' evaluation module to check if input text contains toxic language.

    :param context: MLRun context.
    :param name: Name of the function.
    :param threshold: Toxicity threshold (default 0.7).
    """

    def __init__(self, context, name: str, threshold: float = 0.7, **class_args):
        # Initialize the base server:
        super(ToxicityClassifierModelServer, self).__init__(
            context=context,
            name=name,
            **class_args,
        )

        # Store the threshold of toxicity:
        self.threshold = threshold

    def load(self):
        self.model = evaluate.load("toxicity", module_type="measurement")

    def predict(self, inputs: dict) -> str:
        """
        Predicts whether the input content is below the toxicity threshold.

        :param inputs: A dictionary containing an "inputs" key, which is a list of dictionaries with a "content" key.

        :returns: A list containing a boolean indicating if the predicted toxicity is below the threshold.
        """
        return [
            self.model.compute(predictions=[i["content"] for i in inputs["inputs"]])[
                "toxicity"
            ][0]
            < self.threshold
        ]


class ParallelRunMerger(ParallelRun):
    """
    ParallelRun router that merges outputs under a specified key.

    Used to combine outputs from multiple guardrails or analysis steps in the serving graph.

    :param output_key: Key under which to store merged results in the event body.
    """

    def __init__(self, output_key: str, **kwargs):
        super().__init__(**kwargs)
        self.output_key = output_key

    def merger(self, body, results):
        body[self.output_key] = results
        return body


class SentimentAnalysisModelServer(V2ModelServer):
    """
    MLRun V2ModelServer for sentiment analysis.

    Uses a HuggingFace transformer model to classify sentiment of the latest user message.

    :param context: MLRun context.
    :param name: Name of the function.
    :param model_name: HuggingFace model name (default 'cardiffnlp/twitter-roberta-base-sentiment-latest').
    :param top_k: Number of top predictions to return (default 1).
    """

    def __init__(
        self,
        context,
        name: str,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k: int = 1,
        **class_args,
    ):
        # Initialize the base server:
        super(SentimentAnalysisModelServer, self).__init__(
            context=context,
            name=name,
            **class_args,
        )
        self.model_name = model_name
        self.top_k = top_k

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name)
        self.sentiment_classifier = pipeline(
            task="sentiment-analysis",
            tokenizer=self.tokenizer,
            model=self.model,
            top_k=self.top_k,
        )

    def predict(self, inputs: dict) -> str:
        """
        Predicts the sentiment label for the latest input message.

        :param inputs: A dictionary containing an "inputs" key, which is a list of message dictionaries.
        The latest message's content is used for sentiment analysis.

        :returns: A list containing the predicted sentiment label as a string.
        """
        message = inputs["inputs"][-1]["content"]
        print("MESSAGE", message)
        return [self.sentiment_classifier(message)[0][0]["label"]]


class ChurnModelServer(V2ModelServer):
    """
    MLRun V2ModelServer for churn prediction.

    Looks up user features and queries a deployed churn model endpoint to predict churn propensity, mapping the score to a label.

    :param context: MLRun context.
    :param name: Name of the function.
    :param dataset: Path to the dataset for user features.
    :param label_column: Name of the churn label column.
    :param endpoint_url: URL of the deployed churn model endpoint.
    :param churn_mappings: Mapping of churn labels to score thresholds.
    """

    def __init__(
        self,
        context,
        name: str,
        dataset: str,
        label_column: str,
        endpoint_url: str,
        churn_mappings: dict,
        **class_args,
    ):
        # Initialize the base server:
        super(ChurnModelServer, self).__init__(
            context=context,
            name=name,
            **class_args,
        )
        self.dataset = dataset
        self.label_column = label_column
        self.endpoint_url = endpoint_url
        self.churn_mappings = churn_mappings

    def load(self):
        # TODO: replace this with online feature set
        df = mlrun.get_dataitem(self.dataset).as_df()
        df = df.drop(self.label_column, axis=1)
        index = df.index.tolist()
        d = df.to_dict(orient="split")["data"]
        self.data = {}
        for i in range(len(index)):
            self.data[index[i]] = d[i]
        self.thresholds = sorted(
            self.churn_mappings.items(), key=lambda x: x[1], reverse=True
        )

    def predict(self, inputs: dict) -> str:
        """
        Predicts the churn label for a given user based on their data.

        Makes a POST request to the configured endpoint with the user's data,
        retrieves the churn score from the response, and maps it to a churn label
        using predefined thresholds.

        :param inputs: A dictionary containing input parameters, must include "user_id".

        :returns: A list containing the predicted churn label(s) for the user.
        """
        resp = requests.post(
            url=self.endpoint_url, json={"inputs": [self.data[inputs["user_id"]]]}
        )
        resp_json = resp.json()
        churn_score = resp_json["outputs"][0]

        # TODO: add churn score mapping into the churn model itself
        def map_churn_score(value):
            for label, threshold in self.thresholds:
                if value >= threshold:
                    return label

        return [map_churn_score(churn_score)]


def _format_question(question: str, role: str = "user"):
    """
    Format a question for LLM input.

    :param question: The question text.
    :param role: The role of the message sender (default 'user').

    :returns: Formatted message dictionary.
    """
    return {"role": role, "content": question.strip()}


class BuildContext:
    """
    Utility class to build LLM prompt context from event data.

    Used in the serving graph to extract relevant fields and format a system prompt for the LLM.

    :param context_mappings: Mapping of context variable names to jmespath expressions.
    :param output_key: Key under which to store the formatted prompt.
    :param prompt: Prompt template string.
    :param role: Role for the formatted message (default 'system').
    """

    def __init__(
        self, context_mappings: dict, output_key: str, prompt: str, role: str = "system"
    ):
        self.context_mappings = context_mappings
        self.output_key = output_key
        self.prompt = prompt
        self.role = role

    def do(self, event):
        """
        Processes the input event by extracting context using JMESPath expressions, formats
        a prompt with the extracted context, and updates the event with the formatted question.

        Example of JMESPath usage:
            Given an event: {'user': {'name': 'Alice', 'age': 30}}
            And a context mapping: {'username': 'user.name'}
            The extracted context will be: {'username': 'Alice'}

        More complex example:
            Given an event:
            {
                "input_analysis_output": {
                    "sentiment-analysis": {"outputs": ["negative"]},
                    "churn-prediction": {"outputs": ["high"]}
                }
            }
            And a context mapping:
            {
                "sentiment": 'input_analysis_output."sentiment-analysis".outputs[0]',
                "churn": 'input_analysis_output."churn-prediction".outputs[0]',
            }
            The extracted context will be:
            {
                "sentiment": "negative",
                "churn": "high",
            }

        :param event: The input event dictionary containing data to extract context from.

        :returns: The updated event dictionary with the formatted question added under the specified output key.
        """
        extracted_context = {
            k: jmespath.search(v, event) for k, v in self.context_mappings.items()
        }
        event[self.output_key] = [
            _format_question(self.prompt.format(**extracted_context), role=self.role)
        ]
        return event


class BankingAgent(V2ModelServer):
    """
    MLRun V2ModelServer for the Banking Agent LLM orchestration.

    Combines LLM, vector search, and web search tools to generate context-aware responses for banking queries.
    Used as the final step in the serving graph, leveraging retrieval-augmented generation and external tools.

    :param vector_db_collection: Name of the Milvus collection for vector search.
    :param vector_db_args: Connection arguments for Milvus.
    :param vector_db_description: Description of the vector DB for the retriever tool.
    :param model_name: OpenAI model name.
    :param system_prompt: System prompt for the LLM.
    :param prompt_input_key: Key for the formatted prompt in the request (default 'formatted_prompt').
    :param messages_input_key: Key for the chat history/messages in the request (default 'inputs').
    :param context: MLRun context.
    :param name: Name of the function.
    :param model_path: Path to the model (not used).
    """

    def __init__(
        self,
        vector_db_collection: str,
        vector_db_args: dict,
        vector_db_description: str,
        model_name: str,
        system_prompt: str,
        prompt_input_key: str = "formatted_prompt",
        messages_input_key: str = "inputs",
        context: mlrun.MLClientCtx = None,
        name: str = None,
        model_path: str = None,
        **kwargs,
    ):
        super().__init__(name=name, context=context, model_path=model_path, **kwargs)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.prompt_input_key = prompt_input_key
        self.messages_input_key = messages_input_key
        self.vector_db_collection = vector_db_collection
        self.vector_db_args = vector_db_args
        self.vector_db_description = vector_db_description

    def load(self):
        """
        Initializes and loads the vector store, retriever tool, and agent.
        This method establishes a connection to the OpenAI embedding service and initializes
        the Milvus vector store with the specified collection and connection arguments. It then
        creates a retriever tool using the vector store and sets up the agent with the specified
        model, tools, and system prompt.
        """

        print("Establishing connection to OpenAI")
        self.vectorstore = Milvus(
            collection_name=self.vector_db_collection,
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            connection_args=self.vector_db_args,
            auto_id=True,
        )
        self.retriever_tool = create_retriever_tool(
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 1}),
            name="bank-info-tool",
            description=self.vector_db_description,
        )
        self.agent = create_react_agent(
            model=self.model_name,
            tools=[DuckDuckGoSearchRun(), self.retriever_tool],
            prompt=self.system_prompt,
        )

    def predict(self, request: dict[str, Any]):
        """
        Processes a request to generate a prediction using the agent, parses tool calls, and formats the response.

        :param request: A dictionary containing input data, including prompt and message history.

        :returns: A dictionary with the agent's response and a list of tool call details for UI display.
        """
        messages = request[self.prompt_input_key] + request[self.messages_input_key]
        print(messages)

        resp = self.agent.invoke({"messages": messages})
        response = resp["messages"][-1].content

        # Parse tool calls from the agent's message trace for UI display. This loop iterates over all messages in the agent's output:
        # - If a message contains tool calls, it adds an entry for each tool call (by id) with a title describing the tool and query used.
        # - If a message is a response to a previous tool call (has a tool_call_id), it attaches the tool's output/content to the corresponding tool call entry.
        # Example output: [{'title': '🛠️ Used tool bank-info-tool: cashback rewards checking account IGZ Bank', 'content': 'Guidelines for Opening Checking/Savings Accounts...}]
        tool_calls = {}
        for m in resp["messages"]:
            md = m.dict()
            if "tool_calls" in md and md["tool_calls"]:
                for t in md["tool_calls"]:
                    tool_calls[t["id"]] = {
                        "title": f"🛠️ Used tool {t['name']}: {t['args']['query']}"
                    }
            if "tool_call_id" in md and md["tool_call_id"] in tool_calls:
                tool_calls[md["tool_call_id"]]["content"] = md["content"]

        return {"response": [response], "tool_calls": list(tool_calls.values())}
