"""
agent_graph.py
==================

Implements the core serving graph and model servers for the Banking Agent Demo application. This module defines custom MLRun model servers and utility classes for:

- Input guardrails (toxicity and topic detection)
- Sentiment analysis
- Churn prediction
- Context building for LLM prompts
- Banking agent orchestration with LLM and retrieval-augmented generation

These components are orchestrated in a serving graph (see 03_application_deployment.ipynb) to process user queries, enforce safety, analyze sentiment and churn propensity, and generate context-aware responses using LLMs and vector search.
"""

import logging
import jmespath
import mlrun
import openai
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import create_retriever_tool
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool
from mlrun.serving.routers import ParallelRun
from storey.transformations import Choice
from langchain_core.prompts import PromptTemplate
from typing import Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaForSequenceClassification, pipeline
import torch
from langchain.agents import AgentExecutor, create_react_agent

# Configure logger
logger = logging.getLogger(__name__)



def enrich_request(event: dict):
    logger.info("[enrich_request] Processing incoming request")
    logger.debug(f"[enrich_request] Input event keys: {list(event.keys())}")
    
    # assumes your request body has: {"inputs": [{"role": "...", "content": "..."} , ...]}
    last_content = (event.get("inputs") or [{}])[-1].get("content")

    # Fill only if missing, so you can still override when needed
    event["latest_user_message"] = last_content
    event["question"] = last_content
    
    logger.info(f"[enrich_request] Extracted question: {last_content[:100] if last_content else 'None'}...")
    return event


def _format_question(question: str, role: str = "user"):
    """
    Format a question for LLM input.

    :param question: The question text.
    :param role: The role of the message sender (default 'user').

    :returns: Formatted message dictionary.
    """
    return {"role": role, "content": question.strip()}


def accept(event):
    """
    Accept handler for valid input.
    Returns the event unchanged if all guardrails pass.

    :param event: The event dictionary.
    :returns: The event unchanged if all guardrails pass.
    """
    logger.info("[accept] Guardrails passed - proceeding with request")
    return event


def reject(event):
    """
    Reject handler for invalid input.

    Returns a standard rejection message if any guardrail fails.

    :param event: The event dictionary.

    :returns: The event with a standard rejection message if any guardrail fails.
    """
    logger.warning("[reject] Guardrails failed - request blocked")
    logger.debug(f"[reject] Blocked question: {event.get('question', 'N/A')}")
    event["outputs"] = [
        "As a banking agent, I am not allowed to talk on this subject. Is there anything else I can help with?"
    ]
    return event


def responder(event):
    """
    Final responder handler.
    Restructures the event to match what frontend_ui.py expects.
    """
    # Get the raw outputs from different steps
    guardrails = event.get("guardrails_output", {})

    # Build the response structure frontend expects
    response = {
        # Main output (fallback for frontend)
        "outputs": event.get("outputs", []),

        # Banking agent output (primary path for frontend)
        "banking-agent": {
            "outputs": event.get("banking-agent", {}).get("outputs",
                       {"response": event.get("outputs", []), "tool_calls": []})
        },

        # Guardrails output - convert 'response'/'answer' to 'outputs'
        "guardrails_output": {
            "toxicity-guardrail": {
                "outputs": guardrails.get("toxicity-guardrail", {}).get("response", [True])
            },
            "banking-topic-guardrail": {
                "outputs": [guardrails.get("banking-topic-guardrail", {}).get("answer", True)]
            },
        },

        # Input analysis - restructure from flat to nested format
        "input_analysis_output": {
            "sentiment-analysis": {
                "outputs": event.get("sentiment_analysis_output", {}).get("response", [None])
            },
            "churn-prediction": {
                "outputs": event.get("churn_model_output", {}).get("response", [None])
            },
        },
    }

    return response


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
        logger.info("[GuardrailsChoice] Evaluating guardrail outputs")
        flag = True
        for guardrail, output in event["guardrails_output"].items():
            # common patterns you have in your data:
            if "response" in output:
                val = output["response"][0]  # e.g., True/False
            elif "answer" in output:
                val = output["answer"]  # e.g., "False"
            else:
                # unknown schema: treat as pass or fail depending on your policy
                val = True

            logger.debug(f"[GuardrailsChoice] {guardrail}: {val}")
            if str(val) == "False":
                flag = False

        decision = self.mapping[str(flag)]
        logger.info(f"[GuardrailsChoice] Decision: {decision}")
        return [decision]


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
        logger.info(f"[ParallelRunMerger] Merged {len(results)} results into '{self.output_key}'")
        logger.debug(f"[ParallelRunMerger] Result keys: {list(results.keys())}")
        return body


class SentimentAnalysisModelServer(mlrun.serving.Model):
    """
    MLRun Model for sentiment analysis.

    Uses a HuggingFace transformer model to classify sentiment of the latest user message.

    :param context: MLRun context.
    :param name: Name of the function.
    :param model_name: HuggingFace model name (default 'cardiffnlp/twitter-roberta-base-sentiment-latest').
    :param top_k: Number of top predictions to return (default 1).
    """

    def __init__(
        self,
        name: str,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k: int = 1,
        context=None,
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

    def predict(self, inputs: dict) -> dict:
        """
        Predicts the sentiment label for the latest input message.

        :param inputs: A dictionary containing an "inputs" key, which is a list of message dictionaries.
        The latest message's content is used for sentiment analysis.

        :returns: A list containing the predicted sentiment label as a string.
        """
        logger.info("[SentimentAnalysis] Analyzing sentiment")
        message = inputs["inputs"][-1]["content"]
        logger.debug(f"[SentimentAnalysis] Message: {message[:100]}...")
        
        sentiment = self.sentiment_classifier(message)[0][0]["label"]
        inputs["response"] = [sentiment]
        
        logger.info(f"[SentimentAnalysis] Result: {sentiment}")
        return inputs


class ChurnModelServer(mlrun.serving.Model):
    """
    MLRun Model for churn prediction.

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
        name: str,
        dataset: str,
        label_column: str,
        endpoint_url: str,
        churn_mappings: dict,
        context=None,
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
        user_id = inputs.get("user_id")
        logger.info(f"[ChurnPrediction] Predicting churn for user_id={user_id}")
        
        resp = requests.post(
            url=self.endpoint_url, json={"inputs": [self.data[user_id]]}
        )
        resp_json = resp.json()
        churn_score = resp_json["results"][0]

        def map_churn_score(value):
            for label, threshold in self.thresholds:
                if value >= threshold:
                    return label

        churn_label = map_churn_score(churn_score)
        inputs["response"] = [churn_label]
        
        logger.info(f"[ChurnPrediction] Result: {churn_label} (score={churn_score:.3f})")
        return inputs


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

        :param event: The input event (either full Event object or dict body) containing data to extract context from.

        :returns: The updated event with the formatted question added under the specified output key.
        """
        logger.info("[BuildContext] Building LLM context from event")
        
        # Handle both full Event objects and body-only dicts
        if hasattr(event, 'body'):
            body = event.body
        else:
            body = event

        extracted_context = {k: jmespath.search(v, body) for k, v in self.context_mappings.items()}
        logger.debug(f"[BuildContext] Extracted context: {extracted_context}")

        body[self.output_key] = [
            _format_question(self.prompt.format(**extracted_context), role=self.role)
        ]
        
        logger.info(f"[BuildContext] Context built with keys: {list(extracted_context.keys())}")
        return event


class BankingAgentOpenAI(mlrun.serving.LLModel):
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
        if self.vector_db_args.get("uri", "").startswith("store://"):
            vectordb_path = mlrun.get_dataitem(self.vector_db_args["uri"]).local()
            self.vector_db_args["uri"] = f"{vectordb_path}"
            import time
            time.sleep(5)

        # 1) Create an LLM object
        self.llm = ChatOpenAI(model=self.model_name, temperature=0)

        # 2) Vector store + retriever tool
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

        # 3) ReAct prompt template
        react_template = """{system_prompt}

        You have access to the following tools:
        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        IMPORTANT: If you already know the answer without needing to use any tools, 
        you can skip the Action step and go directly to:
        Thought: I know the answer to this question.
        Final Answer: <your answer here>

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        """
        prompt = PromptTemplate.from_template(react_template).partial(
            system_prompt=self.system_prompt
        )

        # 4) Create agent runnable
        tools = [DuckDuckGoSearchRun(), self.retriever_tool]
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt,
        )

        # 5) Wrap with AgentExecutor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,  # Stop after 5 attempts

        )

    def _messages_to_input_text(self, messages: Any) -> str:
        """
        Your current request seems to contain a 'formatted_prompt' plus chat history.
        ReAct agent expects a single 'input' string, so we collapse messages into text.
        Adjust this to your actual message object types (dicts vs BaseMessage).
        """
        parts = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
            else:
                # LangChain BaseMessage usually has .type / .content
                role = getattr(m, "type", "user")
                content = getattr(m, "content", str(m))
            parts.append(f"{role}: {content}")
        return "\n".join(parts).strip()

    def predict(
            self,
            body: dict[str, Any],
            messages: list = None,
            invocation_config: Optional[dict] = None,
            **kwargs,
    ):
        logger.info("[BankingAgentOpenAI] Processing request")

        # Prefer MLRun-provided `messages` if present, otherwise fall back to graph payload
        if messages:
            combined_messages = messages
            logger.debug("[BankingAgentOpenAI] Using MLRun-provided messages")
        else:
            prompt_messages = body.get(self.prompt_input_key, [])
            user_messages = body.get(self.messages_input_key, [])
            if not user_messages:
                sentiment_output = body.get("sentiment_analysis_output", {})
                if isinstance(sentiment_output, dict):
                    user_messages = sentiment_output.get(self.messages_input_key, [])
            combined_messages = prompt_messages + user_messages
            logger.debug(f"[BankingAgentOpenAI] Combined {len(combined_messages)} messages from body")

        input_text = self._messages_to_input_text(combined_messages)
        logger.debug(f"[BankingAgentOpenAI] Input text length: {len(input_text)} chars")
        
        resp = self.agent_executor.invoke({"input": input_text})
        response_text = resp.get("output", str(resp))

        # Pass through full body with outputs added
        body["outputs"] = [response_text]
        body["tool_calls"] = []
        
        logger.info(f"[BankingAgentOpenAI] Response generated ({len(response_text)} chars)")
        return body


class BankingAgentHuggingFace(mlrun.serving.LLModel):
    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        name: str = None,
        model_path: str = None,
        model_name: str = None,
        prompt_input_key: str = "formatted_prompt",   #  BuildContext output
        messages_input_key: str = "inputs",           #  Chat history
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        **kwargs,
    ):
        super().__init__(name=name, context=context, model_path=model_path, **kwargs)
        self.model_name = model_name
        self.prompt_input_key = prompt_input_key
        self.messages_input_key = messages_input_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    def _coerce_messages(self, body: dict, messages: Optional[list] = None) -> list[dict]:
        # Prefer MLRun-provided `messages` (like in banking_topic_guardrail.py),
        # otherwise fall back to your graph’s request payload.
        if messages:
            return messages
        msgs = []
        if body.get(self.prompt_input_key):
            # formatted_prompt in your graph is a list of {"role","content"} dicts
            msgs += body[self.prompt_input_key]
        # Get user messages - check top level first, then nested in sentiment_analysis_output
        user_messages = body.get(self.messages_input_key, [])
        if not user_messages:
            # Look for inputs in sentiment_analysis_output (due to ModelRunnerStep structure)
            sentiment_output = body.get("sentiment_analysis_output", {})
            if isinstance(sentiment_output, dict):
                user_messages = sentiment_output.get(self.messages_input_key, [])

        if user_messages:
            msgs += user_messages
        return msgs

    def predict(self, body: Any, messages: list = None, invocation_config: Optional[dict] = None, **kwargs):
        logger.info("[BankingAgentHuggingFace] Processing request")

        msgs = self._coerce_messages(body, messages)
        logger.debug(f"[BankingAgentHuggingFace] Processing {len(msgs)} messages")
        
        prompt = "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in msgs]).strip()
        prompt += "\nassistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
        )
        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        answer = decoded[len(decoded) - max(0, len(decoded) - len(prompt)) :]
        answer = answer.replace("assistant:", "").strip() or decoded.strip()

        # Pass through full body with outputs added
        body["outputs"] = [answer]
        body["tool_calls"] = []
        
        logger.info(f"[BankingAgentHuggingFace] Response generated ({len(answer)} chars)")
        return body



