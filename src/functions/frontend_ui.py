import os
from typing import Any, Dict, List

import requests
import streamlit as st

st.set_page_config(page_title="Banking Agent", layout="wide")

API_URL = os.getenv("API_URL")
assert API_URL, "API URL note set"

NAMES = {"Alice": 32, "Bob": 2296}


def format_guardrail_state(state: bool | None, guardrail_name: str) -> None:
    """Display the guardrail state in the sidebar."""
    if state is True or state == "True":
        st.success(f"{guardrail_name}: Passed", icon="✅")
    elif state is False or state == "False":
        st.warning(f"{guardrail_name}: Failed", icon="❌")
    elif state is None:
        st.info(f"{guardrail_name}: Not Evaluated", icon="ℹ️")
    else:
        st.error(f"{guardrail_name}: Unknown State", icon="❓")


def format_sentiment_state(state: str | None) -> None:
    """Display the sentiment analysis state in the sidebar."""
    if state == "positive":
        st.success("Sentiment Analysis: Positive", icon="✅")
    elif state == "neutral":
        st.warning("Sentiment Analysis: Neutral", icon="⚠️")
    elif state == "negative":
        st.error("Sentiment Analysis: Negative", icon="❌")
    elif state is None:
        st.info("Sentiment Analysis: Not Evaluated", icon="ℹ️")
    else:
        st.error(f"Sentiment Analysis: Unknown State ({state})", icon="❓")


def format_churn_state(state: str | None) -> None:
    """Display the churn prediction state in the sidebar."""
    if state == "high":
        st.error("Churn Prediction: High", icon="❌")
    elif state == "medium":
        st.warning("Churn Prediction: Medium", icon="⚠️")
    elif state == "low":
        st.success("Churn Prediction: Low", icon="✅")
    elif state is None:
        st.info("Churn Prediction: Not Evaluated", icon="ℹ️")
    else:
        st.error(f"Churn Prediction: Unknown State ({state})", icon="❓")


def generate(
    prompt: str, name: str, chat_history: List[Dict[str, str]]
) -> Dict[str, Any]:
    """API call for model generation."""
    inputs = chat_history + [{"role": "user", "content": prompt}]
    resp = requests.post(
        API_URL,
        json={"inputs": inputs, "name": name, "user_id": NAMES.get(name)},
        # verify=False,
    )
    resp.raise_for_status()
    resp_json = resp.json()
    return resp_json


def ensure_state() -> None:
    """Initialize session state variables if not already set."""
    defaults = {
        "messages": [],
        "toxicity": None,
        "banking": None,
        "sentiment": None,
        "churn": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar(name: str) -> None:
    """Render the sidebar with parameters and outputs."""
    with st.container(height=600):
        st.write("#### App Parameters")
        st.selectbox(
            "User",
            options=list(NAMES.keys()),
            index=list(NAMES.keys()).index(name),
            key="user_select",
        )
        st.write("#### Guardrails")
        format_guardrail_state(st.session_state["toxicity"], "Toxicity Guardrail")
        format_guardrail_state(st.session_state["banking"], "Banking Topic Guardrail")
        st.write("#### Input Analysis")
        format_sentiment_state(st.session_state["sentiment"])
        format_churn_state(st.session_state["churn"])
        if st.button("Clear", width="stretch"):
            for k in ["messages", "toxicity", "banking", "sentiment", "churn"]:
                st.session_state[k] = [] if k == "messages" else None
            st.rerun()


def render_chat():
    """Render the chat interface and handle user input."""
    with st.container(height=600):
        messages = st.container(height=500)
        # Render prior chat
        for m in st.session_state.messages:
            role = m.get("role", "assistant")
            content = m.get("content", "")
            meta_title = m.get("meta_title")  # tool calls
            if meta_title:
                messages.chat_message(role).expander(meta_title).write(content)
            else:
                messages.chat_message(role).write(content)

        user_prompt = st.chat_input("Subject to ask about:")
        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            messages.chat_message("user").write(user_prompt)

            # Build history for backend
            history_for_api = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
                if m["role"] in ("user", "assistant")
            ]

            # Call backend
            with messages:
                with st.spinner("Generating response..."):
                    resp = generate(
                        user_prompt,
                        st.session_state.get("user_select", "Bob"),
                        history_for_api,
                    )

            # Parse response
            tool_calls = None
            try:
                bot_message = resp["banking-agent"]["outputs"]["response"][0]
                tool_calls = resp["banking-agent"]["outputs"].get("tool_calls")
            except Exception:
                bot_message = resp.get("outputs", [""])[0]

            # Render tool calls
            if tool_calls:
                for t in tool_calls:
                    tc_content = t.get("content", "")
                    tc_title = t.get("title")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": tc_content,
                            "meta_title": tc_title,
                        }
                    )
                    messages.chat_message("assistant").expander(tc_title).write(
                        tc_content
                    )

            # Render assistant's message
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_message}
            )
            with messages.chat_message("assistant"):
                if tool_calls:
                    messages.write(tool_calls)
                messages.write(bot_message)

            # Update sidebar outputs
            try:
                st.session_state["toxicity"] = resp["guardrails_output"][
                    "toxicity-guardrail"
                ]["outputs"][0]
            except Exception:
                st.session_state["toxicity"] = None
            try:
                st.session_state["banking"] = resp["guardrails_output"][
                    "banking-topic-guardrail"
                ]["outputs"][0]
            except Exception:
                st.session_state["banking"] = None
            if "input_analysis_output" in resp:
                try:
                    st.session_state["sentiment"] = resp["input_analysis_output"][
                        "sentiment-analysis"
                    ]["outputs"][0]
                except Exception:
                    st.session_state["sentiment"] = None
                try:
                    st.session_state["churn"] = resp["input_analysis_output"][
                        "churn-prediction"
                    ]["outputs"][0]
                except Exception:
                    st.session_state["churn"] = None
            else:
                st.session_state["sentiment"] = None
                st.session_state["churn"] = None
            st.rerun()


def main():
    ensure_state()
    st.write("# Banking Agent")
    left, right = st.columns([3, 1], gap="small")
    with right:
        render_sidebar(st.session_state.get("user_select", "Bob"))
    with left:
        render_chat()


if __name__ == "__main__":
    main()
