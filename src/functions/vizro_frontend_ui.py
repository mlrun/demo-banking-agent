#!/usr/bin/env python3
"""
Vizro implementation of the Banking Agent UI
Using vizro_chat_component with a custom banking agent action.
"""

import json
import os
from typing import Annotated, Any, Dict, List, Literal

import dash_mantine_components as dmc
import plotly
import requests
import vizro.models as vm
from dash import Input, Output, Patch, dcc, html, no_update
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify
from pydantic import Field, Tag
from vizro import Vizro
from vizro.models import VizroBaseModel
from vizro_chat_component import Chat, ChatAction

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8080")
NAMES = {"Alice": 32, "Bob": 2296}

# Style constants
BADGE_RADIUS = "0px"
SPACING_SM = "8px"
FONT_SIZE_SM = "0.875rem"
FONT_SIZE_XS = "0.8rem"


# =============================================================================
# Helper Functions
# =============================================================================

def generate(prompt: str, name: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """API call for model generation."""
    inputs = chat_history + [{"role": "user", "content": prompt}]
    try:
        resp = requests.post(
            API_URL,
            json={"inputs": inputs, "name": name, "user_id": NAMES.get(name)},
            verify=False,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"API Error: {e}")
        return {"outputs": [f"Error connecting to API: {str(e)}"]}


def create_badge(text: str, color: str, icon: str):
    """Helper function to create a consistent badge."""
    return dmc.Badge(
        text,
        c=color,
        variant="outline",
        leftSection=DashIconify(icon=icon, width=12),
        fullWidth=True,
        radius=BADGE_RADIUS,
    )


def format_guardrail_badge(state, guardrail_name: str):
    """Create a badge for guardrail state."""
    if state is True or state == "True":
        return create_badge(f"{guardrail_name}: Passed", "green", "tabler:check")
    elif state is False or state == "False":
        return create_badge(f"{guardrail_name}: Failed", "red", "tabler:x")
    return create_badge(f"{guardrail_name}: Not Evaluated", "gray", "tabler:info-circle")


def format_sentiment_badge(state: str):
    """Create a badge for sentiment state."""
    mapping = {
        "positive": ("Sentiment: Positive", "green", "tabler:mood-smile"),
        "neutral": ("Sentiment: Neutral", "yellow", "tabler:mood-neutral"),
        "negative": ("Sentiment: Negative", "red", "tabler:mood-sad"),
    }
    if state in mapping:
        return create_badge(*mapping[state])
    return create_badge("Sentiment: Not Evaluated", "gray", "tabler:info-circle")


def format_churn_badge(state: str):
    """Create a badge for churn prediction state."""
    mapping = {
        "high": ("Churn Risk: High", "red", "tabler:alert-triangle"),
        "medium": ("Churn Risk: Medium", "yellow", "tabler:alert-circle"),
        "low": ("Churn Risk: Low", "green", "tabler:shield-check"),
    }
    if state in mapping:
        return create_badge(*mapping[state])
    return create_badge("Churn Risk: Not Evaluated", "gray", "tabler:info-circle")


# =============================================================================
# Custom Banking Agent Action
# =============================================================================

class banking_agent_action(ChatAction):
    """Banking agent chat action that calls the mock API and updates guardrails."""

    type: Literal["banking_agent_action"] = "banking_agent_action"
    selected_user: str = Field(
        default="user-select.value",
        description="Reference to user selection dropdown.",
    )

    @property
    def _parameters(self) -> set[str]:
        params = set(super()._parameters)
        params.add("selected_user")
        return params

    @property
    def outputs(self) -> list[str]:
        return super().outputs + ["guardrails-store.data"]

    def function(self, prompt: str, messages: list[dict[str, Any]], **extra_inputs: Any) -> list[Any]:
        if not prompt or not prompt.strip():
            return [no_update] * len(self.outputs)

        selected_user = extra_inputs.get("selected_user", "Bob")

        # Build API history from stored messages (only plain text messages)
        history = []
        for m in messages:
            content = json.loads(m["content_json"])
            if isinstance(content, str):
                history.append({"role": m["role"], "content": content})

        # Call banking API
        resp = generate(prompt, selected_user, history)

        # Parse response
        tool_calls = None
        try:
            bot_message = resp["banking-agent"]["outputs"]["response"][0]
            tool_calls = resp["banking-agent"]["outputs"].get("tool_calls")
        except Exception:
            bot_message = resp.get("outputs", [""])[0]

        # Build store updates
        latest_input = {"role": "user", "content_json": json.dumps(prompt)}
        messages.append(latest_input)

        store = Patch()
        store.append(latest_input)

        # Add tool calls as component messages
        if tool_calls:
            for t in tool_calls:
                tc_component = dmc.Accordion(
                    [
                        dmc.AccordionItem(
                            [
                                dmc.AccordionControl(
                                    t.get("title", "Tool Call"),
                                    icon=DashIconify(icon="tabler:tool", width=16),
                                    style={"fontSize": FONT_SIZE_SM},
                                ),
                                dmc.AccordionPanel(
                                    t.get("content", ""),
                                    style={"whiteSpace": "pre-wrap", "fontSize": FONT_SIZE_XS},
                                ),
                            ],
                            value=f"tool-{id(t)}",
                        )
                    ],
                    multiple=True,
                    chevronPosition="right",
                    variant="filled",
                )
                content_json = json.dumps(tc_component, cls=plotly.utils.PlotlyJSONEncoder)
                tc_msg = {"role": "assistant", "content_json": content_json}
                messages.append(tc_msg)
                store.append(tc_msg)

        # Add bot response
        bot_msg = {"role": "assistant", "content_json": json.dumps(bot_message)}
        messages.append(bot_msg)
        store.append(bot_msg)

        html_messages = [self.message_to_html(msg) for msg in messages]

        # Parse guardrails from API response
        guardrails = {"toxicity": None, "banking": None, "sentiment": None, "churn": None}
        try:
            guardrails["toxicity"] = resp["guardrails_output"]["toxicity-guardrail"]["outputs"][0]
        except Exception:
            pass
        try:
            guardrails["banking"] = resp["guardrails_output"]["banking-topic-guardrail"]["outputs"][0]
        except Exception:
            pass
        if "input_analysis_output" in resp:
            try:
                guardrails["sentiment"] = resp["input_analysis_output"]["sentiment-analysis"]["outputs"][0]
            except Exception:
                pass
            try:
                guardrails["churn"] = resp["input_analysis_output"]["churn-prediction"]["outputs"][0]
            except Exception:
                pass

        return [store, html_messages, no_update, guardrails]


# =============================================================================
# Sidebar Controls Component
# =============================================================================

class SidebarControls(VizroBaseModel):
    """Custom sidebar component with user selection and status displays."""

    type: Literal["sidebar_controls"] = "sidebar_controls"
    title: str = "Controls"

    def build(self):
        return dmc.Stack(
            [
                dmc.Paper(
                    dmc.Stack(
                        [
                            # App Parameters
                            html.Div(
                                [
                                    dmc.Text("App Parameters", size="lg", fw=600, mb=10),
                                    dmc.Select(
                                        label="User",
                                        id="user-select",
                                        data=[{"value": name, "label": name} for name in NAMES.keys()],
                                        value="Bob",
                                        w="100%",
                                    ),
                                ],
                            ),
                            dmc.Divider(),
                            # Guardrails Status
                            html.Div(
                                [
                                    dmc.Text("Guardrails", size="lg", fw=600, mb=10),
                                    html.Div(id="toxicity-guardrail", style={"marginBottom": SPACING_SM}),
                                    html.Div(id="banking-guardrail"),
                                ],
                            ),
                            # Input Analysis
                            html.Div(
                                [
                                    dmc.Text("Input Analysis", size="lg", fw=600, mb=10),
                                    html.Div(id="sentiment-analysis", style={"marginBottom": SPACING_SM}),
                                    html.Div(id="churn-prediction"),
                                ],
                            ),
                        ],
                        gap="xl",
                    ),
                    p="lg",
                    withBorder=True,
                ),
                # Clear Button (outside the bordered panel)
                dmc.Button(
                    "Clear Conversation",
                    id="clear-button",
                    variant="outline",
                    color="#00B6ED",
                    fullWidth=True,
                    leftSection=DashIconify(icon="tabler:trash", width=16),
                    n_clicks=0,
                ),
                # Guardrails data store
                dcc.Store(
                    id="guardrails-store",
                    data={"toxicity": None, "banking": None, "sentiment": None, "churn": None},
                ),
            ],
            gap="lg",
        )


# =============================================================================
# Register Components
# =============================================================================

vm.Page.add_type("components", Chat)
vm.Container.add_type("components", SidebarControls)
Chat.add_type("actions", Annotated[banking_agent_action, Tag("banking_agent_action")])

# =============================================================================
# Create the Vizro Dashboard
# =============================================================================

page = vm.Page(
    title="Banking Agent",
    layout=vm.Grid(
        grid=[
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        col_gap="48px",
    ),
    components=[
        Chat(
            id="banking_chat",
            placeholder="Type your message here...",
            actions=[banking_agent_action(parent_id="banking_chat")],
            example_questions=[
                "How to apply for a checking account?",
                "What are the nearest ATM locations?",
                "What are the branch hours?",
                "How do I reset my password?",
            ],
        ),
        vm.Container(
            components=[SidebarControls(title="Controls & Status")],
        ),
    ],
)

dashboard = vm.Dashboard(pages=[page], title="Banking Agent")
app = Vizro()
app.build(dashboard)

# =============================================================================
# Callbacks (registered after build)
# =============================================================================


@app.dash.callback(
    [
        Output("toxicity-guardrail", "children"),
        Output("banking-guardrail", "children"),
        Output("sentiment-analysis", "children"),
        Output("churn-prediction", "children"),
    ],
    [Input("guardrails-store", "data")],
)
def update_sidebar_badges(guardrails_data):
    """Update sidebar badges from guardrails store."""
    return (
        format_guardrail_badge(guardrails_data.get("toxicity"), "Toxicity"),
        format_guardrail_badge(guardrails_data.get("banking"), "Banking Topic"),
        format_sentiment_badge(guardrails_data.get("sentiment")),
        format_churn_badge(guardrails_data.get("churn")),
    )


@app.dash.callback(
    [
        Output("banking_chat-hidden-messages", "children"),
        Output("banking_chat-store", "data"),
        Output("guardrails-store", "data", allow_duplicate=True),
    ],
    [Input("clear-button", "n_clicks")],
    prevent_initial_call=True,
)
def handle_clear(n_clicks):
    """Handle clear button - reset chat and guardrails."""
    if not n_clicks:
        raise PreventUpdate
    return [], [], {"toxicity": None, "banking": None, "sentiment": None, "churn": None}


# =============================================================================
# Run the application
# =============================================================================

if __name__ == "__main__":
    print("Starting Vizro Banking Agent UI")
    print(f"API URL: {API_URL}")
    print("Navigate to: http://127.0.0.1:8051")
    app.run(port=8051)
