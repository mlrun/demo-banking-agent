#!/usr/bin/env python3
"""
Vizro implementation of the Banking Agent UI
Using custom components, grid layout, and Dash Mantine Components
"""

import os
from typing import Dict, List, Any, Literal

import requests
import vizro.models as vm
from vizro import Vizro
from vizro.models import VizroBaseModel

from dash import dcc, html, Input, Output, State, Patch
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
from dash_iconify import DashIconify

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8080")
NAMES = {"Alice": 32, "Bob": 2296}

# =============================================================================
# Style Constants
# =============================================================================

# Common style values for consistency
BORDER_RADIUS = "10px"
BADGE_RADIUS = "0px"
SPACING_SM = "8px"
SPACING_MD = "15px"
SPACING_LG = "20px"
FONT_SIZE_SM = "0.875rem"
FONT_SIZE_XS = "0.8rem"

# Message bubble styling
MESSAGE_BUBBLE = {
    "maxWidth": "96%",
    "padding": "10px 15px",
    "marginBottom": SPACING_MD,
    "borderRadius": BORDER_RADIUS,
    "lineHeight": "1.5",
    "letterSpacing": "0.2px",
    "whiteSpace": "pre-wrap",
    "wordBreak": "break-word",
    "minWidth": "100px",
    "color": "var(--text-primary, #212529)"
}

# User message specific styling
USER_MESSAGE_STYLE = {
    **MESSAGE_BUBBLE,
    "backgroundColor": "var(--surfaces-bg-card)",
    "borderLeft": "4px solid #aaa9ba"
}

# Assistant message specific styling
ASSISTANT_MESSAGE_STYLE = {
    **MESSAGE_BUBBLE,
    "backgroundColor": "var(--left-side-bg)",
    "marginRight": "20%",
}

# Container styles
HISTORY_CONTAINER = {
    "maxWidth": "760px",
    "width": "100%",
    "paddingBottom": SPACING_LG,
    "paddingLeft": "5px",
    "overflowY": "auto",
    "height": "100%",
    "display": "flex",
    "flexDirection": "column"
}

HISTORY_SECTION = {
    "display": "flex",
    "justifyContent": "center",
    "width": "100%",
    "flex": "1",
    "overflow": "hidden",
    "paddingTop": SPACING_LG
}

INPUT_SECTION = {
    "display": "flex",
    "justifyContent": "center",
    "width": "100%",
    "marginTop": "auto",
    "paddingBottom": SPACING_LG,
    "paddingLeft": "10px",
    "paddingRight": "10px"
}

# =============================================================================
# Custom Components
# =============================================================================

class ChatInterface(VizroBaseModel):
    """Custom chat interface component for the banking agent"""

    type: Literal["chat_interface"] = "chat_interface"
    title: str = "Chat"

    def build(self):
        """Build the chat interface component"""
        return html.Div([
            # Messages container with legacy styling
            html.Div([
                html.Div(
                    id="chat-messages-container",
                    style=HISTORY_CONTAINER
                )
            ], style=HISTORY_SECTION),

            # Input area
            html.Div([
                dmc.Textarea(
                    id="chat-input",
                    placeholder="Type your message here...",
                    autosize=True,
                    size="md",
                    minRows=1,
                    maxRows=6,
                    radius=BORDER_RADIUS,
                    style={"resize": "none", "width": "100%", "maxWidth": "760px"},
                    rightSection=dmc.ActionIcon(
                        DashIconify(icon="solar:upload-square-bold", width=38),
                        id="send-button",
                        variant="transparent",
                        size="lg",
                        n_clicks=0,
                        color="#00B6ED",
                        radius=BORDER_RADIUS,
                    ),
                    rightSectionWidth=40,
                    rightSectionPointerEvents="all",
                )
            ], style=INPUT_SECTION),

            # Hidden stores for conversation data
            dcc.Store(id="conversation-store", data={"messages": []}),
            dcc.Store(id="guardrails-store", data={
                "toxicity": None,
                "banking": None,
                "sentiment": None,
                "churn": None
            }),
        ], style={"height": "100%", "width": "100%", "display": "flex", "flexDirection": "column"})


class SidebarControls(VizroBaseModel):
    """Custom sidebar component with user selection and status displays"""

    type: Literal["sidebar_controls"] = "sidebar_controls"
    title: str = "Controls"

    def build(self):
        """Build the sidebar controls component"""
        return dmc.Stack([
            # App Parameters
            dmc.Paper([
                dmc.Text("App Parameters", size="lg", fw=600, mb=10),
                dmc.Select(
                    label="User",
                    id="user-select",
                    data=[{"value": name, "label": name} for name in NAMES.keys()],
                    value="Bob",
                    w="100%",
                ),
            ], p="md", withBorder=True),

            # Guardrails Status
            dmc.Paper([
                dmc.Text("Guardrails", size="lg", fw=600, mb=10),
                html.Div(id="toxicity-guardrail", style={"marginBottom": SPACING_SM}),
                html.Div(id="banking-guardrail"),
            ], p="md", withBorder=True),

            # Input Analysis
            dmc.Paper([
                dmc.Text("Input Analysis", size="lg", fw=600, mb=10),
                html.Div(id="sentiment-analysis", style={"marginBottom": SPACING_SM}),
                html.Div(id="churn-prediction"),
            ], p="md", withBorder=True),

            # Clear Button
            dmc.Button(
                "Clear Conversation",
                id="clear-button",
                variant="outline",
                color="#00B6ED",
                fullWidth=True,
                leftSection=DashIconify(icon="tabler:trash", width=16),
                n_clicks=0,
            ),
        ], gap="md")


# =============================================================================
# Register custom components with Vizro
# =============================================================================

vm.Container.add_type("components", ChatInterface)
vm.Container.add_type("components", SidebarControls)


# =============================================================================
# Helper Functions
# =============================================================================

def generate(prompt: str, name: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """API call for model generation - matches Streamlit implementation"""
    inputs = chat_history + [{"role": "user", "content": prompt}]

    try:
        resp = requests.post(
            API_URL,
            json={"inputs": inputs, "name": name, "user_id": NAMES.get(name)},
            verify=False,
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"API Error: {e}")
        return {"outputs": [f"Error connecting to API: {str(e)}"]}


def create_badge(text: str, color: str, icon: str):
    """Helper function to create a consistent badge with common properties"""
    return dmc.Badge(
        text,
        c=color,
        variant="outline",
        leftSection=DashIconify(icon=icon, width=12),
        fullWidth=True,
        radius=BADGE_RADIUS,
    )


def format_guardrail_badge(state, guardrail_name: str):
    """Create a badge for guardrail state"""
    if state is True or state == "True":
        return create_badge(f"{guardrail_name}: Passed", "green", "tabler:check")
    elif state is False or state == "False":
        return create_badge(f"{guardrail_name}: Failed", "red", "tabler:x")
    else:
        return create_badge(f"{guardrail_name}: Not Evaluated", "gray", "tabler:info-circle")


def format_sentiment_badge(state: str):
    """Create a badge for sentiment state"""
    if state == "positive":
        return create_badge("Sentiment: Positive", "green", "tabler:mood-smile")
    elif state == "neutral":
        return create_badge("Sentiment: Neutral", "yellow", "tabler:mood-neutral")
    elif state == "negative":
        return create_badge("Sentiment: Negative", "red", "tabler:mood-sad")
    else:
        return create_badge("Sentiment: Not Evaluated", "gray", "tabler:info-circle")


def format_churn_badge(state: str):
    """Create a badge for churn prediction state"""
    if state == "high":
        return create_badge("Churn Risk: High", "red", "tabler:alert-triangle")
    elif state == "medium":
        return create_badge("Churn Risk: Medium", "yellow", "tabler:alert-circle")
    elif state == "low":
        return create_badge("Churn Risk: Low", "green", "tabler:shield-check")
    else:
        return create_badge("Churn Risk: Not Evaluated", "gray", "tabler:info-circle")


def create_message_component(message: Dict) -> Any:
    """Create a message component from message data with vizro-ai styling"""
    role = message.get("role", "assistant")
    content = message.get("content", "")
    meta_title = message.get("meta_title")

    if meta_title:
        # Tool call message with accordion
        return dmc.Accordion([
            dmc.AccordionItem([
                dmc.AccordionControl(
                    meta_title,
                    icon=DashIconify(icon="tabler:tool", width=16),
                    style={"fontSize": FONT_SIZE_SM}
                ),
                dmc.AccordionPanel(
                    content, style={"whiteSpace": "pre-wrap", "fontSize": FONT_SIZE_XS}
                ),
            ], value=f"tool-{id(message)}")
        ],
        multiple=True,
        chevronPosition="right",
        variant="filled",
        mb="sm"
        )

    if role == "user":
        return html.Div(
            html.Div(content, style={"fontSize": FONT_SIZE_SM}),
            style=USER_MESSAGE_STYLE
        )
    else:
        return html.Div(
            dcc.Markdown(
                content,
                style={"color": "inherit", "fontSize": FONT_SIZE_SM},
                className="assistant-markdown"
            ),
            style=ASSISTANT_MESSAGE_STYLE
        )


# =============================================================================
# Create the Vizro Dashboard
# =============================================================================

# Create the page with grid layout
page = vm.Page(
    title="Banking Agent",
    layout=vm.Grid(
        grid=[
            # 3:1 ratio - chat takes 3 columns, sidebar takes 1 column
            [0, 0, 0, 1],  # Row 1
            [0, 0, 0, 1],  # Row 2
            [0, 0, 0, 1],  # Row 3
            [0, 0, 0, 1],  # Row 4
        ],
        col_gap="48px",
    ),
    components=[
        vm.Container(
            components=[
                ChatInterface(title="Chat Interface"),
            ],
            variant="filled"
        ),
        vm.Container(
            components=[
                SidebarControls(title="Controls & Status"),
            ],
        ),
    ],
)

# Create the dashboard
dashboard = vm.Dashboard(
    pages=[page],
    title="Banking Agent"
)

# Build the Vizro app
app = Vizro()
app.build(dashboard)

# =============================================================================
# Callbacks
# =============================================================================

@app.dash.callback(
    [Output("chat-messages-container", "children", allow_duplicate=True),
     Output("conversation-store", "data", allow_duplicate=True),
     Output("chat-input", "value")],
    [Input("send-button", "n_clicks")],
    [State("chat-input", "value"),
     State("conversation-store", "data")],
    prevent_initial_call=True
)
def add_user_message(n_clicks, input_value, conv_data):
    """Immediately add user message to chat without waiting for API"""
    if not n_clicks or not input_value:
        raise PreventUpdate

    messages = conv_data.get("messages", [])

    # Use Patch to incrementally update
    messages_patch = Patch()

    # Add user message to the messages list
    user_msg = {"role": "user", "content": input_value}
    new_messages = messages + [user_msg]

    # Create message component
    messages_patch.append(create_message_component(user_msg))

    # Add placeholder for assistant response with loading indicator
    placeholder = html.Div([
        dmc.Paper([
            dmc.Group([
                dmc.Loader(size="md", type="dots"),
            ])
        ],
        p="md",
        style={"backgroundColor": "var(--left-side-bg)"}
        )
    ], id="assistant-placeholder")
    messages_patch.append(placeholder)

    # Return updated store with proper dictionary structure
    return messages_patch, {"messages": new_messages}, ""


@app.dash.callback(
    [Output("chat-messages-container", "children", allow_duplicate=True),
     Output("conversation-store", "data", allow_duplicate=True),
     Output("guardrails-store", "data")],
    [Input("conversation-store", "data")],
    [State("user-select", "value"),
     State("guardrails-store", "data")],
    prevent_initial_call=True
)
def generate_assistant_response(conv_data, selected_user, guardrails_data):
    """Generate assistant response after user message is displayed"""
    if not conv_data or not conv_data.get("messages"):
        raise PreventUpdate

    messages = conv_data["messages"]

    # Check if the last message is from user and needs a response
    # Also check if we haven't already processed this (no placeholder after it)
    if not messages or messages[-1]["role"] != "user":
        raise PreventUpdate

    # Build history for API (only user and assistant messages, not tool calls)
    history_for_api = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant") and not m.get("meta_title")
    ]

    # Get response from API
    resp = generate(messages[-1]["content"], selected_user, history_for_api[:-1])

    # Use Patch to update messages display
    messages_patch = Patch()

    # Remove the placeholder by replacing it with an empty div
    messages_patch[-1] = html.Div(style={"display": "none"})

    # Parse response
    tool_calls = None
    try:
        bot_message = resp["banking-agent"]["outputs"]["response"][0]
        tool_calls = resp["banking-agent"]["outputs"].get("tool_calls")
    except:
        bot_message = resp.get("outputs", [""])[0]

    # Build new messages list for store
    new_messages = list(messages)

    # Add tool calls if any
    if tool_calls:
        for t in tool_calls:
            tc_msg = {
                "role": "assistant",
                "content": t.get("content", ""),
                "meta_title": t.get("title")
            }
            new_messages.append(tc_msg)
            messages_patch.append(create_message_component(tc_msg))

    # Add assistant's response
    assistant_msg = {"role": "assistant", "content": bot_message}
    new_messages.append(assistant_msg)
    messages_patch.append(create_message_component(assistant_msg))

    # Update guardrails
    new_guardrails = dict(guardrails_data)

    try:
        new_guardrails["toxicity"] = resp["guardrails_output"]["toxicity-guardrail"]["outputs"][0]
    except:
        new_guardrails["toxicity"] = None

    try:
        new_guardrails["banking"] = resp["guardrails_output"]["banking-topic-guardrail"]["outputs"][0]
    except:
        new_guardrails["banking"] = None

    if "input_analysis_output" in resp:
        try:
            new_guardrails["sentiment"] = resp["input_analysis_output"]["sentiment-analysis"]["outputs"][0]
        except:
            new_guardrails["sentiment"] = None
        try:
            new_guardrails["churn"] = resp["input_analysis_output"]["churn-prediction"]["outputs"][0]
        except:
            new_guardrails["churn"] = None
    else:
        new_guardrails["sentiment"] = None
        new_guardrails["churn"] = None

    return messages_patch, {"messages": new_messages}, new_guardrails


@app.dash.callback(
    [Output("chat-messages-container", "children"),
     Output("conversation-store", "data"),
     Output("guardrails-store", "data", allow_duplicate=True)],
    [Input("clear-button", "n_clicks")],
    prevent_initial_call=True
)
def handle_clear(n_clicks):
    """Handle clear button"""
    if not n_clicks:
        raise PreventUpdate

    return [], {"messages": []}, {
        "toxicity": None,
        "banking": None,
        "sentiment": None,
        "churn": None
    }


@app.dash.callback(
    [Output("toxicity-guardrail", "children"),
     Output("banking-guardrail", "children"),
     Output("sentiment-analysis", "children"),
     Output("churn-prediction", "children")],
    [Input("guardrails-store", "data")]
)
def update_sidebar_badges(guardrails_data):
    """Update sidebar with badge components"""
    return (
        format_guardrail_badge(guardrails_data.get("toxicity"), "Toxicity"),
        format_guardrail_badge(guardrails_data.get("banking"), "Banking Topic"),
        format_sentiment_badge(guardrails_data.get("sentiment")),
        format_churn_badge(guardrails_data.get("churn"))
    )


# Handle Enter key for submission (but allow Shift+Enter for new lines)
app.dash.clientside_callback(
    """
    function(value) {
        // Add event listener for the chat input if not already added
        setTimeout(() => {
            const chatInput = document.getElementById('chat-input');
            if (chatInput && !chatInput.dataset.listenerAdded) {
                chatInput.dataset.listenerAdded = 'true';
                chatInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        const sendButton = document.getElementById('send-button');
                        if (sendButton && chatInput.value.trim()) {
                            sendButton.click();
                        }
                    }
                });
            }
        }, 100);

        return window.dash_clientside.no_update;
    }
    """,
    Output("chat-input", "id"),  # Dummy output
    [Input("chat-input", "value")]
)


# =============================================================================
# Run the application
# =============================================================================

if __name__ == "__main__":
    print(f"Starting Vizro Banking Agent UI")
    print(f"API URL: {API_URL}")
    print(f"Navigate to: http://127.0.0.1:8051")
    app.run(port=8051)