#!/usr/bin/env python3
"""
Plotly Dash implementation of the Banking Agent UI
Replicates the functionality of the Streamlit frontend_ui.py
"""

import os
import json
from typing import Dict, List, Any
import requests
from datetime import datetime
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, MATCH
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8080")
NAMES = {"Alice": 32, "Bob": 2296}

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Banking Agent",
    suppress_callback_exceptions=True
)

# Custom CSS for chat interface
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .chat-container {
                height: 500px;
                overflow-y: auto;
                border: 1px solid #dee2e6;
                border-radius: 0.375rem;
                padding: 1rem;
                background-color: #f8f9fa;
            }
            .message-user {
                background-color: #007bff;
                color: white;
                padding: 0.75rem;
                border-radius: 1rem;
                margin: 0.5rem 0;
                margin-left: 20%;
                word-wrap: break-word;
            }
            .message-assistant {
                background-color: white;
                color: #212529;
                padding: 0.75rem;
                border-radius: 1rem;
                margin: 0.5rem 0;
                margin-right: 20%;
                border: 1px solid #dee2e6;
                word-wrap: break-word;
            }
            .tool-call {
                background-color: #f0f0f0;
                border-left: 3px solid #6c757d;
                padding: 0.5rem;
                margin: 0.5rem 0;
                font-size: 0.9rem;
            }
            .sidebar-container {
                height: 600px;
                overflow-y: auto;
                padding: 1rem;
                background-color: #f8f9fa;
                border-radius: 0.375rem;
            }
            .guardrail-pass {
                color: #28a745;
                font-weight: 500;
            }
            .guardrail-fail {
                color: #dc3545;
                font-weight: 500;
            }
            .guardrail-none {
                color: #6c757d;
                font-weight: 500;
            }
            .sentiment-positive {
                color: #28a745;
                font-weight: 500;
            }
            .sentiment-neutral {
                color: #ffc107;
                font-weight: 500;
            }
            .sentiment-negative {
                color: #dc3545;
                font-weight: 500;
            }
            .churn-high {
                color: #dc3545;
                font-weight: 500;
            }
            .churn-medium {
                color: #ffc107;
                font-weight: 500;
            }
            .churn-low {
                color: #28a745;
                font-weight: 500;
            }
            #chat-input {
                margin-top: 1rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def create_layout():
    """Create the main layout matching Streamlit UI structure"""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Banking Agent", className="mb-4")
            ])
        ]),

        # Main content area with 3:1 ratio (chat:sidebar)
        dbc.Row([
            # Left column - Chat interface (75% width)
            dbc.Col([
                # Chat messages container
                html.Div(
                    id="chat-messages",
                    className="chat-container",
                ),

                # Input area
                dbc.InputGroup([
                    dbc.Input(
                        id="chat-input",
                        placeholder="Subject to ask about:",
                        type="text",
                        value="",
                    ),
                    dbc.Button(
                        "Send",
                        id="send-button",
                        color="primary",
                        n_clicks=0,
                    )
                ], className="mt-3"),

                # Loading spinner
                dbc.Spinner(
                    html.Div(id="loading-output"),
                    size="sm",
                    color="primary",
                    spinner_style={"margin": "1rem"},
                ),
            ], width=9),

            # Right column - Sidebar (25% width)
            dbc.Col([
                html.Div([
                    # App Parameters section
                    html.H5("App Parameters", className="mb-3"),
                    dbc.Label("User"),
                    dbc.Select(
                        id="user-select",
                        options=[
                            {"label": name, "value": name}
                            for name in NAMES.keys()
                        ],
                        value="Bob",
                        className="mb-3",
                    ),

                    html.Hr(),

                    # Guardrails section
                    html.H5("Guardrails", className="mb-3"),
                    html.Div(id="toxicity-guardrail", className="mb-2"),
                    html.Div(id="banking-guardrail", className="mb-3"),

                    html.Hr(),

                    # Input Analysis section
                    html.H5("Input Analysis", className="mb-3"),
                    html.Div(id="sentiment-analysis", className="mb-2"),
                    html.Div(id="churn-prediction", className="mb-3"),

                    html.Hr(),

                    # Clear button
                    dbc.Button(
                        "Clear",
                        id="clear-button",
                        color="secondary",
                        outline=True,
                        className="w-100",
                        n_clicks=0,
                    ),
                ], className="sidebar-container")
            ], width=3),
        ]),

        # Hidden storage for conversation history
        dcc.Store(id="conversation-store", data={"messages": []}),
        dcc.Store(id="guardrails-store", data={
            "toxicity": None,
            "banking": None,
            "sentiment": None,
            "churn": None
        }),
    ], fluid=True, className="py-4")

app.layout = create_layout()

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

def format_guardrail_state(state, guardrail_name: str) -> html.Div:
    """Display the guardrail state - matches Streamlit formatting"""
    if state is True or state == "True":
        icon = "✅"
        text = f"{guardrail_name}: Passed"
        className = "guardrail-pass"
    elif state is False or state == "False":
        icon = "❌"
        text = f"{guardrail_name}: Failed"
        className = "guardrail-fail"
    elif state is None:
        icon = "ℹ️"
        text = f"{guardrail_name}: Not Evaluated"
        className = "guardrail-none"
    else:
        icon = "❓"
        text = f"{guardrail_name}: Unknown State"
        className = "guardrail-fail"

    return html.Div([
        html.Span(f"{icon} ", style={"marginRight": "0.5rem"}),
        html.Span(text, className=className)
    ])

def format_sentiment_state(state: str) -> html.Div:
    """Display the sentiment analysis state - matches Streamlit formatting"""
    if state == "positive":
        icon = "✅"
        text = "Sentiment Analysis: Positive"
        className = "sentiment-positive"
    elif state == "neutral":
        icon = "⚠️"
        text = "Sentiment Analysis: Neutral"
        className = "sentiment-neutral"
    elif state == "negative":
        icon = "❌"
        text = "Sentiment Analysis: Negative"
        className = "sentiment-negative"
    elif state is None:
        icon = "ℹ️"
        text = "Sentiment Analysis: Not Evaluated"
        className = "guardrail-none"
    else:
        icon = "❓"
        text = f"Sentiment Analysis: Unknown State ({state})"
        className = "guardrail-fail"

    return html.Div([
        html.Span(f"{icon} ", style={"marginRight": "0.5rem"}),
        html.Span(text, className=className)
    ])

def format_churn_state(state: str) -> html.Div:
    """Display the churn prediction state - matches Streamlit formatting"""
    if state == "high":
        icon = "❌"
        text = "Churn Prediction: High"
        className = "churn-high"
    elif state == "medium":
        icon = "⚠️"
        text = "Churn Prediction: Medium"
        className = "churn-medium"
    elif state == "low":
        icon = "✅"
        text = "Churn Prediction: Low"
        className = "churn-low"
    elif state is None:
        icon = "ℹ️"
        text = "Churn Prediction: Not Evaluated"
        className = "guardrail-none"
    else:
        icon = "❓"
        text = f"Churn Prediction: Unknown State ({state})"
        className = "guardrail-fail"

    return html.Div([
        html.Span(f"{icon} ", style={"marginRight": "0.5rem"}),
        html.Span(text, className=className)
    ])

def create_message_div(message: Dict) -> html.Div:
    """Create a message div for the chat interface"""
    role = message.get("role", "assistant")
    content = message.get("content", "")
    meta_title = message.get("meta_title")

    if meta_title:
        # Tool call message
        return dbc.Accordion([
            dbc.AccordionItem(
                html.Pre(content, style={"whiteSpace": "pre-wrap", "fontSize": "0.9rem"}),
                title=meta_title,
            )
        ], start_collapsed=True, className="mb-2")
    else:
        # Regular message
        if role == "user":
            return html.Div(content, className="message-user")
        else:
            return html.Div(content, className="message-assistant")

# Callbacks
@app.callback(
    [Output("chat-messages", "children"),
     Output("conversation-store", "data"),
     Output("guardrails-store", "data"),
     Output("chat-input", "value"),
     Output("loading-output", "children")],
    [Input("send-button", "n_clicks"),
     Input("clear-button", "n_clicks"),
     Input("chat-input", "n_submit")],
    [State("chat-input", "value"),
     State("user-select", "value"),
     State("conversation-store", "data"),
     State("guardrails-store", "data")],
    prevent_initial_call=True
)
def handle_chat_interaction(send_clicks, clear_clicks, submit, input_value,
                           selected_user, conv_data, guardrails_data):
    """Handle all chat interactions - matches Streamlit behavior"""

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Handle clear button
    if trigger_id == "clear-button":
        return [], {"messages": []}, {
            "toxicity": None,
            "banking": None,
            "sentiment": None,
            "churn": None
        }, "", None

    # Handle send button or enter key
    if trigger_id in ["send-button", "chat-input"] and input_value:
        messages = conv_data.get("messages", [])

        # Add user message
        messages.append({"role": "user", "content": input_value})

        # Build history for API (only user and assistant messages, not tool calls)
        history_for_api = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in ("user", "assistant") and not m.get("meta_title")
        ]

        # Call API with loading indicator
        loading = html.Div("Generating response...", style={"color": "#007bff"})

        # Get response from API
        resp = generate(input_value, selected_user, history_for_api[:-1])

        # Parse response - matches Streamlit logic
        tool_calls = None
        try:
            bot_message = resp["banking-agent"]["outputs"]["response"][0]
            tool_calls = resp["banking-agent"]["outputs"].get("tool_calls")
        except:
            bot_message = resp.get("outputs", [""])[0]

        # Add tool calls to messages
        if tool_calls:
            for t in tool_calls:
                tc_content = t.get("content", "")
                tc_title = t.get("title")
                messages.append({
                    "role": "assistant",
                    "content": tc_content,
                    "meta_title": tc_title
                })

        # Add assistant's response
        messages.append({"role": "assistant", "content": bot_message})

        # Update guardrails and analysis
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

        # Create message divs
        message_divs = [create_message_div(m) for m in messages]

        return message_divs, {"messages": messages}, new_guardrails, "", None

    raise PreventUpdate

# Update sidebar guardrails display
@app.callback(
    [Output("toxicity-guardrail", "children"),
     Output("banking-guardrail", "children"),
     Output("sentiment-analysis", "children"),
     Output("churn-prediction", "children")],
    [Input("guardrails-store", "data")]
)
def update_sidebar(guardrails_data):
    """Update sidebar with guardrail and analysis states"""
    return (
        format_guardrail_state(guardrails_data.get("toxicity"), "Toxicity Guardrail"),
        format_guardrail_state(guardrails_data.get("banking"), "Banking Topic Guardrail"),
        format_sentiment_state(guardrails_data.get("sentiment")),
        format_churn_state(guardrails_data.get("churn"))
    )

# Enable Enter key to send messages
app.clientside_callback(
    """
    function(n_submit) {
        return window.dash_clientside.no_update;
    }
    """,
    Output("chat-input", "n_submit"),
    Input("chat-input", "n_submit")
)

if __name__ == "__main__":
    print(f"Starting Dash Banking Agent UI")
    print(f"API URL: {API_URL}")
    print(f"Navigate to: http://127.0.0.1:8050")
    app.run(debug=True, port=8050)