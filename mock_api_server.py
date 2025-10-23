#!/usr/bin/env python3
"""
Mock API server for testing the Streamlit frontend
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Knowledge base snippets for more realistic tool calls
KNOWLEDGE_BASE = {
    "checking": """Guidelines for Opening Checking Accounts

Eligibility Requirements:
• Must be 18 years or older
• Valid government-issued photo ID (driver's license, passport)
• Social Security Number (SSN) or Taxpayer Identification Number (TIN)

Required Documents:
• Proof of address (utility bill, lease agreement) dated within last 60 days
• Initial deposit: $50 minimum for checking

Account Options:
1. Basic Checking
   - Monthly fee: $5 (waived with $500+ balance)
   - APY: 0.01%
   - Features: Online bill pay, mobile deposit, overdraft protection

2. Rewards Checking
   - Monthly fee: $10
   - Cashback: Up to 1.5% on debit card purchases
   - APY: 0.10%
   - Features: ATM fee reimbursements (up to $10/month), free checks""",

    "savings": """Guidelines for Opening Savings Accounts

Account Options:
1. Standard Savings
   - Monthly fee: None
   - Minimum balance: None
   - APY: 2.50%
   - Features: Interest compounded daily, 6 withdrawals/month limit

2. Premium Savings
   - Monthly fee: $3 (waived with $100+ balance)
   - Minimum balance: $100
   - APY: 4.50%
   - Features: Tiered interest rates, priority support""",

    "atm": """ATM Information

IGZ Bank ATM Network:
• 500+ ATMs nationwide
• No fees for IGZ Bank customers
• ATM fee reimbursement up to $10/month with Rewards Checking

Partner Networks:
• Access to 30,000+ ATMs through partner networks
• Look for AllPoint and MoneyPass logos""",

    "hours": """Branch Hours and Locations

Main Branch (Downtown):
• Monday-Friday: 9 AM - 5 PM
• Saturday: 9 AM - 1 PM
• Sunday: Closed

Express Branches (Mall Locations):
• Monday-Saturday: 10 AM - 7 PM
• Sunday: 11 AM - 4 PM

24/7 Services:
• Online Banking
• Mobile App
• Phone Banking: 1-800-IGZ-BANK"""
}

@app.route('/', methods=['POST'])
def banking_agent_endpoint():
    """Mock endpoint that mimics the banking agent API"""

    # Add latency to simulate LLM processing
    time.sleep(random.uniform(1.0, 2.5))

    data = request.json

    # Extract input from request - matching exact UI structure
    user_input = data.get('inputs', [])
    name = data.get('name', 'Bob')  # Default is Bob, not User
    user_id = data.get('user_id', 2296)  # Default to Bob's ID

    # Get the last message and build conversation context
    # Note: The UI only sends user and assistant messages, NOT tool calls
    last_message = ""
    conversation_context = []
    if user_input and len(user_input) > 0:
        # Get only user and assistant messages (matching UI's history_for_api logic)
        for msg in user_input:
            if msg.get('role') in ('user', 'assistant'):
                conversation_context.append(msg)
        last_message = user_input[-1].get('content', '')

    # Determine mock guardrail results first
    toxicity_pass = True
    banking_topic_pass = True

    # Check for toxic content (mock)
    toxic_words = ['hate', 'stupid', 'awful', 'terrible', 'idiot', 'dumb']
    if any(word in last_message.lower() for word in toxic_words):
        toxicity_pass = False

    # Check for non-banking topics (mock)
    non_banking_keywords = ['recipe', 'weather', 'sports', 'movie', 'hot dog', 'pizza', 'cooking', 'game']
    if any(keyword in last_message.lower() for keyword in non_banking_keywords):
        banking_topic_pass = False

    # If guardrails fail, return rejection message
    if not toxicity_pass or not banking_topic_pass:
        response_text = "As a banking agent, I am not allowed to talk on this subject. Is there anything else I can help with?"
        tool_calls = []
    else:
        # Generate response and tool calls based on query
        tool_calls = []

        if 'checking' in last_message.lower() or 'check account' in last_message.lower():
            # Add tool call for checking account info
            tool_calls.append({
                "title": "Used tool bank-info-tool: checking account information",
                "content": KNOWLEDGE_BASE["checking"]
            })

            # Customize response based on sentiment and churn
            if user_id == 32:  # High churn user (Alice)
                response_text = (
                    "I can help you with our checking account options! We have two great choices:\n\n"
                    "**Basic Checking**: $5/month (waived with $500+ balance), includes online bill pay and mobile deposit.\n\n"
                    "**Rewards Checking**: $10/month with 1.5% cashback on debit purchases and ATM fee reimbursements.\n\n"
                    "Both require a $50 minimum deposit. You'll need a valid ID, SSN, and proof of address to open.\n\n"
                    "If you're having any difficulties with the application process, I'd be happy to connect you with "
                    "a personal banker who can walk you through everything. Would that be helpful?"
                )
            else:
                response_text = (
                    "I'd be happy to help you open a checking account! We offer two options:\n\n"
                    "**Basic Checking**: $5/month (waived with $500+ balance)\n"
                    "**Rewards Checking**: $10/month with 1.5% cashback\n\n"
                    "You'll need: valid ID, SSN, proof of address, and $50 minimum deposit. "
                    "You can apply online or visit any branch!"
                )

        elif 'saving' in last_message.lower():
            tool_calls.append({
                "title": "Used tool bank-info-tool: savings account information",
                "content": KNOWLEDGE_BASE["savings"]
            })
            response_text = (
                "Great choice looking into savings accounts! We have excellent options:\n\n"
                "**Standard Savings**: No fees, no minimum, 2.50% APY\n"
                "**Premium Savings**: $3/month (waived with $100+), 4.50% APY\n\n"
                "Both feature daily compounding interest. Which would work better for your savings goals?"
            )

        elif 'atm' in last_message.lower():
            tool_calls.append({
                "title": "Used tool bank-info-tool: ATM locations and fees",
                "content": KNOWLEDGE_BASE["atm"]
            })
            response_text = (
                "You can use any of our 500+ IGZ Bank ATMs nationwide with no fees. "
                "Plus, we partner with AllPoint and MoneyPass networks for 30,000+ additional ATMs. "
                "With Rewards Checking, we even reimburse up to $10/month in other banks' ATM fees!"
            )

        elif 'hour' in last_message.lower() or 'open' in last_message.lower():
            tool_calls.append({
                "title": "Used tool bank-info-tool: branch hours and locations",
                "content": KNOWLEDGE_BASE["hours"]
            })
            response_text = (
                "Our main downtown branch is open Monday-Friday 9 AM - 5 PM, Saturday 9 AM - 1 PM. "
                "Mall express branches have extended hours until 7 PM weekdays and are open Sundays 11 AM - 4 PM. "
                "Remember, online and mobile banking are available 24/7!"
            )

        elif 'password' in last_message.lower() or 'reset' in last_message.lower():
            response_text = (
                "To reset your password, you can:\n"
                "1. Use the 'Forgot Password' link on our website or app\n"
                "2. Call our 24/7 support at 1-800-IGZ-BANK\n"
                "3. Visit any branch with your ID\n\n"
                "For security, you'll need to verify your identity with security questions or a code sent to your phone."
            )

        elif 'transfer' in last_message.lower() or 'wire' in last_message.lower():
            response_text = (
                "For transfers:\n"
                "• **Between IGZ accounts**: Free, instant via online/mobile banking\n"
                "• **To other banks**: 1-3 business days, fees may apply\n"
                "• **Wire transfers**: Same-day, $25 domestic / $45 international\n\n"
                "Daily limits apply based on your account type. Need help setting up a transfer?"
            )

        elif 'hello' in last_message.lower() or 'hi' in last_message.lower():
            # Check if this is a follow-up conversation
            if len(conversation_context) > 1:
                response_text = f"Hello again, {name}! How else can I help you today?"
            else:
                response_text = f"Hello {name}! Welcome to IGZ Bank. I'm here to help with all your banking needs. What can I assist you with today?"

        # NEW: Multiple tool calls for comprehensive queries
        elif ('everything' in last_message.lower() or 'all' in last_message.lower()) and ('account' in last_message.lower() or 'service' in last_message.lower()):
            # Call multiple tools for a comprehensive response
            tool_calls.append({
                "title": "Used tool bank-info-tool: checking accounts",
                "content": KNOWLEDGE_BASE["checking"]
            })
            tool_calls.append({
                "title": "Used tool bank-info-tool: savings accounts",
                "content": KNOWLEDGE_BASE["savings"]
            })
            tool_calls.append({
                "title": "Used tool bank-info-tool: ATM network",
                "content": KNOWLEDGE_BASE["atm"]
            })

            response_text = (
                "I've gathered comprehensive information about all our account types and services!\n\n"
                "**Account Options:**\n"
                "• Basic Checking: $5/month (waived with $500+ balance)\n"
                "• Rewards Checking: $10/month with 1.5% cashback\n"
                "• Standard Savings: No fees, 2.50% APY\n"
                "• Premium Savings: 4.50% APY with $100+ balance\n\n"
                "**ATM Access:** 500+ free ATMs plus 30,000+ partner network ATMs\n\n"
                "I've included detailed information in the expandable sections above. "
                "Would you like me to help you open any specific account?"
            )

        # Multiple tools for comparison queries
        elif 'compare' in last_message.lower() or ('which' in last_message.lower() and 'better' in last_message.lower()):
            tool_calls.append({
                "title": "Used tool bank-info-tool: checking account comparison",
                "content": KNOWLEDGE_BASE["checking"]
            })
            tool_calls.append({
                "title": "Used tool bank-info-tool: savings account comparison",
                "content": KNOWLEDGE_BASE["savings"]
            })

            response_text = (
                "Let me help you compare our account options:\n\n"
                "**For daily banking:** Rewards Checking offers cashback but has a $10 fee, "
                "while Basic Checking is cheaper at $5/month.\n\n"
                "**For savings:** Premium Savings offers our highest rate at 4.50% APY, "
                "while Standard Savings has no fees but lower 2.50% APY.\n\n"
                "Check the detailed comparisons above. What are your main banking priorities?"
            )

        else:
            # General response
            response_text = (
                f"I'm here to help with your banking needs, {name}. "
                "I can assist you with:\n"
                "• Opening checking or savings accounts\n"
                "• Finding ATM locations\n"
                "• Branch hours and locations\n"
                "• Password resets\n"
                "• Account transfers\n\n"
                "What would you like to know more about?"
            )

    # Determine sentiment (mock) - more sophisticated
    negative_words = ['frustrated', 'angry', 'upset', 'annoyed', 'problem', 'issue', 'difficult', 'can\'t', 'won\'t']
    positive_words = ['happy', 'great', 'excellent', 'love', 'perfect', 'wonderful', 'amazing', 'thank']

    sentiment = "neutral"
    if any(word in last_message.lower() for word in negative_words):
        sentiment = "negative"
    elif any(word in last_message.lower() for word in positive_words):
        sentiment = "positive"

    # Determine churn based on user_id (mock)
    if user_id == 32:  # Alice
        churn = "high"
    elif user_id == 2296:  # Bob
        churn = "low"
    else:
        churn = "medium"

    # Build response matching the EXACT expected format
    response = {
        "outputs": [response_text],  # Fallback for error handling in UI
        "guardrails_output": {
            "toxicity-guardrail": {
                "outputs": [toxicity_pass]  # Boolean
            },
            "banking-topic-guardrail": {
                "outputs": [str(banking_topic_pass)]  # String "True" or "False"
            }
        },
        "input_analysis_output": {
            "sentiment-analysis": {
                "outputs": [sentiment]  # "positive", "negative", or "neutral"
            },
            "churn-prediction": {
                "outputs": [churn]  # "high", "medium", or "low"
            }
        },
        "banking-agent": {
            "outputs": {
                "response": [response_text],  # Array with single string
                "tool_calls": tool_calls  # Array of {title, content} objects
            }
        }
    }

    return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    print("Starting mock API server on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)