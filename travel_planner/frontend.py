import gradio as gr
from graph.builder import build_travel_planner_with_memory
from config.settings import settings

# Initialize system
graph = build_travel_planner_with_memory()

# Global state dictionary (acts like 'inputs' from main.py)
state = {
    'messages': [],
    'user_id': 'demo_user',
    'memgpt_system': None,
}

def chat_with_travel_agent(user_input, chat_history):
    try:
        # Handle exit
        if user_input.lower() in ['exit', 'quit', 'bye']:
            bot_msg = "👋 Happy travels! Your memories have been saved."
            return chat_history + [(user_input, bot_msg)]
        
        # Handle memory view
        if user_input.lower() == 'memory':
            if state.get('memgpt_system'):
                memgpt = state['memgpt_system']
                bot_msg = (
                    f"📝 Your Core Memory:\n"
                    f"*Persona:* {memgpt.working_context.persona}\n"
                    f"*User Profile:* {memgpt.working_context.user_profile}\n"
                    f"*Context Usage:* {memgpt._calculate_context_size()}/{memgpt.max_tokens} tokens"
                )
            else:
                bot_msg = "No memory available yet!"
            return chat_history + [(user_input, bot_msg)]
        
        if not user_input.strip():
            return chat_history

        # Append user input
        state['messages'].append({"role": "user", "content": user_input})

        # Run through the travel planner graph
        bot_msg = ""
        for step in graph.stream(state):
            for node_name, node_state in step.items():
                messages = node_state.get('messages', [])
                if messages and messages[-1].get('role') == 'assistant':
                    bot_msg = messages[-1]['content']

        # Travel plan ready check
        if 'travel_plan' in state:
            bot_msg += "\n\n✅ Your travel plan is ready and saved to memory!"

        chat_history.append((user_input, bot_msg))
        return chat_history

    except Exception as e:
        bot_msg = f"❌ Error: {e}"
        return chat_history + [(user_input, bot_msg)]


# Gradio UI
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 🌍 AI Travel Planner with Memory\nYour personalized AI that remembers your trips and preferences.")

    chatbot = gr.Chatbot(height=500)
    user_input = gr.Textbox(placeholder="Where would you like to go?", label="Your Message")
    
    send_btn = gr.Button("Send")
    clear_btn = gr.Button("Clear Chat")

    def reset_chat():
        state['messages'] = []
        state['user_preferences'] = None
        state['search_queries'] = None
        state['search_results'] = None
        state['travel_plan'] = None
        return []

    send_btn.click(chat_with_travel_agent, [user_input, chatbot], chatbot)
    clear_btn.click(reset_chat, outputs=chatbot)

# Run app
if _name_ == "_main_":
    demo.launch(server_name="0.0.0.0", server_port=7860)
