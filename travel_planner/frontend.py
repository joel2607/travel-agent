import gradio as gr
from graph.builder import build_travel_planner_with_memory
from config.settings import settings
from memory.memgpt_system import MemGPTSystem
import json

# Initialize system
graph = build_travel_planner_with_memory()

# Global state dictionary (acts like 'inputs' from main.py)
state = {
    'messages': [],
    'user_id': 'demo_user',
    'memgpt_system': MemGPTSystem(user_id='demo_user'),
}

def chat_with_travel_agent(user_input, chat_history):
    try:
        # If the last message was from the assistant, just return and wait for user input
        if state['messages'] and state['messages'][-1].get('role') == 'assistant':
            if not user_input.strip():
                 return state['messages']

        # Handle exit
        if user_input.lower() in ['exit', 'quit', 'bye']:
            bot_msg = "👋 Happy travels! Your memories have been saved."
            state['messages'].append({"role": "user", "content": user_input})
            state['messages'].append({"role": "assistant", "content": bot_msg})
            return state['messages']
        
        # Handle memory view
        if user_input.lower() == 'memory':
            if state.get('memgpt_system'):
                memgpt = state['memgpt_system']
                
                # Core Memory
                core_memory_msg = (
                    f"📝 **Your Core Memory**\n"
                    f"**Persona:** {memgpt.working_context.persona}\n"
                    f"**User Profile:** {memgpt.working_context.user_profile}\n"
                    f"**Context Usage:** {memgpt._calculate_context_size()}/{memgpt.max_tokens} tokens"
                )
                
                # Archival Memory
                archival_memories = memgpt.memory_store.get_all_archival_memories()
                archival_memory_msg = "\n\n📚 **Archival Memory (Past Trips)**\n"
                if archival_memories:
                    for i, mem in enumerate(archival_memories, 1):
                        content = mem.get('content', 'No content')
                        metadata = mem.get('metadata', {})
                        archival_memory_msg += f"\n**Memory {i}**\n"
                        archival_memory_msg += f"**Destination:** {metadata.get('destination', 'N/A')}\n"
                        archival_memory_msg += f"**Timestamp:** {metadata.get('timestamp', 'N/A')}\n"
                        archival_memory_msg += f"**Details:**\n{content}\n"
                else:
                    archival_memory_msg += "No trips saved to archival memory yet."
                
                bot_msg = core_memory_msg + archival_memory_msg
            else:
                bot_msg = "No memory available yet!"
            
            state['messages'].append({"role": "user", "content": user_input})
            state['messages'].append({"role": "assistant", "content": bot_msg})
            return state['messages']
        
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

        # Append bot message to state
        if bot_msg and (not state['messages'] or bot_msg != state['messages'][-1].get('content')):
             state['messages'].append({"role": "assistant", "content": bot_msg})

        return state['messages']

    except Exception as e:
        bot_msg = f"❌ Error: {e}"
        state['messages'].append({"role": "assistant", "content": bot_msg})
        return state['messages']


# Gradio UI
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 🌍 AI Travel Planner with Memory\nYour personalized AI that remembers your trips and preferences.")

    chatbot = gr.Chatbot(height=500, type="messages")
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

    send_btn.click(chat_with_travel_agent, [user_input, chatbot], chatbot).then(
        lambda: "", outputs=user_input
    )
    clear_btn.click(reset_chat, outputs=chatbot)

# Run app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
