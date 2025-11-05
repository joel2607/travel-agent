from graph.builder import build_travel_planner_with_memory
from config.settings import settings
from memory.memgpt_system import MemGPTSystem


def main():
    print("🌍 Welcome to your AI Travel Planner with Memory!")
    print("I'll remember your preferences and past trips.")
    print("-" * 50)
    
    # Get user ID (in production, from authentication)
    user_id = input("Enter your user ID (or press Enter for 'demo_user'): ").strip()
    if not user_id:
        user_id = "demo_user"
    
    # Build graph with memory
    graph = build_travel_planner_with_memory()
    
    # Initialize state
    inputs = {
        'messages': [],
        'user_id': user_id,
        'memgpt_system': None,  # Will be created in first node
    }
    
    print(f"\n👤 Logged in as: {user_id}")
    print("Type 'exit' to quit, 'memory' to view your core memory, or 'clear memory' to reset your profile.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("👋 Happy travels! Your memories have been saved.")
                break
            
            if user_input.lower() == 'memory':
                if inputs.get('memgpt_system'):
                    memgpt = inputs['memgpt_system']
                    print("\n📝 Your Core Memory:")
                    print(f"Persona: {memgpt.working_context.persona}")
                    print(f"User Profile: {memgpt.working_context.user_profile}")
                    print(f"Context Usage: {memgpt._calculate_context_size()}/{memgpt.max_tokens} tokens")
                else:
                    print("\nNo memory system initialized yet.")
                continue

            if user_input.lower() == 'clear memory':
                if inputs.get('memgpt_system'):
                    confirm = input("Are you sure you want to delete all your memories? This cannot be undone. (yes/no): ").strip().lower()
                    if confirm == 'yes':
                        inputs['memgpt_system'].memory_store.clear_all_memory()
                        # Reset the memgpt system in the current state to reflect the cleared memory
                        inputs['memgpt_system'] = MemGPTSystem(user_id=user_id)
                        inputs['messages'] = [] # Clear message history as well
                        print("\n🗑️ All memories have been cleared. Let's start over.")
                    else:
                        print("\n❌ Memory clearing cancelled.")
                else:
                    print("\nNo memory system to clear.")
                continue
            
            if not user_input:
                continue
            
            inputs['messages'].append({"role": "user", "content": user_input})
            
            # Run the graph
            for step in graph.stream(inputs):
                for node_name, node_state in step.items():
                    # Update the inputs dict with the latest state from the graph
                    inputs.update(node_state)
                    messages = node_state.get('messages', [])
                    if messages and messages[-1].get('role') == 'assistant':
                        # Check if the last message is different from the one before it to avoid printing duplicates
                        if len(messages) < 2 or messages[-1]['content'] != messages[-2].get('content'):
                            print(f"\n🤖 Assistant: {messages[-1]['content']}")
            
            # After a full run, check if a plan was created
            if inputs.get('travel_plan'):
                print("\n✅ Your travel plan is ready and saved to memory!")
                
                another = input("\nWould you like to plan another trip? (yes/no): ").strip().lower()
                if another == 'yes':
                    # Reset planning-specific parts of the state but keep the user profile and memory
                    inputs['user_preferences'] = None
                    inputs['search_queries'] = None
                    inputs['search_results'] = None
                    inputs['travel_plan'] = None
                    # Clear messages to start the new planning session fresh
                    inputs['messages'] = []
                    print("\nGreat! Let's plan your next trip. Where and for how long?")
                else:
                    print("👋 Happy travels!")
                    break
        
        except KeyboardInterrupt:
            print("\n👋 Travel planning interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()