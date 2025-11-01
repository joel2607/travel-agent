from graph.builder import build_travel_planner_with_memory
from config.settings import settings


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
    print("Type 'exit' to quit, 'memory' to view your core memory\n")
    
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
                continue
            
            if not user_input:
                continue
            
            inputs['messages'].append({"role": "user", "content": user_input})
            
            # Run the graph
            for step in graph.stream(inputs):
                for node_name, node_state in step.items():
                    messages = node_state.get('messages', [])
                    if messages and messages[-1].get('role') == 'assistant':
                        print(f"\n🤖 Assistant: {messages[-1]['content']}")
            
            # Check if we're done
            if 'travel_plan' in inputs:
                print("\n✅ Your travel plan is ready and saved to memory!")
                
                # Ask if they want another plan
                another = input("\nWould you like to plan another trip? (yes/no): ").strip().lower()
                if another == 'yes':
                    # Reset planning state but keep memory
                    inputs['user_preferences'] = None
                    inputs['search_queries'] = None
                    inputs['search_results'] = None
                    inputs['travel_plan'] = None
                    inputs['messages'] = []
                else:
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