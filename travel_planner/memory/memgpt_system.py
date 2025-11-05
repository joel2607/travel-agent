from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from config.settings import settings
from models.memory import CoreMemory, ConversationMessage, TripMemory
from memory.memory_store import MemoryStore
from utils.prompts import MEMGPT_SYSTEM_PROMPT


class MemGPTSystem:
    """Complete MemGPT implementation for travel planner"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.GEMINI_API_KEY
        )
        
        # Initialize memory store
        self.memory_store = MemoryStore(user_id)
        
        # Main context components
        self.system_instructions = MEMGPT_SYSTEM_PROMPT
        self.working_context = self._load_or_create_core_memory()
        self.fifo_queue: List[ConversationMessage] = []
        self.queue_summary: str = ""
        
        # Context tracking
        self.max_tokens = settings.MAX_CONTEXT_TOKENS
        self.warning_threshold = settings.MEMORY_WARNING_THRESHOLD
        self.flush_threshold = settings.FLUSH_THRESHOLD
        
        # Function definitions
        self.functions = self._define_memory_functions()
    
    def _load_or_create_core_memory(self) -> CoreMemory:
        """Load existing core memory or create new"""
        core = self.memory_store.get_core_memory()
        if core:
            return CoreMemory(**core)
        
        # Create a new core memory with a default user profile
        new_core = CoreMemory(
            user_id=self.user_id,
            user_profile="I am a new user. Please ask me about my travel preferences."
        )
        self.memory_store.save_core_memory(new_core.dict())
        return new_core
    
    def _define_memory_functions(self) -> List[Dict]:
        """Define all memory management functions"""
        return [
            {
                "name": "core_memory_append",
                "description": "Append content to a field in core memory (persona or user_profile)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": ["persona", "user_profile"],
                            "description": "Field to append to"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to append"
                        },
                        "request_heartbeat": {
                            "type": "boolean",
                            "description": "Request immediate follow-up inference",
                            "default": False
                        }
                    },
                    "required": ["name", "content"]
                }
            },
            {
                "name": "core_memory_replace",
                "description": "Replace content in core memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": ["persona", "user_profile"],
                            "description": "Field to modify"
                        },
                        "old_content": {
                            "type": "string",
                            "description": "Content to replace"
                        },
                        "new_content": {
                            "type": "string",
                            "description": "New content"
                        },
                        "request_heartbeat": {
                            "type": "boolean",
                            "default": False
                        }
                    },
                    "required": ["name", "old_content", "new_content"]
                }
            },
            {
                "name": "conversation_search",
                "description": "Semantic search over conversation history",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "page": {
                            "type": "integer",
                            "description": "Page number for pagination",
                            "default": 1
                        },
                        "request_heartbeat": {
                            "type": "boolean",
                            "default": False
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "archival_memory_search",
                "description": "Search past trips and travel documents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "page": {
                            "type": "integer",
                            "default": 1
                        },
                        "request_heartbeat": {
                            "type": "boolean",
                            "default": False
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "archival_memory_insert",
                "description": "Store a trip memory or travel document",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to store"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional metadata"
                        },
                        "request_heartbeat": {
                            "type": "boolean",
                            "default": False
                        }
                    },
                    "required": ["content"]
                }
            },
            {
                "name": "send_message",
                "description": "Send a message to the user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to send"
                        },
                        "request_heartbeat": {
                            "type": "boolean",
                            "default": False
                        }
                    },
                    "required": ["message"]
                }
            }
        ]
    
    def _calculate_context_size(self) -> int:
        """Estimate current token usage"""
        # Simplified token counting - use tiktoken in production
        context_str = (
            self.system_instructions +
            json.dumps(self.working_context.dict()) +
            json.dumps([msg.dict() for msg in self.fifo_queue]) +
            self.queue_summary
        )
        # Rough estimate: 1 token ≈ 4 characters
        return len(context_str) // 4
    
    def _build_prompt(self) -> List[BaseMessage]:
        """Construct prompt from main context components"""
        messages = []
        
        # System instructions with function definitions
        system_content = f"""{self.system_instructions}

## AVAILABLE FUNCTIONS
{json.dumps(self.functions, indent=2)}

## CORE MEMORY
{json.dumps(self.working_context.dict(), indent=2)}

## QUEUE SUMMARY
{self.queue_summary if self.queue_summary else "No messages evicted yet."}
"""
        messages.append(SystemMessage(content=system_content))
        
        # Add messages from FIFO queue (keep last 20)
        for msg in self.fifo_queue[-20:]:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
            elif msg.role == "function":
                messages.append(FunctionMessage(
                    name=msg.metadata.get('function_name', 'unknown'),
                    content=msg.content
                ))
            elif msg.role == "system":
                messages.append(SystemMessage(content=msg.content))
        
        return messages
    
    def process_message(self, user_message: str) -> Dict[str, Any]:
        """Main MemGPT processing loop with heartbeats"""
        # Add user message to queue and recall storage
        msg = ConversationMessage(
            role="user",
            content=user_message,
            timestamp=datetime.now().isoformat()
        )
        self.fifo_queue.append(msg)
        self.memory_store.save_conversation_message(msg)
        
        # Check for memory pressure
        current_tokens = self._calculate_context_size()
        if current_tokens > self.max_tokens * self.warning_threshold:
            warning_msg = ConversationMessage(
                role="system",
                content=f"⚠️ Memory Pressure Warning: {current_tokens}/{self.max_tokens} tokens used. "
                        f"Consider saving important information to core memory or archival storage.",
                timestamp=datetime.now().isoformat()
            )
            self.fifo_queue.append(warning_msg)
        
        # Execute agent loop with heartbeats
        final_response = self._agent_loop_with_heartbeats()
        
        # Check if queue flush needed
        if current_tokens > self.max_tokens * self.flush_threshold:
            self._flush_queue()
        
        return {
            "response": final_response,
            "context_usage": current_tokens,
            "max_context": self.max_tokens
        }
    
    def _agent_loop_with_heartbeats(self) -> str:
        """Execute agent with function calling and heartbeat mechanism"""
        heartbeat_requested = True
        final_response = None
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while heartbeat_requested and iteration < max_iterations:
            iteration += 1
            
            # Build prompt from main context
            prompt = self._build_prompt()
            
            # Bind functions to LLM
            llm_with_tools = self.llm.bind_tools(self.functions)
            
            # LLM inference
            response = llm_with_tools.invoke(prompt)
            
            # Check if LLM made function calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    # Execute function
                    function_result = self._execute_function(
                        tool_call['name'],
                        tool_call['args']
                    )
                    
                    # Add function result to queue
                    func_msg = ConversationMessage(
                        role="function",
                        content=json.dumps(function_result),
                        timestamp=datetime.now().isoformat(),
                        metadata={"function_name": tool_call['name']}
                    )
                    self.fifo_queue.append(func_msg)
                    
                    # Check for heartbeat request
                    heartbeat_requested = function_result.get('request_heartbeat', False)
                    
                    # If send_message called, extract final response
                    if tool_call['name'] == 'send_message':
                        final_response = function_result['message']
                        if not heartbeat_requested:
                            # This is the final message, exit loop
                            return final_response
            else:
                # No function calls, treat content as final response
                final_response = response.content
                heartbeat_requested = False
                
                # Add to queue
                assistant_msg = ConversationMessage(
                    role="assistant",
                    content=final_response,
                    timestamp=datetime.now().isoformat()
                )
                self.fifo_queue.append(assistant_msg)
                self.memory_store.save_conversation_message(assistant_msg)
        
        return final_response or "I apologize, but I encountered an issue processing your request."
    
    def _execute_function(self, function_name: str, args: Dict) -> Dict:
        """Execute memory management functions"""
        request_heartbeat = args.get('request_heartbeat', False)
        
        if function_name == "core_memory_append":
            field = args['name']
            content = args['content']
            
            # Update working context
            if field == "persona":
                self.working_context.persona += "\n" + content
            elif field == "user_profile":
                self.working_context.user_profile += "\n" + content
            
            # Persist to storage
            self.memory_store.save_core_memory(self.working_context.dict())
            
            return {
                "status": "success",
                "message": f"Appended to {field}",
                "request_heartbeat": request_heartbeat
            }
        
        elif function_name == "core_memory_replace":
            field = args['name']
            old_content = args['old_content']
            new_content = args['new_content']
            
            if field == "persona":
                self.working_context.persona = self.working_context.persona.replace(
                    old_content, new_content
                )
            elif field == "user_profile":
                self.working_context.user_profile = self.working_context.user_profile.replace(
                    old_content, new_content
                )
            
            self.memory_store.save_core_memory(self.working_context.dict())
            
            return {
                "status": "success",
                "message": f"Replaced content in {field}",
                "request_heartbeat": request_heartbeat
            }
        
        elif function_name == "conversation_search":
            query = args['query']
            page = args.get('page', 1)
            
            results = self.memory_store.search_conversations(query, page=page)
            
            return {
                "results": results,
                "page": page,
                "total_results": len(results),
                "request_heartbeat": request_heartbeat
            }
        
        elif function_name == "archival_memory_search":
            query = args['query']
            page = args.get('page', 1)
            
            results = self.memory_store.search_archival(query, page=page)
            
            return {
                "results": results,
                "page": page,
                "request_heartbeat": request_heartbeat
            }
        
        elif function_name == "archival_memory_insert":
            content = args['content']
            metadata = args.get('metadata', {})
            
            self.memory_store.insert_archival(content, metadata)
            
            return {
                "status": "success",
                "message": "Stored in archival memory",
                "request_heartbeat": request_heartbeat
            }
        
        elif function_name == "send_message":
            return {
                "message": args['message'],
                "request_heartbeat": request_heartbeat
            }
        
        else:
            return {
                "error": f"Unknown function: {function_name}",
                "request_heartbeat": False
            }
    
    def _flush_queue(self):
        """Evict old messages and create recursive summary"""
        if len(self.fifo_queue) < 10:
            return  # Not enough messages to flush
        
        # Keep recent 50% of messages
        keep_count = len(self.fifo_queue) // 2
        evicted = self.fifo_queue[:-keep_count]
        self.fifo_queue = self.fifo_queue[-keep_count:]
        
        # Generate recursive summary
        evicted_text = "\n".join([
            f"{msg.role}: {msg.content}" 
            for msg in evicted
        ])
        
        summary_prompt = f"""Previous summary: {self.queue_summary}

New messages to summarize:
{evicted_text}

Create a concise summary combining the previous summary with new messages. 
Focus on user preferences, requests, and important context."""
        
        self.queue_summary = self.llm.invoke([
            SystemMessage(content=summary_prompt)
        ]).content
        
        print(f"✅ Flushed {len(evicted)} messages. New summary created.")