# openai_utils.py
import json
import logging
from typing import Dict, Any, List, Optional
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential

# OpenAI import
from openai import OpenAI

# Import from modular utilities
from config_utils import load_config
from db_utils import get_mongodb_instance

# Constants
FUNCTION_CALLING_MODEL = "gpt-4o"
RESPONSE_GENERATION_MODEL = "gpt-4o-mini"
MAX_QUERY_ITERATIONS = 3  # Maximum number of query refinement attempts

# Configure logging
logger = logging.getLogger("openai")

class OpenAIClient:
    """Handles interactions with OpenAI API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI client with API key."""
        try:
            self.client = OpenAI(api_key=config['openai_api_key'])
            self.function_model = config.get('function_model', FUNCTION_CALLING_MODEL)
            self.response_model = config.get('response_model', RESPONSE_GENERATION_MODEL)
            self.timeout = config.get('openai_timeout', 30)
            logger.info(f"OpenAI client initialized with models: {self.function_model} and {self.response_model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI's embedding model."""
        try:
            # Truncate text if too long
            if len(text) > 4000:
                text = text[:4000]
                
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
    def should_search_knowledge_base(self, query: str) -> Dict[str, Any]:
        """Determine if the query requires searching the knowledge base."""
        try:
            # Define the function for tool calling
            tools = [{
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": "Search the Confluent documentation to find information that helps answer the query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "should_search": {
                                "type": "boolean",
                                "description": "Whether to search the knowledge base or not. Set to true if the query is about Confluent products, services, or documentation."
                            },
                            "search_query": {
                                "type": "string",
                                "description": "The refined search query to use when querying the knowledge base."
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of top results to return, between 1 and 10."
                            }
                        },
                        "required": ["should_search", "search_query", "num_results"],
                        "additionalProperties": False
                    }
                }
            }]
            
            messages = [
                {"role": "system", "content": "You are an assistant that helps determine if a user query requires searching Confluent documentation."},
                {"role": "user", "content": query}
            ]
            
            response = self.client.chat.completions.create(
                model=self.function_model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            # Extract the function call if available
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                function_args = json.loads(tool_call.function.arguments)
                logger.info(f"Function call decision: {function_args}")
                return function_args
            else:
                # Default behavior if no tool call is made
                logger.info("No tool call made, defaulting to no search")
                return {
                    "should_search": False,
                    "search_query": query,
                    "num_results": 3
                }
                
        except Exception as e:
            logger.error(f"Failed to determine if knowledge base search is needed: {str(e)}")
            # Default to search on failure
            return {
                "should_search": True,
                "search_query": query,
                "num_results": 3
            }

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
    def evaluate_context_relevance(self, original_query: str, retrieved_context: str) -> Dict[str, Any]:
        """Evaluate if the retrieved context is relevant to answer the original query."""
        try:
            tools = [{
                "type": "function",
                "function": {
                    "name": "evaluate_context",
                    "description": "Evaluate if the retrieved context can adequately answer the original user query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_relevant": {
                                "type": "boolean",
                                "description": "Whether the context contains sufficient information to answer the original query."
                            },
                            "relevance_score": {
                                "type": "number",
                                "description": "Relevance score from 0.0 to 1.0, where 1.0 means perfect relevance."
                            },
                            "missing_information": {
                                "type": "string",
                                "description": "What specific information is missing or needed to better answer the query."
                            },
                            "suggested_refined_query": {
                                "type": "string",
                                "description": "A refined search query that might retrieve more relevant context."
                            }
                        },
                        "required": ["is_relevant", "relevance_score", "missing_information", "suggested_refined_query"],
                        "additionalProperties": False
                    }
                }
            }]
            
            evaluation_prompt = f"""
            Original User Query: {original_query}
            
            Retrieved Context:
            {retrieved_context}
            
            Evaluate if this context can adequately answer the original query. Consider:
            1. Does the context directly address the user's question?
            2. Is there enough detail to provide a comprehensive answer?
            3. Are there gaps in information that would prevent a good response?
            """
            
            messages = [
                {"role": "system", "content": "You are an expert at evaluating document relevance for question answering."},
                {"role": "user", "content": evaluation_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.function_model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                evaluation_result = json.loads(tool_call.function.arguments)
                logger.info(f"Context evaluation: {evaluation_result}")
                return evaluation_result
            else:
                # Default to relevant if no evaluation made
                return {
                    "is_relevant": True,
                    "relevance_score": 0.7,
                    "missing_information": "No specific gaps identified",
                    "suggested_refined_query": original_query
                }
                
        except Exception as e:
            logger.error(f"Failed to evaluate context relevance: {str(e)}")
            # Default to considering context as relevant on error
            return {
                "is_relevant": True,
                "relevance_score": 0.5,
                "missing_information": "Evaluation failed",
                "suggested_refined_query": original_query
            }

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
    def generate_response(self, query: str, context: Optional[str] = None) -> str:
        """Generate a response using OpenAI's chat model."""
        try:
            # Prepare system message based on whether context is provided
            if context:
                system_content = f"""You are a helpful assistant for Confluent that answers questions based on the provided context. 
                    Your task:
                    1. Carefully analyze the provided CONTEXT sections below
                    2. Extract relevant information that directly addresses the user's question
                    3. Provide a comprehensive, detailed answer based ONLY on the information in the context
                    4. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation
                    5. Format your response with clear headings, bullet points, and code examples (if relevant)
                    6. Never make up information that is not in the context

                    CONTEXT:
                    {context}
                    Remember: Base your answer ONLY on the information provided in the context. If information is missing, say "Based on the available documentation, I don't have complete information about [specific topic]" rather than making up details."""
            else:
                system_content = """You are a helpful assistant for Confluent. 
                If you don't know the answer, simply say so. Don't make up information."""
            
            logger.info(f"\n\n---- prompt ------\n\n {system_content}\n\n question = {query}")
            messages = [
                {"role": "user", "content": system_content+"\n"+query}
            ]
            
            response = self.client.chat.completions.create(
                model=self.response_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return "I'm sorry, I encountered an error while generating a response."

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
    def rewrite_query(self, query: str) -> str:
        """Rewrite the query to make it more suitable for vector search."""
        try:
            system_content = """Rewrite the query to focus on specific technical terms.
Keep under 30 words. Preserve exact product names."""
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Rewrite this query for vector search: {query}"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.response_model,
                messages=messages,
                temperature=0.1,
                max_tokens=200
            )
            
            rewritten_query = response.choices[0].message.content.strip()
            logger.info(f"Original query: '{query}'\n========\nRewritten query: '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Failed to rewrite query: {str(e)}")
            return query

# Singleton pattern for OpenAI client
_openai_client = None

def get_openai_client() -> OpenAIClient:
    """Get or create OpenAI client using singleton pattern."""
    global _openai_client
    if _openai_client is None:
        config = load_config()
        _openai_client = OpenAIClient(config)
    return _openai_client

class ConfluentRAG:
    """Retrieval-Augmented Generation system for Confluent documentation with iterative query refinement."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the RAG system with configuration."""
        try:
            self.config = load_config(config_path)
            self.mongodb = get_mongodb_instance()
            self.openai_client = get_openai_client()
            logger.info("ConfluentRAG initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ConfluentRAG: {str(e)}")
            raise

    def iterative_context_retrieval(self, original_query: str, max_iterations: int = MAX_QUERY_ITERATIONS) -> tuple[str, List[Dict[str, Any]]]:
        """
        Iteratively refine query and retrieve context until satisfactory results are found.
        Returns the best context found and the search history.
        """
        search_history = []
        best_context = ""
        best_relevance_score = 0.0
        current_query = original_query
        
        logger.info(f"Starting iterative context retrieval for: '{original_query}'")
        
        for iteration in range(max_iterations):
            logger.info(f"Iteration {iteration + 1}/{max_iterations} - Searching with query: '{current_query}'")
            
            try:
                # Rewrite query for vector search
                rewritten_query = self.openai_client.rewrite_query(current_query)
                
                # Get embedding and search
                query_embedding = self.openai_client.get_embedding(rewritten_query)
                results = self.mongodb.vector_search(
                    query_embedding, 
                    limit=3,
                    query_text=rewritten_query
                )
                
                if not results:
                    logger.warning(f"No results found in iteration {iteration + 1}")
                    search_history.append({
                        "iteration": iteration + 1,
                        "query": current_query,
                        "rewritten_query": rewritten_query,
                        "results_count": 0,
                        "relevance_score": 0.0,
                        "evaluation": "No results found"
                    })
                    continue
                
                # Extract context from results
                context = "\n\n".join([doc["chunk_content"] for doc in results])
                
                # Evaluate context relevance
                evaluation = self.openai_client.evaluate_context_relevance(original_query, context)
                
                # Log the evaluation
                search_history.append({
                    "iteration": iteration + 1,
                    "query": current_query,
                    "rewritten_query": rewritten_query,
                    "results_count": len(results),
                    "relevance_score": evaluation.get("relevance_score", 0.0),
                    "evaluation": evaluation
                })
                
                # Check if this is the best context so far
                current_score = evaluation.get("relevance_score", 0.0)
                if current_score > best_relevance_score:
                    best_relevance_score = current_score
                    best_context = context
                
                # If context is highly relevant, stop iterating
                if evaluation.get("is_relevant", False) and current_score >= 0.8:
                    logger.info(f"Found highly relevant context (score: {current_score}) in iteration {iteration + 1}")
                    break
                
                # If context is moderately relevant and we're not on the last iteration, continue
                if current_score >= 0.6 and iteration < max_iterations - 1:
                    logger.info(f"Found moderately relevant context (score: {current_score}), trying to improve...")
                    current_query = evaluation.get("suggested_refined_query", current_query)
                elif current_score < 0.6 and iteration < max_iterations - 1:
                    logger.info(f"Low relevance context (score: {current_score}), refining query...")
                    current_query = evaluation.get("suggested_refined_query", current_query)
                else:
                    logger.info(f"Final iteration reached with score: {current_score}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in iteration {iteration + 1}: {str(e)}")
                search_history.append({
                    "iteration": iteration + 1,
                    "query": current_query,
                    "error": str(e)
                })
                break
        
        logger.info(f"Iterative search completed. Best relevance score: {best_relevance_score}")
        return best_context, search_history

    def answer_question(self, query: str) -> str:
        """Generate an answer for the user's question using iterative context retrieval."""
        try:
            # Check for empty query
            if not query or not query.strip():
                return "Please ask a question about Confluent."
            
            # Use function calling to determine if we should search the knowledge base
            search_decision = self.openai_client.should_search_knowledge_base(query)
            
            # If the model decides we should search
            if search_decision["should_search"]:
                logger.info("Starting iterative context retrieval process...")
                
                # Use iterative context retrieval
                context, search_history = self.iterative_context_retrieval(query)
                
                if context:
                    logger.info(f"Final context retrieved with {len(search_history)} iterations")
                    
                    # Add search history to logs for debugging
                    for entry in search_history:
                        logger.debug(f"Search iteration {entry.get('iteration', 'N/A')}: "
                                   f"Score={entry.get('relevance_score', 'N/A')}, "
                                   f"Results={entry.get('results_count', 'N/A')}")
                    
                    # Generate response using the best context found
                    response = self.openai_client.generate_response(query, context)
                    
                    # Optionally append search quality info for transparency
                    best_score = max([entry.get('relevance_score', 0.0) for entry in search_history], default=0.0)
                    if best_score < 0.6:
                        response += f"\n\n*Note: The information found may not fully address your question. You might want to rephrase your query for better results.*"
                    
                    return response
                else:
                    # No context found even after iterations
                    return self.openai_client.generate_response(
                        query,
                        "I searched the Confluent documentation thoroughly but couldn't find specific information about this topic."
                    )
            else:
                # Model decided no need to search, generate response directly
                logger.info("Skipping knowledge base search as per model decision")
                return self.openai_client.generate_response(query)
                
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return "I'm sorry, I encountered an error while processing your question."
