# ui.py
import asyncio
import logging
import streamlit as st
from pathlib import Path
import time

# Import from modular utilities
from config_utils import load_config
from kafka_utils import process_url_and_send_to_kafka
from openai_utils import ConfluentRAG
from db_utils import get_stored_urls, store_processed_url

# Configure logging for UI components
logger = logging.getLogger("ui")

def create_streamlit_app():
    """Create and configure the Streamlit app with integrated chunker and RAG assistant."""
    
    st.set_page_config(
        page_title="Confluent Documentation Tools",
        page_icon="💬",
        layout="wide"
    )
    
    # Initialize RAG system
    try:
        if 'rag' not in st.session_state:
            st.session_state.rag = ConfluentRAG()
            st.session_state.config = load_config()
        
        # Initialize session state variables
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        
        if 'show_search_details' not in st.session_state:
            st.session_state.show_search_details = False
            
    except Exception as e:
        st.error(f"Failed to initialize the application: {str(e)}")
        return
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["🤖 Documentation Assistant", "🔗 URL Chunker", "📋 Processed URLs"])
    
    # Tab 1: Documentation Assistant
    with tab1:
        st.title("🚀 Real-time RAG using Confluent")
        st.markdown("Ask me anything about Confluent Cloud and documentation!")
        
        # Create a container for chat messages
        chat_container = st.container()
        
        # Display chat history in the container
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Show search details if available
                    if message["role"] == "assistant" and "search_history" in message:
                        if st.session_state.show_search_details:
                            with st.expander("🔍 Search Process Details", expanded=False):
                                search_history = message["search_history"]
                                
                                for entry in search_history:
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    
                                    with col1:
                                        st.metric("Iteration", entry.get("iteration", "N/A"))
                                    
                                    with col2:
                                        st.text(f"Query: {entry.get('query', 'N/A')}")
                                        if entry.get('rewritten_query'):
                                            st.text(f"Rewritten: {entry.get('rewritten_query', 'N/A')}")
                                    
                                    with col3:
                                        score = entry.get('relevance_score', 0.0)
                                        if score > 0:
                                            st.metric("Relevance", f"{score:.2f}")
                                        st.metric("Results", entry.get('results_count', 0))
                                    
                                    st.markdown("---")
        
        # Chat input at the bottom
        if prompt := st.chat_input("Ask your question about Confluent...", disabled=st.session_state.processing):
            # Immediately add user message to history and display
            user_message = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_message)
            
            # Display the new user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            # Set processing state to prevent new messages
            st.session_state.processing = True
            
            # Create placeholder for assistant response
            with chat_container:
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    
                    # Show detailed processing steps
                    with st.status("🤔 Processing your question...", expanded=True) as status:
                        try:
                            # Step 1: Determine if search is needed
                            st.write("🔍 Analyzing question...")
                            search_decision = st.session_state.rag.openai_client.should_search_knowledge_base(prompt)
                            
                            if search_decision["should_search"]:
                                st.write("📚 Knowledge base search required")
                                st.write("🔄 Starting iterative context retrieval...")
                                
                                # Step 2: Iterative context retrieval
                                context, search_history = st.session_state.rag.iterative_context_retrieval(prompt)
                                
                                # Show search progress
                                for i, entry in enumerate(search_history):
                                    score = entry.get('relevance_score', 0.0)
                                    results_count = entry.get('results_count', 0)
                                    st.write(f"   • Iteration {i+1}: Found {results_count} results (relevance: {score:.2f})")
                                
                                if context:
                                    st.write("✅ Relevant context found")
                                    st.write("🤖 Generating response...")
                                    
                                    # Step 3: Generate response
                                    response = st.session_state.rag.openai_client.generate_response(prompt, context)
                                    
                                    # Update status
                                    status.update(label="✅ Response generated!", state="complete")
                                    
                                    # Display final response
                                    response_placeholder.markdown(response)
                                    
                                    # Add assistant response to history with search details
                                    assistant_message = {
                                        "role": "assistant", 
                                        "content": response,
                                        "search_history": search_history
                                    }
                                    st.session_state.messages.append(assistant_message)
                                    
                                    # Show search quality indicator
                                    best_score = max([entry.get('relevance_score', 0.0) for entry in search_history], default=0.0)
                                    if best_score >= 0.8:
                                        st.success(f"🎯 High quality search results (score: {best_score:.2f})")
                                    elif best_score >= 0.6:
                                        st.info(f"✅ Good search results (score: {best_score:.2f})")
                                    else:
                                        st.warning(f"⚠️ Limited search results (score: {best_score:.2f})")
                                else:
                                    st.write("❌ No relevant context found")
                                    response = st.session_state.rag.openai_client.generate_response(
                                        prompt, 
                                        "I searched the Confluent documentation thoroughly but couldn't find specific information about this topic."
                                    )
                                    status.update(label="⚠️ Limited information available", state="complete")
                                    response_placeholder.markdown(response)
                                    
                                    assistant_message = {
                                        "role": "assistant", 
                                        "content": response,
                                        "search_history": search_history
                                    }
                                    st.session_state.messages.append(assistant_message)
                            else:
                                st.write("💭 Generating direct response...")
                                response = st.session_state.rag.openai_client.generate_response(prompt)
                                status.update(label="✅ Response generated!", state="complete")
                                response_placeholder.markdown(response)
                                
                                assistant_message = {"role": "assistant", "content": response}
                                st.session_state.messages.append(assistant_message)
                            
                        except Exception as e:
                            error_message = "❌ I'm sorry, I encountered an error while processing your question. Please try again."
                            status.update(label="❌ Processing failed", state="error")
                            response_placeholder.error(error_message)
                            logger.error(f"Error processing question: {str(e)}")
                            
                            # Add error message to history
                            error_msg = {"role": "assistant", "content": error_message}
                            st.session_state.messages.append(error_msg)
            
            # Reset processing state
            st.session_state.processing = False
            
            # Force rerun to update the interface
            st.rerun()
        
        # Sidebar for documentation assistant
        with st.sidebar:
            st.header("🛠️ Assistant Options")
            
            # Search details toggle
            st.session_state.show_search_details = st.checkbox(
                "Show Search Details",
                value=st.session_state.show_search_details,
                help="Display detailed search process information"
            )
            
            # Chat statistics
            if st.session_state.messages:
                user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
                assistant_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
                searches_with_history = len([m for m in st.session_state.messages if m["role"] == "assistant" and "search_history" in m])
                
                st.metric("Questions Asked", user_msgs)
                st.metric("Responses Given", assistant_msgs)
                st.metric("Knowledge Base Searches", searches_with_history)
                st.markdown("---")
            
            # Chat history management
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.session_state.processing = False
                st.rerun()
            
            # Export chat option
            if st.session_state.messages and st.button("📥 Export Chat", use_container_width=True):
                chat_export = ""
                for msg in st.session_state.messages:
                    role = "You" if msg["role"] == "user" else "Assistant"
                    chat_export += f"**{role}:** {msg['content']}\n\n"
                    
                    # Include search details if available
                    if "search_history" in msg:
                        chat_export += "**Search Details:**\n"
                        for entry in msg["search_history"]:
                            chat_export += f"- Iteration {entry.get('iteration', 'N/A')}: "
                            chat_export += f"Score {entry.get('relevance_score', 'N/A')}, "
                            chat_export += f"Results {entry.get('results_count', 'N/A')}\n"
                        chat_export += "\n"
                
                st.download_button(
                    label="Download Chat",
                    data=chat_export,
                    file_name=f"confluent_chat_{int(time.time())}.md",
                    mime="text/markdown"
                )
            
            # About section
            st.markdown("---")
            st.markdown("### ℹ️ About")
            st.markdown("""
            This assistant uses **Iterative RAG** with smart query refinement:
            
            **Features:**
            - 🔍 Smart knowledge base search
            - 🔄 Iterative query refinement (up to 3 attempts)
            - 🎯 Context relevance evaluation  
            - 📚 Real-time documentation access
            - 💾 Chat history with search details
            """)
            
            # Status indicator
            if st.session_state.processing:
                st.info("🔄 Processing your question...")
            else:
                st.success("✅ Ready for questions")
    
    # Tab 2: URL Chunker (keeping existing implementation)
    with tab2:
        st.title("🔗 URL Chunker and Kafka Streamer")
        st.markdown("Process web content and stream chunks to Kafka for real-time indexing")
        
        # Create form for better UX
        with st.form("url_processor_form", clear_on_submit=False):
            # URL input with validation
            url_input = st.text_input(
                "🌐 Website URL", 
                placeholder="https://docs.confluent.io/cloud/current/overview.html",
                help="Enter the full URL to process and chunk"
            )
            
            # Options in columns
            col1, col2 = st.columns([2, 1])
            with col1:
                chunker_type = st.selectbox(
                    "🔧 Chunker Algorithm",
                    options=["sentence", "semantic"],
                    index=0,
                    help="Choose how to split the content into chunks"
                )
            
            with col2:
                chunk_preview = st.checkbox(
                    "Preview Chunks",
                    value=True,
                    help="Show chunk preview after processing"
                )
            
            # Submit button
            submit_btn = st.form_submit_button(
                "🚀 Process URL", 
                type="primary", 
                use_container_width=True
            )
        
        # Processing status and results
        if submit_btn:
            if not url_input:
                st.error("❌ Please enter a URL to process")
            elif not url_input.startswith(("http://", "https://")):
                st.error("❌ Please enter a valid URL starting with http:// or https://")
            else:
                # Processing section
                with st.status("Processing URL...", expanded=True) as status:
                    st.write("🔍 Fetching content from URL...")
                    st.write("✂️ Chunking content...")
                    st.write("📤 Streaming to Kafka...")
                    
                    try:
                        # Process URL
                        result = asyncio.run(process_url_and_send_to_kafka(
                            url_input, 
                            st.session_state.config, 
                            chunker_type
                        ))
                        
                        if not result or not result.get("success"):
                            error_msg = result.get("error") if result else "Unknown error"
                            status.update(label="❌ Processing failed", state="error")
                            st.error(f"Error processing URL: {error_msg}")
                        else:
                            chunks = result.get("chunks", [])
                            status.update(label="✅ Processing completed!", state="complete")
                            
                            # Success metrics
                            col1, col2, col3 = st.columns(3)
                            col1.metric("📄 Chunks Created", len(chunks))
                            col2.metric("🔧 Chunker Type", chunker_type.title())
                            col3.metric("📊 Total Words", sum(c.get('metadata', {}).get('word_count', 0) for c in chunks))
                            
                            # Store in database
                            if store_processed_url(url_input, chunker_type, len(chunks)):
                                st.success("✅ URL information stored in database")
                            
                            # Show chunks preview if requested
                            if chunk_preview and chunks:
                                st.markdown("### 📋 Chunk Preview")
                                
                                # Chunk navigation
                                if len(chunks) > 1:
                                    chunk_idx = st.selectbox(
                                        "Select chunk to preview:",
                                        range(len(chunks)),
                                        format_func=lambda x: f"Chunk {x+1}/{len(chunks)} ({chunks[x].get('metadata', {}).get('word_count', 0)} words)"
                                    )
                                else:
                                    chunk_idx = 0
                                
                                # Display selected chunk
                                chunk = chunks[chunk_idx]
                                
                                # Chunk metadata
                                metadata_cols = st.columns(4)
                                metadata_cols[0].metric("Type", chunk.get('type', 'text'))
                                metadata_cols[1].metric("Words", chunk.get('metadata', {}).get('word_count', 0))
                                metadata_cols[2].metric("Position", chunk.get('metadata', {}).get('position', 0))
                                metadata_cols[3].metric("Content Type", chunk.get('metadata', {}).get('type', 'text'))
                                
                                # Chunk content
                                st.markdown("**Content:**")
                                st.code(chunk.get('content', ''), language='text')
                            
                            elif not chunks:
                                st.warning("⚠️ No chunks were created from the processed URL")
                                
                    except Exception as e:
                        status.update(label="❌ Processing failed", state="error")
                        logger.error(f"Error in URL processing: {str(e)}", exc_info=True)
                        st.error(f"An error occurred: {str(e)}")
    
    # Tab 3: Processed URLs
    with tab3:
        st.title("📋 Processed URLs")
        st.markdown("View and manage all previously processed URLs")
        
        # Header controls
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        # Get and display stored URLs
        try:
            urls = get_stored_urls()
            
            if not urls:
                st.info("📭 No URLs have been processed yet. Go to the URL Chunker tab to get started!")
            else:
                # Summary metrics
                total_urls = len(urls)
                total_chunks = sum(url.get('chunk_count', 0) for url in urls)
                chunker_types = list(set(url.get('chunker_type', 'unknown') for url in urls))
                
                col1, col2, col3 = st.columns(3)
                col1.metric("🔗 Total URLs", total_urls)
                col2.metric("📄 Total Chunks", total_chunks)
                col3.metric("🔧 Chunker Types", len(chunker_types))
                
                st.markdown("---")
                
                # Search and filter
                search_term = st.text_input("🔍 Search URLs", placeholder="Enter URL or keyword...")
                selected_chunker = st.selectbox(
                    "Filter by Chunker Type", 
                    ["All"] + chunker_types,
                    index=0
                )
                
                # Filter URLs
                filtered_urls = urls
                if search_term:
                    filtered_urls = [url for url in filtered_urls 
                                   if search_term.lower() in url.get('url', '').lower()]
                if selected_chunker != "All":
                    filtered_urls = [url for url in filtered_urls 
                                   if url.get('chunker_type') == selected_chunker]
                
                st.markdown(f"### 📊 Showing {len(filtered_urls)} of {total_urls} URLs")
                
                # Display URLs in a better format
                for i, url in enumerate(filtered_urls):
                    with st.expander(f"🔗 {url['url']}", expanded=False):
                        info_cols = st.columns(4)
                        info_cols[0].metric("Chunker Type", url.get('chunker_type', 'unknown'))
                        info_cols[1].metric("Chunk Count", url.get('chunk_count', 0))
                        processed_at = url.get('processed_at', 'unknown')
                        if hasattr(processed_at, 'strftime'):
                            processed_at = processed_at.strftime('%Y-%m-%d %H:%M')
                        info_cols[2].metric("Processed", str(processed_at))
                        
                        # Action buttons
                        action_cols = st.columns(3)
                        with action_cols[0]:
                            if st.button(f"🌐 Open URL", key=f"open_{i}"):
                                st.markdown(f"[Open in new tab]({url['url']})")
                        with action_cols[1]:
                            if st.button(f"🔄 Reprocess", key=f"reprocess_{i}"):
                                st.info("Feature coming soon!")
                        with action_cols[2]:
                            if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                                st.warning("Feature coming soon!")
                
                # Pagination for large lists
                if len(filtered_urls) > 10:
                    st.markdown("---")
                    st.info(f"💡 Showing first 10 results. Use search to find specific URLs.")
                    
        except Exception as e:
            logger.error(f"Error retrieving stored URLs: {str(e)}")
            st.error("❌ Failed to load processed URLs. Please try refreshing the page.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit • Confluent Cloud • OpenAI"
        "</div>", 
        unsafe_allow_html=True
    )
