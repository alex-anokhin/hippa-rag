import os
import json
import gradio as gr
import httpx
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get backend URL from environment or use default
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8080")

def format_source(source):
	"""Format a source reference for display"""
	metadata = source.get("metadata", {}) or {}

	# Build citation parts list
	citation_parts = []

	# Part reference with correction for all three HIPAA parts
	if metadata.get("part"):
		part_num = metadata["part"]
		
		# Fix truncated part numbers
		if part_num == "16":
			# Check section to determine actual part
			section = metadata.get("section", "")
			
			if section.startswith("160."):
				part_num = "160"
			elif section.startswith("164."):
				part_num = "164"
			else:
				# Default to 162 if can't determine
				part_num = "162"
		
		citation_parts.append(f"Part {part_num}")

	# Section reference with part-specific correction
	if metadata.get("section"):
		section = metadata["section"]
		# Remove spaces in section numbers
		section = re.sub(r'\s+', '', section)
		
		# Fix truncated sections based on detected part
		if section.startswith("16."):
			part_num = next((part for part in citation_parts if part.startswith("Part ")), "")
			
			if "Part 160" in part_num:
				section = "160" + section[2:]
			elif "Part 164" in part_num:
				section = "164" + section[2:]
			else:
				section = "162" + section[2:]
				
		citation_parts.append(f"§ {section}")

	# Subpart reference
	if metadata.get("subpart"):
		if metadata.get("subpart_title"):
			citation_parts.append(f"Subpart {metadata['subpart']} - {metadata['subpart_title']}")
		else:
			citation_parts.append(f"Subpart {metadata['subpart']}")

	# Build the final source string
	content = source.get("content", "").replace("\n", " ").strip()
	content = re.sub(r'§\s+(\d+)\s+\.(\d+)', r'§ \1.\2', content)
	content = re.sub(r'(\d+)\s+\.(\d+)', r'\1.\2', content)

	if citation_parts:
		citation = " | ".join(citation_parts)
		# Add first 60 chars of content if available
		if content:
			preview = content[:60] + "..." if len(content) > 60 else content
			return f"{citation}\n   \"{preview}\""
		return citation
	else:
		# Fallback to showing clean content
		preview = content[:100] + "..." if len(content) > 100 else content
		return f"\"{preview}\""

async def add_user_message(message, history):
    """Immediately add the user message to the chat history"""
    if not message.strip():
        return "", history

    # Create a copy of the history to avoid modifying the original
    new_history = history.copy()
    
    # Add user message immediately
    new_history.append({"role": "user", "content": message})
    
    # Return empty message input and updated history with user message
    return "", new_history

async def get_bot_response(history):
    """Get response from backend after user message is displayed"""
    if not history or len(history) == 0:
        return history

    # Get the last user message
    user_message = None
    for msg in reversed(history):
        if msg["role"] == "user":
            user_message = msg["content"]
            break

    if not user_message:
        return history

    # Create a copy to modify
    new_history = history.copy()

    try:
        async with httpx.AsyncClient() as client:
            # Create message history for context (excluding the last user message)
            message_history = []
            for msg in history[:-1]:  # All messages except the last user message
                message_history.append(msg)
            
            # Send request to backend
            response = await client.post(
                f"{BACKEND_URL}/api/query",
                json={
                    "query": user_message,
                    "history": message_history
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "")
                
                # Format sources
                if "sources" in result:
                    source_text = "\n\n**Sources:**\n"
                    seen_citations = set()
                    
                    for source in result["sources"]:
                        # Your existing source formatting code...
                        # [KEEP YOUR EXISTING SOURCE FORMATTING CODE HERE]
                        citation = ""
                        
                        # Try to get formatted citation first
                        if "formatted_citation" in source and source["formatted_citation"] != "HIPAA Reference":
                            citation = source["formatted_citation"]
                        else:
                            # Create a citation from metadata
                            metadata = source.get("metadata", {}) or {}
                            content = source.get("content", "")
                            
                            # Extract section number first (to infer part if needed)
                            section = None
                            if metadata.get("section"):
                                section = re.sub(r'\s+', '', metadata["section"])
                            
                            # Get part number or infer it from section
                            part_num = None
                            if metadata.get("part"):
                                part_num = metadata["part"]
                            elif section:
                                # Infer part from section number
                                if section.startswith("160."):
                                    part_num = "160"
                                elif section.startswith("162."):
                                    part_num = "162"
                                elif section.startswith("164."):
                                    part_num = "164"
                            
                            # Build citation
                            parts = []
                            if part_num:
                                if part_num == "16":
                                    if section and section.startswith("160."):
                                        part_num = "160"
                                    elif section and section.startswith("164."):
                                        part_num = "164"
                                    else:
                                        part_num = "162"
                                parts.append(f"Part {part_num}")
                                
                            if section:
                                if section.startswith("16."):
                                    if part_num == "160":
                                        section = f"160{section[2:]}"
                                    elif part_num == "164":
                                        section = f"164{section[2:]}"
                                    else:
                                        section = f"162{section[2:]}"
                                parts.append(f"§ {section}")
                                
                            citation = " | ".join(parts) if parts else ""
                            
                            # If still no citation, try to extract from content
                            if not citation and content:
                                section_match = re.search(r'§\s*(\d{3}\.\d{3})', content)
                                if section_match:
                                    section = section_match.group(1)
                                    part = section[:3]
                                    citation = f"Part {part} | § {section}"
                                else:
                                    part_match = re.search(r'Part (\d{3})', content)
                                    if part_match:
                                        citation = f"Part {part_match.group(1)}"
                        
                        # Only add non-empty, non-duplicate citations
                        if citation and citation not in seen_citations:
                            source_text += f"\n- {citation}"
                            seen_citations.add(citation)
                    
                    full_answer = answer + source_text
                else:
                    full_answer = answer
                
                # Add assistant response to history
                new_history.append({"role": "assistant", "content": full_answer})
            else:
                error_msg = f"Error: Unable to get response from the backend. Status code: {response.status_code}"
                new_history.append({"role": "assistant", "content": error_msg})

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        new_history.append({"role": "assistant", "content": error_msg})
    
    return new_history

# Create Gradio interface
with gr.Blocks(title="HIPAA RAG Assistant") as demo:
	gr.Markdown("# HIPAA Regulations Assistant")
	gr.Markdown("Ask questions about HIPAA regulations and get accurate answers with source references.")

	with gr.Tab("Chat"):
		# Create a chatbot component with history
		chat_history = gr.Chatbot(height=500, label="Chat History", type="messages")
		
		# Create input components with a send button
		with gr.Row():
			msg_input = gr.Textbox(
				label="Ask a question about HIPAA",
				placeholder="What is the purpose of HIPAA Part 160?",
				lines=2,
				show_label=True,
				scale=9,
				autofocus=True  # Focus on page load
			)
			send_btn = gr.Button("Send", variant="primary", scale=1)
			
			# First, add the user message immediately
			send_btn.click(
				fn=add_user_message,
				inputs=[msg_input, chat_history],
				outputs=[msg_input, chat_history],
				queue=False  # Execute immediately without queueing
			).then(  # Then get the bot response
				fn=get_bot_response,
				inputs=[chat_history],
				outputs=[chat_history]
			)
		
		# Clear button
		clear_btn = gr.Button("Clear History", variant="secondary")
		
		# Also handle Enter key submission
		msg_input.submit(
			fn=add_user_message,
			inputs=[msg_input, chat_history],
			outputs=[msg_input, chat_history],
			queue=False
		).then(
			fn=get_bot_response,
			inputs=[chat_history],
			outputs=[chat_history]
		)
		
		# Clear history button
		clear_btn.click(lambda: (None, None), None, [msg_input, chat_history], queue=False)
		
		# Add a footer
		gr.Markdown(
			"""
			<div style="text-align: center; margin-top: 20px;">
				<p>Powered by FastAPI, Gradio, OpenAI, and PostgreSQL</p>
				<p>© 2025 HIPAA RAG Assistant</p>
			</div>
			""",
			elem_id="footer"
		)

	with gr.Tab("About"):
		gr.Markdown("""
		# HIPAA RAG Assistant
		
		This application helps you find information in HIPAA regulations using:
		
		- Natural language questions to query HIPAA documents
		- AI-powered search to find relevant sections
		- Accurate answers with source citations
		- Document management with vector database
		
		Built with FastAPI, Gradio, OpenAI, and PostgreSQL vector database.
		""")

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)