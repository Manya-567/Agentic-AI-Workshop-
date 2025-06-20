
import os
import json
import streamlit as st
from typing import Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import markdown
import time
import logging
from google.api_core.exceptions import ResourceExhausted

# Set up logging
logging.basicConfig(
    filename="agent_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get Google API key with fallback
try:
    GOOGLE_API_KEY = "AIzaSyCQHeljXr58nW5nVh2QSMGs4-jlnKfGAno"
    if not GOOGLE_API_KEY:     
        raise ValueError("No Google API key found.")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except Exception as e:
    st.error(f"Failed to load Google API key: {str(e)}")
    st.error("Set GOOGLE_API_KEY in .streamlit/secrets.toml or environment variable.")
    st.stop()

# Initialize Gemini LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
except Exception as e:
    st.error(f"Failed to initialize Gemini LLM: {str(e)}")
    logging.error(f"Gemini LLM initialization failed: {str(e)}")
    st.stop()

# Initialize MiniLM embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# User-defined RAG vector store with default file
def initialize_user_rag():
    persist_directory = "./chroma_db_minilm"
    default_rag_file = "default_rag.txt"
    
    if not os.path.exists(default_rag_file):
        with open(default_rag_file, "w") as f:
            f.write("WCAG 2.1: Use high contrast ratios. Ensure keyboard navigation. Support voice commands.\n")
            f.write("UI Best Practices: Map 'Tap Start' to navigate to meditation_screen.\n")
        logging.info("Created default RAG file.")

    uploaded_file = st.file_uploader("Upload a text file for RAG (e.g., guidelines)", type="txt")
    if uploaded_file is not None:
        try:
            with open("temp_rag.txt", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = TextLoader("temp_rag.txt")
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
            st.success("RAG vector store updated with uploaded file.")
            logging.info("RAG vector store updated with uploaded file.")
            return vectorstore
        except Exception as e:
            st.error(f"Failed to process uploaded file: {str(e)}")
            logging.error(f"Failed to process uploaded file: {str(e)}")
            return Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    else:
        try:
            loader = TextLoader(default_rag_file)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
            logging.info("RAG vector store initialized with default file.")
            return vectorstore
        except Exception as e:
            st.error(f"Failed to initialize default RAG: {str(e)}")
            logging.error(f"Failed to initialize default RAG: {str(e)}")
            return Chroma(embedding_function=embeddings, persist_directory=persist_directory)

# Initialize vector store
vectorstore = initialize_user_rag()

# RAG chain with caching
@st.cache_resource
def get_rag_chain():
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )
rag_chain = get_rag_chain()

# Custom tools for agents
def map_input_to_state(input_data: str) -> str:
    """Maps user input to app state changes using RAG with fallback."""
    last_rag_result: Optional[str] = None
    for attempt in range(4):
        try:
            # Validate and parse JSON input
            if not input_data or not isinstance(input_data, str):
                raise ValueError("Invalid or empty input data")
            input_json = json.loads(input_data)
            user_input = input_json.get("user_input", "").strip()
            app_purpose = input_json.get("app_purpose", "").strip()
            if not user_input or not app_purpose:
                raise ValueError("Missing user_input or app_purpose")
            
            query = f"State for '{user_input}' in {app_purpose}"
            logging.info(f"RAG Query: {query}")
            rag_result = rag_chain({"query": query})
            result_text = rag_result.get("result", "")
            
            if not result_text or not isinstance(result_text, str):
                logging.warning("RAG returned empty result, using fallback state")
                result_text = "default_state"
            else:
                result_text = result_text.strip().split("->")[-1].strip() if "->" in result_text else result_text
                last_rag_result = result_text
            
            logging.info(f"RAG Result: {result_text}")
            
            state_changes = {
                "input": user_input,
                "text_map": f"User {user_input} -> Navigates to next screen",
                "onscreen": "Layout updates",
                "offscreen": "Triggers update",
                "schema": {
                    "trigger": user_input.lower().replace(" ", "_"),
                    "action": "navigate",
                    "state": result_text.lower().replace(" ", "_")
                }
            }
            return json.dumps(state_changes, indent=2)
        except ResourceExhausted as e:
            if attempt < 3:
                delay = 9
                logging.warning(f"Quota exceeded, retrying: {str(e)}, delay: {delay}s")
                time.sleep(delay)
            else:
                st.error(f"Quota exceeded after retries: {str(e)}. Using last RAG result. Check quota at https://ai.google.dev/gemini-api/docs/rate-limits (Limit: 15 requests/min, retry after 9s).")
                logging.error(f"Quota exceeded after retries: {str(e)}")
                if last_rag_result:
                    state_changes = {
                        "input": input_json.get("user_input", "unknown"),
                        "text_map": f"User {input_json.get('user_input', 'unknown')} -> Navigates to next screen",
                        "onscreen": "Layout updates",
                        "offscreen": "Triggers update",
                        "schema": {
                            "trigger": input_json.get("user_input", "unknown").lower().replace(" ", "_"),
                            "action": "navigate",
                            "state": last_rag_result.lower().replace(" ", "_")
                        }
                    }
                    return json.dumps(state_changes, indent=2)
                return json.dumps({
                    "status": "error",
                    "message": f"Quota exceeded after retries: {str(e)}",
                    "fallback": {
                        "input": input_json.get("user_input", "unknown"),
                        "text_map": "Navigation failed due to quota limit",
                        "onscreen": "Error message displayed",
                        "offscreen": "No update",
                        "schema": {"trigger": "unknown", "action": "none", "state": "error"}
                    }
                }, indent=2)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON input: {str(e)} - Input: {input_data}")
            logging.error(f"Invalid JSON input: {str(e)} - Input: {input_data}")
            return json.dumps({"status": "error", "message": f"Invalid JSON input: {str(e)}"}, indent=2)
        except Exception as e:
            st.error(f"Input-Response Mapping failed: {str(e)} - Input: {input_data}")
            logging.error(f"Input-Response Mapping failed: {str(e)} - Input: {input_data}")
            return json.dumps({"status": "error", "message": f"Input-Response Mapping failed: {str(e)}"}, indent=2)

def suggest_accessibility(input_data: str) -> str:
    """Suggests accessible interaction strategies using RAG."""
    for attempt in range(4):
        try:
            input_json = json.loads(input_data)
            user_activity = input_json.get("user_activity", "").strip()
            target_audience = input_json.get("target_audience", "").strip()
            query = f"WCAG for {user_activity} targeting {target_audience}"
            logging.info(f"RAG Query: {query}")
            rag_result = rag_chain({"query": query})
            result_text = rag_result.get("result", "")
            
            if not result_text or not isinstance(result_text, str):
                result_text = "Ensure high contrast and keyboard access"
                logging.warning("RAG returned empty result for accessibility")
            else:
                result_text = result_text.strip()
            
            accessibility_plan = {
                "accessibility_plan": {
                    "keyboard": "Tab navigation",
                    "touch": "48x48px targets",
                    "voice": "Voice command support",
                    "alt_descriptions": f"Button: {user_activity}",
                    "sound_feedback": "Chime",
                    "color_contrast": result_text
                }
            }
            return json.dumps(accessibility_plan, indent=2)
        except ResourceExhausted as e:
            if attempt < 3:
                delay = 9
                logging.warning(f"Quota exceeded, retrying: {str(e)}, delay: {delay}s")
                time.sleep(delay)
            else:
                st.error(f"Quota exceeded after retries: {str(e)}. Check quota at https://ai.google.dev/gemini-api/docs/rate-limits (Limit: 15 requests/min, retry after 9s).")
                logging.error(f"Quota exceeded after retries: {str(e)}")
                return json.dumps({"status": "error", "message": f"Quota exceeded after retries: {str(e)}"}, indent=2)
        except Exception as e:
            st.error(f"Accessibility Design failed: {str(e)}")
            logging.error(f"Accessibility Design failed: {str(e)}")
            return json.dumps({"status": "error", "message": f"Accessibility Design failed: {str(e)}"}, indent=2)

def generate_ui_blueprint(input_data: str) -> str:
    """Generates textual UI blueprints."""
    try:
        input_json = json.loads(input_data)
        user_journey = input_json.get("user_journey", "").strip()
        blueprint = {
            "screen": "Home",
            "components": [
                {"type": "header", "text": "Welcome"},
                {"type": "button", "text": user_journey},
                {"type": "icon", "text": "Gear"}
            ],
            "dynamic_states": f"If '{user_journey}' tapped, expand input"
        }
        logging.info("Generated UI blueprint")
        return json.dumps(blueprint, indent=2)
    except Exception as e:
        st.error(f"UI Blueprint Generation failed: {str(e)}")
        logging.error(f"UI Blueprint Generation failed: {str(e)}")
        return json.dumps({"status": "error", "message": f"UI Blueprint Generation failed: {str(e)}"}, indent=2)

def generate_wiring(input_data: str) -> str:
    """Generates screen-to-screen transition logic."""
    try:
        input_json = json.loads(input_data)
        screen = input_json.get("screen", "Home").strip()
        user_journey = input_json.get("user_journey", "").strip()
        flow = {
            "flow": f"{screen} -> '{user_journey}' -> Home",
            "conditional_logic": f"if invalid input -> show toast",
            "mermaid": f"graph TD A[{screen}] -->|{user_journey}| B[Home]"
        }
        logging.info("Generated wiring flow")
        return json.dumps(flow, indent=2)
    except Exception as e:
        st.error(f"Prototype Wiring failed: {str(e)}")
        logging.error(f"Prototype Wiring failed: {str(e)}")
        return json.dumps({"status": "error", "message": f"Prototype Wiring failed: {str(e)}"}, indent=2)

def generate_documentation(input_data: str) -> str:
    """Generates dev-ready documentation."""
    try:
        input_json = json.loads(input_data)
        doc = {
            "specification": f"# Specification\n## Flow\n{input_json.get('flow', '')}",
            "accessibility_checklist": input_json.get("accessibility_plan", ""),
            "behavior_tree": input_json.get("mermaid", ""),
            "ui_blueprint": input_json.get("blueprint", "")
        }
        result = markdown.markdown(json.dumps(doc, indent=2), extensions=['fenced_code', 'tables'])
        logging.info("Generated documentation")
        return result
    except Exception as e:
        st.error(f"Documentation Generation failed: {str(e)}")
        logging.error(f"Documentation Generation failed: {str(e)}")
        return json.dumps({"status": "error", "message": f"Documentation Generation failed: {str(e)}"}, indent=2)

# Define tools for each agent
tools = [
    Tool(name="MapInputToState", func=map_input_to_state, description="Maps user input to app state changes with RAG"),
    Tool(name="SuggestAccessibility", func=suggest_accessibility, description="Suggests WCAG 2.1-compliant accessibility strategies"),
    Tool(name="GenerateUIBlueprint", func=generate_ui_blueprint, description="Generates textual UI blueprints"),
    Tool(name="GenerateWiring", func=generate_wiring, description="Generates screen transition logic with Mermaid syntax"),
    Tool(name="GenerateDocumentation", func=generate_documentation, description="Generates Markdown documentation")
]

# Prompt template for ReAct agents
prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    template="""
You are an AI agent simulating app behavior. Use these tools:

{tools}

Tool names: {tool_names}

Steps:
1. Analyze: {input}
2. Choose tool.
3. Respond:
   Thought: [Reasoning]
   Action: [Tool name]
   Action Input: {{[JSON input]}}

Example:
Thought: Map input to state.
Action: MapInputToState
Action Input: {{"user_input": "Tap Start", "app_purpose": "Meditation App"}}

Use JSON/Markdown outputs. Return minimal JSON if failed.

Scratchpad: {agent_scratchpad}

If no action:
Thought: No action needed.
Action: None
Action Input: {{}}
"""
)

# Agent initializer
def agent_initializer():
    """Initializes and returns all five agents."""
    agent1 = create_agent([tools[0]], "InputResponseMappingAgent")
    agent2 = create_agent([tools[1]], "AccessibleInteractionDesignerAgent")
    agent3 = create_agent([tools[2]], "SemanticUIBlueprintAgent")
    agent4 = create_agent([tools[3]], "PrototypeWiringAgent")
    agent5 = create_agent([tools[4]], "PrototypeDocumentationAgent")
    return agent1, agent2, agent3, agent4, agent5

def create_agent(tools_subset, agent_name):
    """Creates a ReAct agent."""
    agent = create_react_agent(llm=llm, tools=tools_subset, prompt=prompt_template)
    return AgentExecutor(
        agent=agent,
        tools=tools_subset,
        verbose=True,
        name=agent_name,
        handle_parsing_errors=True,
        max_iterations=75,
        max_execution_time=180
    )

# Streamlit app
def main():
    st.title("Autonomous Agentic AI Workflow")
    st.write("Simulate app behavior, interaction flows, and accessibility logic using MiniLM and Gemini.")

    # Input form
    with st.form("project_form"):
        app_purpose = st.text_input("App Purpose", value="Meditation App for Kids")
        user_scenario = st.text_input("User Scenario", value="Schedule a meditation session")
        target_audience = st.text_input("Target Audience", value="Children aged 8-12")
        user_input = st.text_input("User Input", value="Tap 'Start Session'")
        device_type = st.selectbox("Device Type", ["Mobile", "Web", "Both"], index=0)
        input_modes = st.multiselect("Input Modes", ["Touch", "Voice", "Keyboard"], default=["Touch"])
        submitted = st.form_submit_button("Run Workflow")

    if submitted:
        # Validate inputs
        if not all([app_purpose, user_scenario, target_audience, user_input]):
            st.error("All input fields are required.")
            return

        # Initialize agents
        with st.spinner("Initializing agents..."):
            try:
                agent1, agent2, agent3, agent4, agent5 = agent_initializer()
                st.success("Agents initialized successfully.")
            except Exception as e:
                st.error(f"Agent initialization failed: {str(e)}")
                logging.error(f"Agent initialization failed: {str(e)}")
                return

        # Prepare input data
        input_data: Dict[str, Any] = {
            "app_purpose": app_purpose.strip(),
            "user_scenario": user_scenario.strip(),
            "user_activity": user_scenario.strip(),
            "target_audience": target_audience.strip(),
            "user_input": user_input.strip(),
            "user_journey": user_scenario.strip(),
            "device_type": device_type.strip(),
            "input_modes": ", ".join(input_modes),
            "screen": "Home"
        }

        # Execute agents with progress bar
        st.subheader("Agent Outputs")
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Helper function to process agent
        def process_agent(agent, agent_name, input_data, progress):
            status_text.text(f"Running {agent_name}...")
            try:
                result = agent.invoke({"input": json.dumps(input_data)})
                if "output" in result and result["output"]:
                    try:
                        output_json = json.loads(result["output"])
                        st.json(output_json)
                        if output_json.get("status") == "error":
                            st.error(f"{agent_name} Error: {output_json.get('message')}")
                            if "fallback" in output_json:
                                return output_json["fallback"]
                            return {"status": "fallback", "message": "Using default output", "data": {"input": input_data.get("user_input", "unknown"), "state": "default_state"}}
                        return output_json
                    except json.JSONDecodeError as e:
                        st.error(f"JSON Parse Error in {agent_name}: {str(e)} - Raw Output: {result['output']}")
                        logging.error(f"JSON Parse Error in {agent_name}: {str(e)}")
                        if "Agent stopped due to iteration limit or time limit" in str(result["output"]):
                            return {"status": "fallback", "message": "Iteration/time limit exceeded", "data": {"input": input_data.get("user_input", "unknown"), "state": "default_state"}}
                        return {"status": "fallback", "message": f"Parse error: {str(e)}", "data": {"input": input_data.get("user_input", "unknown"), "state": "default_state"}}
                else:
                    st.error(f"No valid output from {agent_name} - Raw Response: {result}")
                    logging.error(f"No valid output from {agent_name}")
                    return {"status": "fallback", "message": "No output", "data": {"input": input_data.get("user_input", "unknown"), "state": "default_state"}}
            except ResourceExhausted as e:
                st.error(f"Quota exceeded for {agent_name}: {str(e)}. Check quota at https://ai.google.dev/gemini-api/docs/rate-limits (Limit: 15 requests/min, retry after 9s).")
                logging.error(f"Quota exceeded for {agent_name}: {str(e)}")
                return {"status": "fallback", "message": f"Quota exceeded: {str(e)}", "data": {"input": input_data.get("user_input", "unknown"), "state": "default_state"}}
            except Exception as e:
                st.error(f"{agent_name} failed: {str(e)}")
                logging.error(f"{agent_name} failed: {str(e)}")
                return {"status": "fallback", "message": f"Unexpected error: {str(e)}", "data": {"input": input_data.get("user_input", "unknown"), "state": "default_state"}}
            finally:
                progress_bar.progress(progress)

        # Agent 1: Input-Response Mapping
        st.write("### Input-Response Mapping Agent")
        result1 = process_agent(agent1, "Input-Response Mapping Agent", input_data, 0.2)
        if result1:
            if result1.get("status") == "fallback":
                st.warning(f"{result1['message']}")
                input_data["state_changes"] = result1["data"]
            else:
                input_data["state_changes"] = result1
        else:
            st.warning("Using default state due to failure.")
            input_data["state_changes"] = {"input": input_data.get("user_input", "unknown"), "state": "default_state"}

        # Agent 2: Accessible Interaction Designer
        st.write("### Accessible Interaction Designer Agent")
        result2 = process_agent(agent2, "AccessibleInteractionDesignerAgent", input_data, 0.2)
        if result2 and "accessibility_plan" in result2:
            if result2.get("status") == "fallback":
                st.warning(f"{result2['message']}")
                input_data["accessibility_plan"] = result2["data"]
            else:
                input_data["accessibility_plan"] = result2["accessibility_plan"]
        else:
            st.warning("Using default accessibility plan due to failure.")
            input_data["accessibility_plan"] = {"keyboard": "Tab navigation", "touch": "48x48px targets"}

        # Agent 3: Semantic UI Behavior Generator
        st.write("### Semantic UI Behavior Generator Agent")
        result3 = process_agent(agent3, "SemanticUIBlueprintAgent", input_data, 0.6)
        if result3:
            if result3.get("status") == "fallback":
                st.warning(f"{result3['message']}")
                input_data["blueprint"] = result3["data"]
            else:
                input_data["blueprint"] = result3
        else:
            st.warning("Using default UI blueprint due to failure.")
            input_data["blueprint"] = {"screen": "Home", "components": [{"type": "button", "text": "Default"}]}

        # Agent 4: Prototype Wiring
        st.write("### Prototype Wiring Agent")
        result4 = process_agent(agent4, "PrototypeWiringAgent", input_data, 0.8)
        if result4:
            if result4.get("status") == "fallback":
                st.warning(f"{result4['message']}")
                input_data["flow"] = result4["data"].get("flow", "Default -> Home")
                input_data["mermaid"] = result4["data"].get("mermaid", "graph TD A[Default] --> B[Home]")
            else:
                input_data["flow"] = result4["flow"]
                input_data["mermaid"] = result4["mermaid"]
        else:
            st.warning("Using default wiring due to failure.")
            input_data["flow"] = "Default -> Home"
            input_data["mermaid"] = "graph TD A[Default] --> B[Home]"

        # Agent 5: Prototype Documentation
        st.write("### Prototype Documentation Agent")
        status_text.text("Generating documentation...")
        try:
            result5 = agent5.invoke({"input": json.dumps(input_data)})
            if "output" in result5 and result5["output"]:
                try:
                    st.markdown(result5["output"])
                    with open("prototype_doc.md", "w") as f:
                        f.write(result5["output"])
                    st.success("Documentation saved as prototype_doc.md")
                    logging.info("Documentation saved successfully")
                except Exception as e:
                    st.error(f"Markdown Render Error: {str(e)} - Raw Output: {result5['output']}")
                    logging.error(f"Markdown Render Error: {str(e)}")
            else:
                st.error(f"No valid output from Prototype Documentation Agent - Raw Response: {result5}")
                logging.error("No valid output from Prototype Documentation Agent")
                st.markdown("# Prototype Specification\n## Default Documentation")
        except ResourceExhausted as e:
            st.error(f"Quota exceeded for Prototype Documentation Agent: {str(e)}. Check your API quota.")
            logging.error(f"Quota exceeded for Prototype Documentation Agent: {str(e)}")
            st.markdown("# Prototype Specification\n## Default Documentation")
        except Exception as e:
            st.error(f"Prototype Documentation Agent failed: {str(e)}")
            logging.error(f"Prototype Documentation Agent failed: {str(e)}")
            st.markdown("# Prototype Specification\n## Default Documentation")
        finally:
            progress_bar.progress(1.0)
            status_text.text("Workflow completed.")

if __name__ == "__main__":
    main()