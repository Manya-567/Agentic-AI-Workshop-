import streamlit as st
from pydantic import BaseModel, Field
from typing import Type
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

# ---------- LLM Setup ----------
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key="AIzaSyDpLFPHv3b6PzwbUrxXG1BuHzZlCkQVGhw"  # Consider using st.secrets in production
)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Agentic Prototype Simulator", layout="wide")
st.title("🤖 Autonomous Agentic UI Behavior Simulator")

app_purpose = st.text_input("📱 App Purpose (e.g., Grocery Delivery)")
user_scenario = st.text_area("👤 User Scenario (e.g., user adds items to cart)", height=150)
target_audience = st.text_input("🎯 Target Audience (e.g., visually impaired users)")

run_button = st.button("🚀 Simulate Agent Workflow")

# ---------- Define Agents ----------
input_mapper = Agent(
    role="Input-Response Mapping Agent",
    goal="Map user inputs to corresponding app state transitions and behaviors.",
    backstory="Expert in mapping user gestures and voice commands to structured app logic.",
    llm=llm
)

accessibility_agent = Agent(
    role="Accessible Interaction Designer Agent",
    goal="Design inclusive interaction alternatives using keyboard, voice, and color contrast.",
    backstory="Advocate for accessibility-first UI design, specialized in inclusive workflows.",
    llm=llm
)

ui_behavior_agent = Agent(
    role="Semantic UI Behavior Generator Agent",
    goal="Define screens and UI components in structured semantic format.",
    backstory="Proficient in translating flow into semantic component-based blueprints.",
    llm=llm
)

wiring_agent = Agent(
    role="Prototype Wiring Agent",
    goal="Map out all screen transitions and conditional flows.",
    backstory="Interaction logic engineer focused on defining screen-to-screen wiring schemas.",
    llm=llm
)

doc_agent = Agent(
    role="Documentation & Handoff Agent",
    goal="Compile all agent outputs into a single developer handoff-ready document.",
    backstory="Experienced UX writer and developer handoff coordinator.",
    llm=llm
)

manager = Agent(
    role="System Architect",
    goal="Coordinate all agents and validate logical consistency across tasks.",
    backstory="Senior AI system integrator overseeing prototype simulation pipelines.",
    llm=llm,
    allow_delegation=True
)

# ---------- Run Button Handler ----------
if run_button:
    if not app_purpose or not user_scenario or not target_audience:
        st.warning("⚠️ Please fill in all fields before running the simulation.")
    else:
        with st.spinner("🧠 Agents are working on your prototype..."):
            try:
                # ---------- Define Tasks ----------
                task1 = Task(
                    description=f"Map possible user inputs (tap, swipe, voice) to actions and app states for:\nApp Purpose: {app_purpose}\nScenario: {user_scenario}",
                    expected_output="Input → Trigger → Action → App State JSON mapping",
                    agent=input_mapper,
                    verbose=True
                )

                task2 = Task(
                    description=f"Design accessible interaction strategies for target audience: {target_audience}. Include input alternatives, color guidelines, and alt-descriptions.",
                    expected_output="Accessibility plan covering inputs, visual design, and interaction types.",
                    agent=accessibility_agent,
                    verbose=True
                )

                task3 = Task(
                    description=f"Generate a screen-by-screen UI component breakdown for the given user journey.",
                    expected_output="Text-based blueprint with component names, behaviors, and interaction states.",
                    agent=ui_behavior_agent,
                    verbose=True
                )

                task4 = Task(
                    description="Define logical wiring between screens including conditions and fallback paths.",
                    expected_output="Textual flow schema + screen transition logic.",
                    agent=wiring_agent,
                    verbose=True
                )

                task5 = Task(
                    description="Compile final developer handoff document with input mappings, accessibility plan, UI blueprint, and flow logic.",
                    expected_output="Markdown-format prototype specification with clear headers.",
                    agent=doc_agent,
                    verbose=True
                )

                # ---------- Debug: Print Prompts (Optional) ----------
                print("Task 1 Prompt:", task1.description)
                print("Task 2 Prompt:", task2.description)
                print("Task 3 Prompt:", task3.description)
                print("Task 4 Prompt:", task4.description)
                print("Task 5 Prompt:", task5.description)

                # ---------- Crew Setup ----------
                crew = Crew(
                    agents=[input_mapper, accessibility_agent, ui_behavior_agent, wiring_agent, doc_agent],
                    tasks=[task1, task2, task3, task4, task5],
                    manager_agent=manager,
                    process=Process.hierarchical
                )

                crew.verbose = True
                result = crew.kickoff()

                st.success("✅ Simulation Complete!")
                st.subheader("📄 Prototype Behavior Output")
                st.text_area("📋 Agentic Output", result, height=500)

            except Exception as e:
                st.error("❌ Simulation failed.")
                st.exception(e)