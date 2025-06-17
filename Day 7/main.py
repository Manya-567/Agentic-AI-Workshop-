import streamlit as st
from agents.discussion_agent import DiscussionSimulator
from agents.participation_agent import ParticipationAnalyzer
from agents.reporting_agent import InsightReporter
from agents.recommendation_agent import RecommendationAgent

st.title("🧭 AI-Led Group Discussion Simulator")

discussion = DiscussionSimulator()
analysis = ParticipationAnalyzer()
reporting = InsightReporter()
recommender = RecommendationAgent()

st.subheader("💬 Enter a discussion topic:")
topic = st.text_input("Topic", "Is remote work better than office work?")

if st.button("Start Discussion"):
    discussion_log = discussion.simulate(topic)
    for statement in discussion_log:
        st.write(statement)

    user_input = st.text_area("🗣️ Your Response:")
    if st.button("Submit Response"):
        analysis.log_user_response(user_input)
        reporting.add_response(user_input)
        st.success("✅ Response submitted!")

if st.button("📈 Analyze Participation"):
    result = analysis.analyze()
    st.markdown(result)

if st.button("📊 Generate Insight Report"):
    summary = reporting.generate_summary()
    st.markdown(summary)

if st.button("🎯 Get Improvement Tips"):
    feedback = recommender.recommend("How can I improve my participation in a group discussion?")
    st.markdown(f"**Tips:** {feedback}")
