class InsightReporter:
    def __init__(self):
        self.user_log = []

    def add_response(self, response: str):
        self.user_log.append(response)

    def generate_summary(self) -> str:
        if not self.user_log:
            return "No user responses recorded."
        total_words = sum(len(r.split()) for r in self.user_log)
        talk_time_ratio = total_words / (total_words + 100)  # Assume 100 from AI agents
        missed = "Could explore counterarguments more." if talk_time_ratio < 0.3 else "Engaged well."

        return (
            f"📊 Insight Summary:\n"
            f"- Estimated Talk Time Ratio: {talk_time_ratio:.2f}\n"
            f"- Feedback: {missed}\n"
        )