class ParticipationAnalyzer:
    def __init__(self):
        self.user_log = []

    def log_user_response(self, response: str):
        self.user_log.append(response)

    def analyze(self) -> str:
        total_responses = len(self.user_log)
        avg_length = sum(len(r.split()) for r in self.user_log) / total_responses if total_responses else 0
        clarity = "clear" if avg_length > 5 else "needs more depth"

        return (
            f"🧠 User Participation Analysis:\n"
            f"- Responses: {total_responses}\n"
            f"- Avg Length: {avg_length:.2f} words\n"
            f"- Clarity: {clarity}\n"
        )