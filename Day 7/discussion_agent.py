from typing import List

class SimulatedParticipant:
    def __init__(self, name: str, personality: str, strengths: List[str]):
        self.name = name
        self.personality = personality
        self.strengths = strengths

    def respond(self, topic: str) -> str:
        return f"[{self.name} - {self.personality}] On '{topic}', I think {self._generate_opinion(topic)}."

    def _generate_opinion(self, topic: str) -> str:
        # Simulated opinion generator based on strengths
        if "logic" in self.strengths:
            return f"logically, it's important to analyze both pros and cons."
        elif "creativity" in self.strengths:
            return f"it's a great topic to think outside the box."
        elif "data" in self.strengths:
            return f"data shows interesting trends around this."
        else:
            return f"everyone will have a unique take."

class DiscussionSimulator:
    def __init__(self):
        self.participants = [
            SimulatedParticipant("Alex", "analytical", ["logic"]),
            SimulatedParticipant("Taylor", "creative", ["creativity"]),
            SimulatedParticipant("Jordan", "practical", ["data"]),
        ]

    def simulate(self, topic: str) -> List[str]:
        print(f"\n🎤 Simulated Discussion on: {topic}")
        return [p.respond(topic) for p in self.participants]
