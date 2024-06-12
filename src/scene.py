import os
from abc import ABC, abstractmethod

from src.config import Configuration
from src.llm import LLM, LLMFunction
from src.prompter import Prompt, SystemPrompt


class Describer(ABC):
    def __init__(self, description: str, answer: str):
        self._description = description
        self._answer = answer

    @abstractmethod
    def describe_to_llm(self) -> str:
        pass

    @abstractmethod
    def describe_to_player(self) -> str:
        pass


class Action(Describer):
    def describe_to_llm(self):
        return f"\"{self.name}\": {self._description}"

    def describe_to_player(self):
        return self._answer

    def __init__(self, name: str, description: str, answer: str):
        super().__init__(description, answer)
        self.name = name

    def set_answer(self, answer: str):
        self._answer = answer

    def get_answer(self) -> str:
        return self._answer


class Scene(Describer):
    def describe_to_player(self) -> str:
        return self._answer

    def describe_to_llm(self) -> str:
        return self._description

    def __init__(self, name: str, description: str, answer: str, actions: list):
        super().__init__(description, answer)
        self.name = name
        self.actions = actions
        self.actions.append(Action("scene", "The player investigates his surroundings by asking questions about the "
                                            "scene.", None))
        self.actions.append(Action("troll", "The player wants to simply troll the Game Master, break the 4th wall or "
                                            "mess around. Try to respond with the right amount of wit!", None))
        self.actions.append(Action("advice", "The player seeks for advice. If you can not answer his questions, "
                                             "tell him that you do not know everything and he should continue asking "
                                             "questions.", None))
        self.actions.append(Action("failure", "The user input is not clear enough.",
                                   "What are you talking about...? You need to ask questions and stop trolling me :("))

    def get_action(self, name: str):
        for a in self.actions:
            if a.name == name:
                return a
        return None
