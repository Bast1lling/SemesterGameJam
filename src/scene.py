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
        self.actions.append(Action("failure", "The user input is not clear enough.",
                                   "What are you talking about...? You need to ask questions and stop trolling me :("))

    def get_action(self, name: str):
        for a in self.actions:
            if a.name == name:
                return a
        return None


class SceneLLM(LLM):
    def __init__(
            self,
            config: Configuration,
            scene: Scene,
    ):
        super().__init__(config.gpt_version, config.openai_api_key, config.save_traffic, None, None, None)

        self.scene = scene
        self.config = config
        self.llm_function = LLMFunction()
        self.llm_function.add_string_parameter("action", "chosen action")
        self.llm_function.add_string_parameter("description", "description of the action if required")

        script_dir = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(script_dir, "prompts/scene_system.txt"), "r") as file:
            self.system_prompt = SystemPrompt(file.read())

        self.user_prompt_structure = [
            "scene",
            "actions",
            "task",
            "history",
            "user",
        ]
        self.user_prompt = Prompt(self.user_prompt_structure)

        scene_content = "The player is currently here:\n"
        scene_content += self.scene.describe_to_llm()
        self.user_prompt.set("scene", scene_content)

        actions_content = "The player can take the following actions:\n"
        for action in self.scene.actions:
            actions_content += action.describe_to_llm() + "\n"
        self.user_prompt.set("actions", actions_content)

        if self.config.save_mode:
            history_content = "These were the previous interactions between you and the player:\n"
        else:
            history_content = "These were the previous messages from the player:\n"
        self.user_prompt.set("history", history_content)

        with open(os.path.join(script_dir, "prompts/scene.txt"), "r") as file:
            self.user_prompt.set("task", file.read())

    def query(self, user_input: str):
        user_content = "This was the input of the user:\n"
        user_content += user_input
        self.user_prompt.set("user", user_content)
        result = self._query()
        if "description" in result.keys():
            result_descr = result["description"]
            history = self.user_prompt.get("history")
            if len(history) < 1000:
                if self.config.save_mode:
                    history += f"Input: {user_input} Output: {result_descr}\n"
                else:
                    history += f"{user_input}\n"
            self.user_prompt.set("history", history)
        return result
