from abc import ABC, abstractmethod
from typing import Union

from src.config import Configuration
from src.llm import QuestionLLM, StoryLLM, DialogueLLM, ActionLLM
from src.memory import VectorStore


class Describer(ABC):
    def __init__(self, description: str):
        self._description = description

    @abstractmethod
    def describe(self) -> str:
        pass


class Character(Describer):
    def __init__(self, name: str, description: str):
        super().__init__(description)
        self.name = name

    def describe(self) -> str:
        return f"The character is called {self.name}. {self._description}"


class Object(Describer):
    def __init__(self, name: str, description: str):
        super().__init__(description)
        self.name = name

    def describe(self) -> str:
        return f"The {self.name} can be described like this: {self._description}"


class Item(Describer):
    def __init__(self, name: str, description: str):
        super().__init__(description)
        self.name = name

    def describe(self) -> str:
        return f"The Item \"{self.name}\" can be described like this: {self._description}"


class Action(Describer, ABC):
    def describe(self):
        return f'"{self.name}": {self._description}'

    @abstractmethod
    def output(
        self, user_input: Union[str, None]
    ) -> (Union[str, dict], Union[str, None]):
        pass

    def __init__(self, name: str, description: str):
        super().__init__(description)
        self.name = name


class ActionQuestion(Action):
    def output(
        self, user_input: Union[str, None]
    ) -> (Union[str, dict], Union[str, None]):
        assert user_input
        return self.llm.query(user_input), None

    def __init__(self, memory: VectorStore, config: Configuration, scene_descr: str):
        name = "question"
        # TODO; formulate this better
        description = "The player asks serious questions to investigate his environment, to seek advice etc..."
        super().__init__(name, description)
        self.llm = QuestionLLM(memory, config, scene_descr)


class ActionTroll(Action):
    def output(
        self, user_input: Union[str, None]
    ) -> (Union[str, dict], Union[str, None]):
        assert user_input
        return self.llm.query(user_input), None

    def __init__(self, memory: VectorStore, config: Configuration, scene_descr: str):
        name = "troll"
        description = "The player simply wants to troll the Game Master, break the 4th wall or mess around."
        llm_descr = (
            description
            + " Therefore, you should not answer seriously but rather give a witty response!"
        )
        llm_descr += f" To help you with that I provide you with a short description of the current scene: {scene_descr}"
        super().__init__(name, description)
        self.llm = QuestionLLM(memory, config, llm_descr)


class ActionFailure(Action):
    def output(
        self, user_input: Union[str, None]
    ) -> (Union[str, dict], Union[str, None]):
        response = "Sorry, I do not understand what you want from me..."
        return response, None

    def __init__(self):
        name = "failure"
        description = "The user input is not clear enough."
        super().__init__(name, description)


class ActionStory(Action):
    def output(
        self, user_input: Union[str, None]
    ) -> (Union[str, dict], Union[str, None]):
        return self.llm.query(), self.target_state

    def __init__(
        self,
        memory: VectorStore,
        target_state: str,
        config: Configuration,
        name: str,
        description: str,
        effect: str,
        scene_descr: str,
    ):
        super().__init__(name, description)
        effect = f"This was the player's action: {effect}\n"
        effect += "Describe the action's effect"
        self.llm = StoryLLM(memory, config, scene_descr, effect)
        self.target_state = target_state


class ActionSimpleStory(Action):
    def output(
        self, user_input: Union[str, None]
    ) -> (Union[str, dict], Union[str, None]):
        return self.target_description, self.target_state

    def __init__(
        self, target_state: str, target_descr: str, name: str, description: str
    ):
        super().__init__(name, description)
        self.target_state = target_state
        self.target_description = target_descr


class ActionSelf(Action):
    def output(
        self, user_input: Union[str, None]
    ) -> (Union[str, dict], Union[str, None]):
        assert user_input
        return self.llm.query(user_input), None

    def __init__(self, memory: VectorStore, config: Configuration, self_descr: str):
        name = "self"
        description = "The player wants to know some more about him-/herself."
        super().__init__(name, description)
        self.llm = QuestionLLM(memory, config, self_descr)


class ActionTalk(Action):
    def output(
        self, user_input: Union[str, None]
    ) -> (Union[str, dict], Union[str, None]):
        assert user_input
        return self.llm.query(user_input), None

    def __init__(
        self,
        memory: VectorStore,
        config: Configuration,
        description: str,
        character_name: str,
        character_descr: str,
        scene_descr: str,
    ):
        name = f"talk_{character_name}"
        super().__init__(name, description)
        self.llm = DialogueLLM(memory, config, scene_descr, character_descr)


class Scene(Describer):
    def describe(self) -> str:
        return self._description

    def _describe_actions(self) -> str:
        description = ""
        for action in self.actions:
            description += action.describe() + "\n"
        return description

    def __init__(
        self, memory: VectorStore, config: Configuration, name: str, description: str, characters: dict
    ):
        super().__init__(description)
        self.name = name
        self.config = config
        self.memory = memory
        self.characters = characters
        # Basic actions which are available in each scene
        self.actions: list = [
            ActionQuestion(memory, config, description),
            ActionSelf(memory, config, characters["self"].describe()),
            ActionTroll(memory, config, description),
            ActionFailure(),
        ]
        self.llm = ActionLLM(self.memory, self.config, description, self._describe_actions())

    def add_action_story(
        self, target_state: str, name: str, description: str, effect: str
    ):
        self.actions.append(
            ActionStory(
                self.memory, target_state, self.config, name, description, effect, self.describe()
            )
        )
        self.llm = ActionLLM(self.memory, self.config, description, self._describe_actions())

    def add_simple_action_story(
        self, target_state: str, target_descr: str, name: str, description: str
    ):
        self.actions.append(
            ActionSimpleStory(target_state, target_descr, name, description)
        )
        self.llm = ActionLLM(self.memory, self.config, description, self._describe_actions())

    def add_action_talk(self, character_name, character_descr: str):
        description = f"The player wants to talk to or approach {character_name}"
        self.actions.append(
            ActionTalk(
                self.memory,
                self.config,
                description,
                character_name,
                character_descr,
                self.describe(),
            )
        )
        self.llm = ActionLLM(self.memory, self.config, description, self._describe_actions())

    def get_action(self, name: str) -> Union[Action, None]:
        for a in self.actions:
            if a.name == name:
                return a
        return None
