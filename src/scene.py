from typing import Union, Tuple

from src.config import Configuration
from src.content import Character, Object
from src.llm import (
    DescriberLLM,
    ExplorerLLM,
    InteracterLLM,
    TalkerLLM,
    ThinkerLLM,
)
from src.memory import VectorStore


class Scene:
    def __init__(
        self,
        memory: VectorStore,
        config: Configuration,
        name: str,
        description: str,
        layout: str,
        characters: dict[str, Character],
        objects: dict[str, Object],
    ):
        self.description = description
        self.layout = layout
        self.name = name
        self.config = config
        self.memory = memory
        self.characters = characters
        self.objects = objects
        self.things = {}
        for name, obj in self.objects.items():
            self.things[name] = obj.describe()
        for name, character in self.characters.items():
            self.things[name] = character.describe()

        # things which existence is known to player:
        self.explored = set()
        # things which have already been described to the player:
        self.described = set()
        # objects the player already interacted with:
        self.interacted = set()
        # character the player already talked to:
        self.talked = set()

        # action independent llm
        self.thinker_llm = ThinkerLLM(self.memory, self.config)
        self.thinker_llm.set_prompt(
            self.description, self.layout, self.characters["player"].describe()
        )
        self.explorer_llm = ExplorerLLM(self.memory, self.config)
        self.explorer_llm.set_prompt(
            self.description, self.layout, list(self.things.keys())
        )

        # action dependant llm
        self.describer_llm = DescriberLLM(self.memory, self.config)
        self.interacter_llm = InteracterLLM(self.memory, self.config)
        self.talker_llm = TalkerLLM(self.memory, self.config)

    def evaluate(self, action: str, user_input: str) -> Tuple[str, Union[str, None]]:
        if "explore" in action:
            response = self.explorer_llm.query(user_input)
            self.explored.update(set(response["objects"]))
            return response["description"], None
        elif "other" in action:
            return self.thinker_llm.query(user_input), None
        elif "failure" in action:
            return "Sorry, I did not understand you...", None
        elif "describe" in action:
            thing = action.split("_")[1]
            self.describer_llm.set_prompt(self.description, self.things[thing])
            return self.describer_llm.query(user_input), None
        elif "interact" in action:
            name = action.split("_")[1]
            if name in self.objects.keys():
                obj = self.objects[name]
                self.interacter_llm.set_prompt(obj.describe(), obj.effect)
                return self.interacter_llm.query(user_input), obj.target_scene
            elif name in self.characters.keys():
                self.talker_llm.set_prompt(
                    self.description, self.characters[name].describe()
                )
                response = self.talker_llm.query(user_input)
                # todo move this out of the scene?
                while response["continue"]:
                    print(response["response"])
                    user_input = input()
                    response = self.talker_llm.query(user_input)
                return response["response"], None
            else:
                return f"You can not interact with {name}!", None
        else:
            return f"Unknown action: {action}", None
