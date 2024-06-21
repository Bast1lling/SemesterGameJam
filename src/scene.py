from typing import Union, Tuple

from src.config import Configuration
from src.object import Character, Object, Rule
from src.llm import (
    DescriberLLM,
    InteracterLLM,
    TalkerLLM,
    QuestionLLM,
)
from src.memory import VectorStore


class Scene(Object):
    def __init__(
        self,
        memory: VectorStore,
        config: Configuration,
        name: str,
        summary: str,
        description: str,
        description_rules: list[Rule],
        interaction_rules: list[Rule],
        children: list,
        explored: set,
    ):
        super().__init__(
            name,
            summary,
            description,
            description_rules,
            interaction_rules,
            children=children,
            explored=explored,
        )
        self.config = config
        self.memory = memory
        self.children[self.name] = self
        self.explored.add(self.name)
        self.explored.add("player")
        self.objects: dict[str, Object] = {}
        self.characters: dict[str, Character] = {}
        for name, obj in self.children.items():
            if isinstance(obj, Character):
                self.characters[name] = obj
            else:
                self.objects[name] = obj

        # # things which existence is known to player:
        # self.explored = set()
        # # things which have already been described to the player:
        # self.described = set()
        # # objects the player already interacted with:
        # self.interacted = set()
        # # character the player already talked to:
        # self.talked = set()

        # action independent llm
        self.other_llm = QuestionLLM(self.memory, self.config, "other")
        self.other_llm.set_prompt(
            self.summary_prompt()
        )

        self.question_llm = QuestionLLM(self.memory, self.config, "question")
        self.question_llm.set_prompt(
            self.summary_prompt()
        )

        # action dependant llm
        self.describer_llm = DescriberLLM(self.memory, self.config)
        self.interacter_llm = InteracterLLM(self.memory, self.config)
        self.talker_llm = TalkerLLM(self.memory, self.config)

    def other_prompt(self) -> str:
        player = self.characters["player"]
        result = "The game is made of multiple scenes which contain objects and character which the player can look " \
                 "at or interact with\n"
        result += "The player should move around in the scenes and explore everything. Occasionally, he can also pick " \
                  "up items.\n"
        result += f"The player is {player.description}\n"
        result += f"The current scene is {self.summary_prompt()}"
        return result

    def question_prompt(self) -> str:
        player = self.characters["player"]
        result = "This is a in-depth description of the current scene:\n"
        result += f"{self.describer_prompt()}\n"
        result += "Here is an in-depth description of the player:\n"
        result += f"{player.describer_prompt()}\n"
        result += "This is a lot of information most of might not even be known to the player, so be careful with " \
                  "what you say. In some cases it might be best to remain as vague as possible.\n"
        return result

    def actor_prompt(self):
        result = "This is a list of all game-objects which are inside the scene:\n"
        for obj_name in self.explored:
            obj = self.children[obj_name]
            result += f"{obj.summary_prompt()}\n"
        return result[:-1]

    def evaluate(
        self, action: str, user_input: str, debug_msg=""
    ) -> Tuple[str, Union[str, None]]:
        if "other" in action:
            self.other_llm.set_prompt(self.other_prompt())
            return self.other_llm.query(user_input), None
        elif "question" in action:
            self.question_llm.set_prompt(self.question_prompt())
            return self.question_llm.query(user_input), None
        elif "failure" in action:
            return f"Sorry, I did not understand you due to {debug_msg}...", None
        else:
            command = action.split("_")
            if len(command) < 2:
                return self.evaluate(
                    "failure", user_input, debug_msg='command did not contain "_"'
                )
            else:
                command_name = command.pop(0)
                obj_name = "_".join(command)
                if obj_name not in self.children.keys():
                    return self.evaluate(
                        "failure",
                        user_input,
                        debug_msg=f"{obj_name} not part of current scene",
                    )

                obj = self.children[obj_name]
                if "describe" in command_name:
                    self.describer_llm.set_prompt(obj.describer_prompt())
                    response = self.describer_llm.query(user_input)
                    children = response["children"]
                    description = response["description"]
                    obj.description = description
                    obj.add_explored(children)
                    return description, None
                elif "interact" in command_name:
                    if obj_name in self.objects.keys():
                        self.interacter_llm.set_prompt(obj.interacter_prompt())
                        response = self.interacter_llm.query(user_input)
                        action = response["action"]
                        description = response["description"]
                        obj.description = description
                        return action, obj.target
                    elif obj_name in self.characters.keys():
                        character = self.characters[obj_name]
                        self.talker_llm.set_prompt(character.talker_prompt())
                        response = self.talker_llm.query(user_input)
                        # todo move this out of the scene?
                        while response["continue"]:
                            print(response["response"])
                            user_input = input()
                            response = self.talker_llm.query(user_input)
                        return response["response"], None
                    else:
                        return self.evaluate(
                            "failure",
                            user_input,
                            debug_msg=f"unknown command {command_name}",
                        )
