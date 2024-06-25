from typing import Union, Tuple

from src.config import Configuration
from src.object import Character, Object, Fact, Interaction
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
        facts: list[Fact],
        interactions: list[Interaction],
        objects: dict[str, Object],
        characters: dict[str, Character],
    ):
        super().__init__(name, summary, description, facts, interactions, parent=None)
        self.config = config
        self.memory = memory
        self.objects = objects
        self.characters = characters
        self.all: dict[str, Object] = {}
        self.all.update(self.objects)
        self.all.update(self.characters)
        self.all["scene"] = self
        self.explored: set[str] = set()
        self.explored.add("scene")
        self.explored.add("player")

        # action independent llm
        self.other_llm = QuestionLLM(self.memory, self.config, "other")
        self.other_llm.set_prompt(self.summary_prompt())

        self.question_llm = QuestionLLM(self.memory, self.config, "question")
        self.question_llm.set_prompt(self.summary_prompt())

        # action dependant llm
        self.describer_llm = DescriberLLM(self.memory, self.config)
        self.interacter_llm = InteracterLLM(self.memory, self.config)
        self.talker_llm = TalkerLLM(self.memory, self.config)

    def explore(self, revealed_objects: list[str]):
        self.explored.update(revealed_objects)

    def other_prompt(self) -> str:
        player = self.characters["player"]
        result = (
            "The game is made of multiple scenes which contain objects and character which the player can look "
            "at or interact with\n"
        )
        result += (
            "The player should move around in the scenes and explore everything. Occasionally, he can also pick "
            "up items.\n"
        )
        result += f"The player is {player.description}\n"
        result += f"The current scene is {self.summary_prompt()}"
        return result

    def question_prompt(self) -> str:
        player = self.characters["player"]
        result = "This is a in-depth description of the current scene:\n"
        result += f"{self.describer_prompt()}\n"
        result += "Here is an in-depth description of the player:\n"
        result += f"{player.describer_prompt()}\n"
        result += (
            "This is a lot of information most of might not even be known to the player, so be careful with "
            "what you say. In some cases it might be best to remain as vague as possible.\n"
        )
        return result

    def actor_prompt(self):
        result = "This is a list of all game-objects which are inside the scene:\n"
        for obj_name in self.explored:
            obj = self.all[obj_name]
            result += f"{obj.summary_prompt()}\n"
        return result[:-1]

    def evaluate(
        self, action: str, explanation: str, user_input: str, debug_msg=""
    ) -> Tuple[str, Union[str, None]]:
        if "other" in action:
            self.other_llm.set_prompt(self.other_prompt())
            return self.other_llm.query(user_input, explanation), None
        elif "question" in action:
            self.question_llm.set_prompt(self.question_prompt())
            return self.question_llm.query(user_input, explanation), None
        elif "failure" in action:
            if len(debug_msg) > 0:
                return f"Sorry, I did not understand you due to {debug_msg}...", None
            else:
                return "Sorry, I did not understand you", None
        else:
            command = action.split("_")
            if len(command) < 2:
                return self.evaluate(
                    "failure",
                    explanation,
                    user_input,
                    debug_msg='command did not contain "_"',
                )
            else:
                command_name = command.pop(0)
                obj_name = "_".join(command)
                if obj_name not in self.all.keys():
                    # todo debug msg
                    print(
                        f"{obj_name} can not be {command_name} since it does not exist."
                    )
                    return self.evaluate(
                        f"{command_name}_scene",
                        explanation,
                        user_input,
                    )
                elif obj_name not in self.explored:
                    return self.evaluate(
                        "failure",
                        explanation,
                        user_input,
                    )

                obj = self.all[obj_name]
                if "describe" in command_name:
                    self.describer_llm.set_prompt(obj.describer_prompt())
                    response = self.describer_llm.query(user_input, explanation)

                    indices = response["indices"]
                    if isinstance(indices, list):
                        for i, x in enumerate(indices):
                            indices[i] = int(x)
                    answer = response["answer"]
                    if "description" in response.keys():
                        description = response["description"]
                        answer += f"\n{description}"
                        explored = obj.reveal_fact(description, indices)
                        self.explore(explored)
                    elif len(indices) > 0:
                        print("Description has not been updated, but facts have???")
                    return answer, None
                elif "interact" in command_name:
                    self.interacter_llm.set_prompt(obj.interacter_prompt())
                    response: dict = self.interacter_llm.query(user_input, explanation)

                    indices = response["indices"]
                    if isinstance(indices, list):
                        for i, x in enumerate(indices):
                            indices[i] = int(x)
                    description = None
                    if "description" in response.keys():
                        description = response["description"]
                    explored = obj.trigger_interaction(description, indices)
                    self.explore(explored)
                    effect = response["effect"]
                    return effect, None
