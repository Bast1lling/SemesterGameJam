import os
import json

from src.config import Configuration
from src.llm import ActorLLM
from src.memory import VectorStore
from src.object import parse_object, Rule
from src.scene import Scene


def debug_msg(s: str, config: Configuration):
    if config.debug:
        print(s)


class Game:
    def __init__(self):
        self.config = Configuration()
        self.memory = VectorStore()
        self.finished = False
        self.characters = {}
        self.scenes = {}

        # load character files
        path_to_characters = os.path.join(self.config.path_to_story, "characters")
        files = os.listdir(path_to_characters)
        for file in files:
            if file.endswith(".json"):
                self.parse_character_json(os.path.join(path_to_characters, file))

        # load scene files
        path_to_scenes = os.path.join(self.config.path_to_story, "scenes_v2")
        files = os.listdir(path_to_scenes)
        files = sorted(files)
        for file in files:
            if file.endswith(".json"):
                self.parse_scene_json(os.path.join(path_to_scenes, file))

        # set first to current scene
        self.current_scene: Scene = list(self.scenes.values())[0]

        # init llm determining the current action
        self.llm = ActorLLM(self.memory, self.config)

    def next(self, user_input: str):
        if self.finished:
            return "restart"
        self.llm.set_prompt(self.current_scene.actor_prompt())
        action_name = self.llm.query(user_input)
        llm_output, scene = self.current_scene.evaluate(action_name, user_input)
        if scene:
            if scene == "end":
                self.finished = True
            self.current_scene = self.scenes[scene]
        self.memory.insert(user_input, llm_output)
        return llm_output

    def parse_scene_json(self, path_to_scene_file: str):
        scene_data = Game.load_json(path_to_scene_file)
        scene_name = scene_data["name"]
        summary = scene_data["summary"]
        description = scene_data["description"]
        if "description_rules" in scene_data.keys():
            description_rules = scene_data["description_rules"]
        else:
            description_rules = ["DO NOT CHANGE ANYTHING OF THIS DESCRIPTION!"]
        description_rules = [Rule(s) for s in description_rules]

        if "interaction_rules" in scene_data.keys():
            interaction_rules = scene_data["interaction_rules"]
        else:
            interaction_rules = [
                f"The player can not interact with {scene_name}. Tell him that it is senseless!"
            ]
        interaction_rules = [Rule(s) for s in interaction_rules]

        explored = set()
        if "explored" in scene_data.keys():
            explored = set(scene_data["explored"])

        children = []
        character_names: list = scene_data["characters"]
        character_names.append("player")
        for name in character_names:
            children.append(self.characters[name])

        object_data: dict = scene_data["children"]
        for obj_name, obj_data in object_data.items():
            children.append(parse_object(obj_name, obj_data))

        scene = Scene(
            self.memory,
            self.config,
            scene_name,
            summary,
            description,
            description_rules,
            interaction_rules,
            children,
            explored,
        )
        self.scenes[scene_name] = scene

    def parse_character_json(self, path_to_character_file: str):
        character_data = Game.load_json(path_to_character_file)
        name = character_data["name"]
        self.characters[name] = parse_object(name, character_data)

    @staticmethod
    def load_json(path: str):
        with open(path, "r") as file:
            return json.load(file)
