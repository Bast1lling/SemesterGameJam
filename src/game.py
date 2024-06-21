import os
import json

from src.config import Configuration
from src.llm import ActorLLM
from src.memory import VectorStore
from src.scene import Scene, Character, Object


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
        path_to_scenes = os.path.join(self.config.path_to_story, "scenes")
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
        self.llm.set_prompt(
            self.current_scene.description, self.current_scene.things.keys()
        )
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
        name = scene_data["name"]
        description = scene_data["description"]
        layout = scene_data["layout"]
        object_data: dict = scene_data["objects"]
        character_names: list = scene_data["characters"]
        characters = {
            k: self.characters[k] for k in character_names if k in self.characters
        }
        characters["player"] = self.characters["player"]
        objects = {}
        for key, value in object_data.items():
            if "target_scene" in value.keys():
                target_scene = value["target_scene"]
            else:
                target_scene = None
            object_description = value["description"]
            effect = value["effect"]
            objects[key] = Object(key, effect, target_scene, object_description)

        scene = Scene(
            self.memory, self.config, name, description, layout, characters, objects
        )
        self.scenes[name] = scene

    def parse_character_json(self, path_to_character_file: str):
        character_data = Game.load_json(path_to_character_file)
        key = os.path.basename(path_to_character_file)[:-5]
        name = character_data["name"]
        description = character_data["description"]
        self.characters[key] = Character(name, description)

    @staticmethod
    def load_json(path: str):
        with open(path, "r") as file:
            return json.load(file)
