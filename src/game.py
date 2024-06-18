import os
import json

from src.config import Configuration
from src.memory import VectorStore
from src.scene import Scene, Character


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
        jobs = []
        for file in files:
            if file.endswith(".json"):
                jobs.extend(self.parse_scene_json(os.path.join(path_to_scenes, file)))
        for scene, target_scene, name, descr in jobs:
            target_descr = self.scenes[target_scene].describe()
            scene.add_simple_action_story(target_scene, target_descr, name, descr)

        # set first to current scene
        self.current_scene = list(self.scenes.values())[0]

    def next(self, user_input: str):
        if self.finished:
            return "restart"
        action_name = self.current_scene.llm.query(user_input)
        action = self.current_scene.get_action(action_name)
        llm_output, scene = action.output(user_input)
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
        action_data: dict = scene_data["actions"]
        character_names: list = scene_data["characters"]
        characters = {
            k: self.characters[k] for k in character_names if k in self.characters
        }
        characters["self"] = self.characters["self"]
        scene = Scene(self.memory, self.config, name, description, characters)
        jobs = []
        for key, value in action_data.items():
            if "target_scene" in value.keys():
                target_state = value["target_scene"]
            else:
                target_state = None
            if "effect" in value.keys():
                scene.add_action_story(
                    target_state, key, value["description"], value["effect"]
                )
            else:
                args = (scene, target_state, key, value["description"])
                jobs.append(args)

        for name in character_names:
            scene.add_action_talk(name, self.characters[name])
        self.scenes[name] = scene
        return jobs

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
