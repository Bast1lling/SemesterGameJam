import os
import json

from src.config import Configuration
from src.scene import Scene, Action, Character


def debug_msg(s: str, config: Configuration):
    if config.debug:
        print(s)


def load_scene(name: str, story="macbeth") -> Scene:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, f"stories/{story}/story/{name}.txt"), "r") as file:
        content = file.read()
        lines = content.split('\n')
        i = 0
        actions = []
        while i < len(lines):
            if i == 0:
                answer = lines[i]
            elif i == 1:
                description = lines[i]
            else:
                action_name = lines[i]
                action_description = lines[i + 1]
                action_answer = [lines[i + 2]]

                if len(action_answer[0]) <= 0:
                    actions.append(Action(action_name, action_description, None))
                    i = i + 3
                    continue

                i = i + 3
                line = lines[i]
                while len(line) > 0:
                    action_answer.append(line)
                    i += 1
                    line = lines[i]
                actions.append(Action(action_name, action_description, "".join(action_answer)))
            i += 1
        return Scene(name, description, answer, actions)


class Game:
    def __init__(self):
        self.config = Configuration()
        self.finished = False
        self.characters = {}
        self.scenes = {}

        # load character files
        path_to_characters = self.config.path_to_story + "characters/"
        files = os.listdir(path_to_characters)
        for file in files:
            if file.endswith('.json'):
                self.parse_character_json(os.path.join(path_to_characters,file))

        # load scene files
        path_to_scenes = self.config.path_to_story + "scenes/"
        files = os.listdir(path_to_scenes)
        files = sorted(files)
        for file in files:
            if file.endswith('.json'):
                self.parse_scene_json(os.path.join(path_to_scenes,file))

        # set first to current scene
        self.current_scene = list(self.scenes.values())[0]

    def next(self, user_input: str):
        if self.finished:
            return "restart"
        action_name = self.current_scene.llm.query(user_input)
        action = self.current_scene.get_action(action_name)
        data, scene = action.output(user_input)
        if scene:
            if scene == "end":
                self.finished = True
            self.current_scene = self.scenes[scene]
        return data

    def parse_scene_json(self, path_to_scene_file: str):
        scene_data = Game.load_json(path_to_scene_file)
        name = scene_data["name"]
        description = scene_data["description"]
        action_data: dict = scene_data["actions"]
        character_names: list = scene_data["characters"]
        characters = {k: self.characters[k] for k in character_names if k in self.characters}
        characters["self"] = self.characters["self"]
        scene = Scene(self.config, name, description, characters)
        for key, value in action_data.items():
            if "target_scene" in value.keys():
                target_state = value["target_scene"]
            else:
                target_state = None
            if "effect" in value.keys():
                scene.add_action_story(target_state, key, value["description"], value["effect"])
            else:
                scene.add_action_story(target_state, key, value["description"], None)

        for name in character_names:
            scene.add_action_talk(name, self.characters[name])
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