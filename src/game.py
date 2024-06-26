import os
from typing import Tuple

from src.config import Configuration
from src.llm import ActorLLM
from src.memory import VectorStore
from src.object import parse_object, Object, Fact, Interaction
from src.scene import Scene
from src.util import load_json

colors = {
    "red": "\033[31m",
    "green": "\033[32m",
    "blue": "\033[34m",
}


def log(messages: list[str]):
    end_symbol = "\033[0m"
    color_queue = list(colors.keys())
    for msg in messages:
        color = color_queue.pop(0)
        color_queue.append(color)
        content = colors[color] + msg + end_symbol
        print(content)


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
            self.parse_scene_folder(os.path.join(path_to_scenes, file))

        # set first to current scene
        self.current_scene: Scene = list(self.scenes.values())[0]
        # init llm determining the current action
        self.llm = ActorLLM(self.memory, self.config)

    def run(self):
        while True:
            user_input = input()
            answer = self.next(user_input)
            log(answer)

    def next(self, user_input: str) -> list[str]:
        if self.finished:
            return ["restart"]
        self.llm.set_prompt(self.current_scene.actor_prompt())
        action_name, action_explanation = self.llm.query(user_input)
        llm_output, scene = self.current_scene.evaluate(
            action_name, action_explanation, user_input
        )
        if scene:
            if scene == "end":
                self.finished = True
            self.current_scene = self.scenes[scene]

        unique_llm_output: set[str] = set()
        threshold = 0.7
        while len(llm_output) > 0:
            # determine bs
            a = llm_output.pop(0)
            bs: list[str] = [a]
            for b in llm_output:
                similarity = self.memory.compare(a, b)
                print(f"similarity score: {similarity} a: {a} b: {b}")
                if similarity > threshold:
                    bs.append(b)
            bs = list(sorted(bs, key=lambda x: len(x)))
            unique_llm_output.add(bs.pop())
            llm_output = [x for x in llm_output if x not in bs]

        for x in unique_llm_output:
            self.memory.insert(user_input, x)
        return list(unique_llm_output)

    def parse_scene_folder(self, path_to_dir: str):
        scene_data, children = Game.parse_folder(path_to_dir)
        name = scene_data["name"]
        summary = scene_data["summary"]
        description = scene_data["description"]
        fact_data: list[dict] = scene_data["facts"]
        facts = []
        for fact in fact_data:
            fact_descr = fact["description"]
            if "reveals" in fact.keys():
                facts.append(Fact(fact_descr, reveals=fact["reveals"]))
            else:
                facts.append(Fact(fact_descr))

        interaction_data: list[dict] = scene_data["interactions"]
        interactions = []
        for interaction in interaction_data:
            interaction_descr = interaction["description"]
            if "reveals" in interaction.keys():
                interactions.append(
                    Interaction(interaction_descr, reveals=interaction["reveals"])
                )
            else:
                interactions.append(Interaction(interaction_descr))
        character_names = scene_data["characters"]
        characters = {"player": self.characters["player"]}
        for character_name in character_names:
            characters[character_name] = self.characters[character_name]

        self.scenes[name] = Scene(
            self.memory,
            self.config,
            name,
            summary,
            description,
            facts,
            interactions,
            children,
            characters,
        )

    def parse_character_json(self, path_to_character_file: str):
        character_data = load_json(path_to_character_file)
        name = character_data["name"]
        self.characters[name] = parse_object(character_data)

    @staticmethod
    def parse_folder(path_to_dir: str) -> Tuple[dict, dict[str, Object]]:
        parent_name = os.path.basename(path_to_dir)
        parent_data = None
        children = {}
        for entry in os.scandir(path_to_dir):
            if entry.is_file():
                if parent_name in str(entry):
                    parent_data = load_json(os.path.join(path_to_dir, entry))
                else:
                    data = load_json(os.path.join(path_to_dir, entry))
                    child = parse_object(data)
                    children[child.name] = child
            elif entry.is_dir():
                sub_parent_data, children = Game.parse_folder(
                    os.path.join(path_to_dir, entry)
                )
                sub_parent = parse_object(sub_parent_data)
                children[sub_parent.name] = sub_parent
                children.update(children)

        return parent_data, children
