import os
import json

from src.config import Configuration
from src.llm import StoryLLM, DialogueLLM
from src.scene import Scene, Action, SceneLLM


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


class State:
    def __init__(self, config: Configuration, name: str, scene: Scene):
        self.name = name
        self.scene = scene
        self.config = config
        self.just_started = True
        if scene:
            self.llm = SceneLLM(config, scene)

    def describe(self) -> str:
        if self.just_started:
            self.just_started = False
            return self.scene.describe_to_player()
        else:
            return ""

    def evaluate(self, user_input: str) -> (str, str):
        response: dict = self.llm.query(user_input)

        if "action" in response.keys():
            action_name = response["action"]
        else:
            raise ValueError("No action recommended by the LLM!")

        print(f"You chose \"{action_name}\"")
        action_answer = None
        if "description" in response.keys():
            action_answer = response["description"]

        action = self.scene.get_action(action_name)

        if not action:
            raise ValueError("There is no such action!")

        if not action.get_answer():
            if action_answer:
                action.set_answer(action_answer)
            else:
                raise ValueError("The LLM did not explain the action")

        return action.describe_to_player(), action.name


class EndState(State):
    def __init__(self, config: Configuration, story: str, setting: str):
        super().__init__(config, "end_state", None)
        self.llm = StoryLLM(self.config.gpt_version, self.config.openai_api_key, story, setting)
        self.last_input = ""

    def describe(self) -> str:
        return self.llm.query(self.last_input)

    def evaluate(self, user_input: str) -> (str, str):
        self.last_input = user_input
        return "No matter how much you cry, the game is over... sorry!", "end"


class DialogueState(State):
    def __init__(self, config: Configuration, setting: str):
        super().__init__(config, "end_state", None)
        character = "The witches are not very talkative and only respond to make fun of him or to insult him."
        self.llm = DialogueLLM(self.config.gpt_version, self.config.openai_api_key, character, setting)

    def describe(self) -> str:
        return ""

    def evaluate(self, user_input: str) -> (str, str):
        response = self.llm.query(user_input)
        action_name = "talk"
        if "state" in response.keys():
            if response["state"] == "stop":
                action_name = "stop_dialogue"
        text = ""
        if "response" in response.keys():
            text = response["response"]
        return text, action_name


class TransitionModel:
    def __init__(self, src: State):
        self.src = src
        self.model = {
            "scene": src,
            "advice": src,
            "self": src,
            "troll": src,
            "failure": src,
        }

    def add_transition(self, action_name: str, dst: State):
        self.model[action_name] = dst

    def evaluate(self, action_name: str) -> State:
        return self.model[action_name]


class Story:
    def __init__(self, config: Configuration):
        self.config = config
        # load scenes and states
        if self.config.story == "macbeth":
            scenes = [load_scene("start"), load_scene("willow_tree")]
        else:
            scenes = [
                load_scene("start", story=self.config.story),
                load_scene("kitchen", story=self.config.story),
                load_scene("living_room", story=self.config.story),
                      ]
        # configure FSA
        self.states = []
        self.models = {}

        # add states
        for scene in scenes:
            self.states.append(State(self.config, scene.name, scene))
        self.initial_state = self.states[0]
        self.current_state = self.initial_state

        # add transitions
        for state in self.states:
            model = self.load_model(state.name)
            self.models[state.name] = model

    def get_state(self, name: str):
        for state in self.states:
            if state.name == name:
                return state
        raise ValueError("There is no such state in the story!")

    def load_ending(self, state_name: str, action_name: str) -> EndState:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, f"stories/{self.config.story}/story/endings.txt"), "r") as file:
            content = file.read()
            lines = content.split('\n')
            key = f"{state_name},{action_name}"
            for i in range(len(lines)):
                line = lines[i]
                if line == key:
                    return EndState(self.config, lines[i+1], self.get_state(state_name).scene.describe_to_llm())
                else:
                    i += 1

    def load_model(self, state_name: str) -> TransitionModel:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, f"stories/{self.config.story}/models/{state_name}.json")
        src = self.get_state(state_name)
        actions = src.scene.actions
        actions = [action.name for action in actions]
        model = TransitionModel(src)
        with open(path, 'r') as file:
            dictionary = json.load(file)
            for key, value in dictionary.items():
                actions.remove(key)
                if value == "end":
                    final_state = self.load_ending(src.name, key)
                    model.add_transition(key, final_state)
                    continue
                model.add_transition(key, self.get_state(value))

        for a in actions:
            model.add_transition(a, src)

        return model

    def question(self) -> str:
        return self.current_state.describe()

    def answer(self, user_input: str) -> str:
        response, action_name = self.current_state.evaluate(user_input)

        if action_name == "end":
            return response

        model = self.models[self.current_state.name]
        self.current_state = model.evaluate(action_name)
        return response

