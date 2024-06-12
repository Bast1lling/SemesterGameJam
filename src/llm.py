# adapt from https://github.com/real-stanford/reflect
from abc import ABC, abstractmethod
from datetime import datetime
import os
from enum import Enum
from typing import Union

import openai
import json

from src.config import Configuration
from src.prompter import Prompt, SystemPrompt
from src.scene import Scene


def check_openai_api_key(api_key, mockup=False):
    if mockup:
        return True
    openai.api_key = api_key
    try:
        openai.models.list()
    except openai.AuthenticationError as _:
        return False
    else:
        return True


def mockup_query(
        iteration,
        directory="/home/sebastian/Documents/Uni/Bachelorarbeit/DrPlanner_Data/mockup/debug",
):
    filenames = []
    # finds all .jsons in the directory and assumes them to be mockup responses
    for file_name in os.listdir(directory):
        if file_name.endswith(".json"):
            # Construct the full file path
            file_path = os.path.join(directory, file_name)
            filenames.append(file_path)

    filenames.sort()
    index = iteration % len(filenames)
    filename_response = filenames[index]

    with open(filename_response) as f:
        return json.load(f)


# An enum representing all possible methods of interacting with OpenAI api
class LLMType(Enum):
    SCENE = "scene"
    STORY = "story"
    DIALOGUE = "dialogue"


# A class representing an OpenAI api function call
class LLMFunction:
    def __init__(self):
        # initialize function parameters with summary array
        function_parameters_dict = {}

        self.parameters: dict = LLMFunction._object_parameter(function_parameters_dict)
        self.parameters["type"] = "object"

    # transforms the function into a form required by the OpenAI api
    def get_function_as_list(self, not_required: Union[list, None]):
        required_parameters = list(self.parameters["properties"].keys())
        if not_required:
            required_parameters = [x for x in required_parameters if x not in not_required]
        self.parameters["required"] = required_parameters
        return [
            {
                "name": "game_master_response",
                "description": "Response to some user input",
                "parameters": self.parameters,
            }
        ]

    def add_string_parameter(self, parameter_name: str, parameter_description: str):
        self.parameters["properties"][parameter_name] = LLMFunction._string_parameter(
            parameter_description
        )

    def add_code_parameter(self, parameter_name: str, parameter_description: str):
        self.parameters["properties"][parameter_name] = LLMFunction._code_parameter(
            parameter_description
        )

    def add_number_parameter(self, parameter_name: str, parameter_description: str):
        self.parameters["properties"][parameter_name] = LLMFunction._number_parameter(
            parameter_description
        )

    @staticmethod
    def _number_parameter(description: str) -> dict:
        return {"type": "number", "description": description}

    @staticmethod
    def _string_parameter(description: str) -> dict:
        return {"type": "string", "description": description}

    @staticmethod
    def _code_parameter(description: str) -> dict:
        return {"type": "string", "format": "python-code", "description": description}

    @staticmethod
    def _object_parameter(properties: dict) -> dict:
        return {"type": "object", "properties": properties}

    @staticmethod
    def _add_array_parameter(items: dict, description: str) -> dict:
        return {"type": "array", "items": items, "description": description}


def get_function(llm_type: LLMType) -> Union[LLMFunction, None]:
    llm_function = LLMFunction()
    if llm_type == LLMType.SCENE:
        llm_function.add_string_parameter("action", "chosen action")
        llm_function.add_string_parameter("response", "description of the action")
    elif llm_type == LLMType.STORY:
        return None
    elif llm_type == LLMType.DIALOGUE:
        llm_function.add_string_parameter("state", "conversation state")
        llm_function.add_string_parameter("response", "response to user")
    else:
        raise ValueError("There exists no such LLMType")
    return llm_function


def get_system_prompt(llm_type: LLMType) -> SystemPrompt:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, f"prompts/{llm_type.name}_system.txt"), "r") as file:
        return SystemPrompt(file.read())


# interface class managing communication with OpenAI api through query method
class LLM(ABC):
    def __init__(
            self,
            llm_type: LLMType,
            gpt_version,
            api_key,
            save: bool,
            temperature=0.2,
    ) -> None:
        self.llm_type = llm_type
        self.system_prompt = get_system_prompt(self.llm_type)
        self.llm_function = get_function(self.llm_type)
        self.gpt_version = gpt_version
        openai.api_key = api_key
        self.temperature = temperature
        self._save = save
        self._save_dir = "/home/sebastian/Documents/gamejam/Data/"

    @abstractmethod
    def _get_user_prompt(self) -> str:
        pass

    # send <messages> to the OpenAI api
    def _query(self, not_required: Union[list, None] = None):
        messages = [
            {"role": "system", "content": self.system_prompt.__str__()},
            {
                "role": "user",
                "content": self._get_user_prompt(),
            },
        ]
        if self.llm_function:
            functions = self.llm_function.get_function_as_list(not_required)
            response = openai.chat.completions.create(
                model=self.gpt_version,
                messages=messages,
                functions=functions,
                function_call={"name": functions[0]["name"]},
                temperature=self.temperature,
            )
        else:
            response = openai.chat.completions.create(
                model=self.gpt_version,  # Use the appropriate model
                messages=messages
            )
            return response.choices[0].message.content

        # print("RESPONSE: " + str(response), Configuration().debug)
        if self._save and response:
            content = response.choices[0].message.function_call.arguments
            content_json = json.loads(content)

            self._save_iteration_as_json(messages, content_json)
            self._save_iteration_as_txt(messages, content_json)
            return content_json
        else:
            print(f"*\t <Prompt> failed, no response is generated")
            return None

    # helper function to save both prompts and responses in a human-readable form
    def _save_iteration_as_txt(self, messages, content_json):
        run_start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        text_filename_result = f"result_{run_start_time}.txt"
        text_filename_prompt = f"prompt_{run_start_time}.txt"
        txt_save_dir = self._save_dir + "text/"

        if not os.path.exists(txt_save_dir):
            os.makedirs(txt_save_dir, exist_ok=True)
        with open(os.path.join(txt_save_dir, text_filename_result), "w") as txt_file:
            for value in content_json.values():
                if isinstance(value, str):
                    txt_file.write(value + "\n")
                elif isinstance(value, list):
                    for item in value:
                        txt_file.write(json.dumps(item) + "\n")

        with open(os.path.join(txt_save_dir, text_filename_prompt), "w") as txt_file:
            for d in messages:
                for value in d.values():
                    if type(value) is str:
                        txt_file.write(value + "\n")
                    elif type(value) is list:
                        for item in value:
                            txt_file.write(json.dumps(item))

    # helper function to save both prompts and
    def _save_iteration_as_json(self, messages, content_json):
        run_start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        identifier = str(type(self))
        json_filename_result = f"result_{run_start_time}_{identifier}.txt"
        json_filename_prompt = f"prompt_{run_start_time}_{identifier}.txt"
        json_save_dir = self._save_dir + "json/"
        if not os.path.exists(json_save_dir):
            os.makedirs(json_save_dir, exist_ok=True)
        # save the prompt
        with open(os.path.join(json_save_dir, json_filename_prompt), "w") as file:
            json.dump(messages, file)
        # save the result
        with open(os.path.join(json_save_dir, json_filename_result), "w") as file:
            json.dump(content_json, file)


# creates immersion by adding details to a story part
class StoryLLM(LLM):

    def __init__(self, config: Configuration, scene_descr: str, story_part: str):
        super().__init__(LLMType.STORY, config.gpt_version, config.openai_api_key, config.save_traffic)
        # user prompt syntax
        self.user_prompt_structure = [
            "scene",
            "story",
        ]
        if config.history:
            self.user_prompt_structure.insert(0, "history")
        self.user_prompt = Prompt(self.user_prompt_structure)
        # user prompt semantic
        history_content = "These were the previous player messages to you, consider them to add immersion:\n"
        self.user_prompt.set("history", history_content)
        self.user_prompt.set("scene", f"This is the current setting: {scene_descr}")
        self.user_prompt.set("story", f"Now describe the following in more detail: {story_part}")

    def _get_user_prompt(self) -> str:
        return self.user_prompt.__str__()

    def query(self, user_input):
        history = self.user_prompt.get("history")
        history += user_input + "\n"
        self.user_prompt.set("history", history)
        return self._query()


class DialogueLLM(LLM):
    def __init__(self, config: Configuration, scene_descr: str, character_descr: str):
        super().__init__(LLMType.DIALOGUE, config.gpt_version, config.openai_api_key, config.save_traffic)
        # user prompt syntax
        self.user_prompt_structure = [
            "scene",
            "character",
        ]
        if config.history:
            self.user_prompt_structure.append("history")
        self.user_prompt = Prompt(self.user_prompt_structure)
        # user prompt semantic
        self.user_prompt.set("scene", f"This is the current setting: {scene_descr}")
        self.user_prompt.set("character", f"This is the character(s) you should simulate: {character_descr}")
        history_content = "This is the conversation so far, continue it:\n"
        self.user_prompt.set("history", history_content)

    def _get_user_prompt(self) -> str:
        return self.user_prompt.__str__()

    def query(self, user_input):
        history = self.user_prompt.get("history")
        history += f"Player: {user_input}\n"
        self.user_prompt.set("history", history)
        response = self._query()
        if "response" in response.keys():
            history += f"Character: {response}\n"
            self.user_prompt.set("history", history)
        return response


class SceneLLM(LLM):
    def __init__(
            self,
            config: Configuration,
            scene: Scene,
    ):
        super().__init__(LLMType.SCENE, config.gpt_version, config.openai_api_key, config.save_traffic)
        # user prompt syntax
        self.user_prompt_structure = [
            "scene",
            "actions",
            "user",
        ]
        if config.history:
            self.user_prompt_structure.insert(2, "history")
        self.user_prompt = Prompt(self.user_prompt_structure)
        # user prompt semantic
        self.user_prompt.set("scene", f"This is where the player currently is: {scene.describe_to_llm()}")
        actions_content = "The player can take the following actions:\n"
        for action in scene.actions:
            actions_content += action.describe_to_llm() + "\n"
        self.user_prompt.set("actions", actions_content)
        self.user_prompt.set("history", "These were the previous messages from the player:\n")

    def _get_user_prompt(self) -> str:
        return self.user_prompt.__str__()

    def query(self, user_input: str):
        self.user_prompt.set("user", f"This was the input of the user: {user_input}")
        result = self._query()
        history = self.user_prompt.get("history")
        if len(history) < 1000:
            history += f"{user_input}\n"
        self.user_prompt.set("history", history)
        return result
