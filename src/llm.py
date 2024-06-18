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
    # decides which action the player wants to take
    ACTION = "action"
    # adds immersive details to a story element
    STORY = "story"
    # simulates a NPC conversation
    DIALOGUE = "dialogue"
    # answers player questions
    QUESTION = "question"


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

    def add_bool_parameter(self, parameter_name: str, parameter_description: str):
        self.parameters["properties"][parameter_name] = LLMFunction._boolean_parameter(
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
    def _boolean_parameter(description: str) -> dict:
        return {"type": "boolean", "description": description}

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
    if llm_type == LLMType.ACTION:
        llm_function.add_string_parameter("action", "chosen action")
    elif llm_type == LLMType.QUESTION:
        return None
    elif llm_type == LLMType.STORY:
        return None
    elif llm_type == LLMType.DIALOGUE:
        llm_function.add_bool_parameter("continue", "should the conversation continue?")
        llm_function.add_string_parameter("response", "dialogue response")
    else:
        raise ValueError("There exists no such LLMType")
    return llm_function


def get_system_prompt(llm_type: LLMType) -> SystemPrompt:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, f"prompts/{llm_type.name.lower()}_system.txt"), "r") as file:
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
        # TODO configure this path if you want to save the prompts
        self._save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

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
            content = response.choices[0].message.function_call.arguments
            content = json.loads(content)
        else:
            response = openai.chat.completions.create(
                model=self.gpt_version,  # Use the appropriate model
                messages=messages
            )
            content = response.choices[0].message.content

        if self._save:

            self._save_iteration_as_json(messages, content)
            self._save_iteration_as_txt(messages, content)
            return content
        else:
            print(f"*\t <Prompt> failed, no response is generated")
            return None

    # helper function to save both prompts and responses in a human-readable form
    def _save_iteration_as_txt(self, messages, content_json):
        identifier = self.__class__.__name__
        text_filename_result = f"result_{identifier}.txt"
        text_filename_prompt = f"prompt_{identifier}.txt"
        txt_save_dir = os.path.join(self._save_dir, "text")

        if not os.path.exists(txt_save_dir):
            os.makedirs(txt_save_dir, exist_ok=True)

        with open(os.path.join(txt_save_dir, text_filename_prompt), "w") as txt_file:
            for d in messages:
                for value in d.values():
                    if type(value) is str:
                        txt_file.write(value + "\n")
                    elif type(value) is list:
                        for item in value:
                            txt_file.write(json.dumps(item))

        if isinstance(content_json, str):
            with open(os.path.join(txt_save_dir, text_filename_result), "w") as txt_file:
                txt_file.write(content_json)
        else:
            with open(os.path.join(txt_save_dir, text_filename_result), "w") as txt_file:
                for value in content_json.values():
                    if isinstance(value, str):
                        txt_file.write(value + "\n")
                    elif isinstance(value, list):
                        for item in value:
                            txt_file.write(json.dumps(item) + "\n")

    # helper function to save both prompts and
    def _save_iteration_as_json(self, messages, content_json):
        identifier = self.__class__.__name__
        json_filename_result = f"result_{identifier}.txt"
        json_filename_prompt = f"prompt_{identifier}.txt"
        json_save_dir = os.path.join(self._save_dir, "json")
        if not os.path.exists(json_save_dir):
            os.makedirs(json_save_dir, exist_ok=True)
        # save the prompt
        with open(os.path.join(json_save_dir, json_filename_prompt), "w") as file:
            json.dump(messages, file)
        # save the result
        if isinstance(content_json, str):
            return
        with open(os.path.join(json_save_dir, json_filename_result), "w") as file:
            json.dump(content_json, file)


class QuestionLLM(LLM):
    def __init__(self, config: Configuration, game_state: str):
        super().__init__(LLMType.QUESTION, config.gpt_version, config.openai_api_key, config.save_traffic)
        # user prompt syntax
        self.user_prompt_structure = [
            "game_state",
            "question",
        ]
        self.user_prompt = Prompt(self.user_prompt_structure)
        # user prompt semantic
        self.user_prompt.set("game_state", f"This is the current game state: {game_state}\n")

    def _get_user_prompt(self) -> str:
        return self.user_prompt.__str__()

    def query(self, user_input: str):
        self.user_prompt.set("question", f"This is the player's question: {user_input}\nWhat would you answer?")
        return self._query()


# creates immersion by adding details to a story part
class StoryLLM(LLM):

    def __init__(self, config: Configuration, scene_descr: str, story_part: str):
        super().__init__(LLMType.STORY, config.gpt_version, config.openai_api_key, config.save_traffic)
        # user prompt syntax
        self.user_prompt_structure = [
            "scene",
            "story",
        ]
        self.user_prompt = Prompt(self.user_prompt_structure)
        # user prompt semantic
        self.user_prompt.set("scene", f"This is the current setting: {scene_descr}")
        self.user_prompt.set("story", f"Now describe the following in more detail: {story_part}")

    def _get_user_prompt(self) -> str:
        return self.user_prompt.__str__()

    def query(self):
        return self._query()


class DialogueLLM(LLM):
    def __init__(self, config: Configuration, scene_descr: str, character_descr: str):
        super().__init__(LLMType.DIALOGUE, config.gpt_version, config.openai_api_key, config.save_traffic)
        # user prompt syntax
        self.user_prompt_structure = [
            "scene",
            "character",
            "history",
        ]
        self.user_prompt = Prompt(self.user_prompt_structure)
        # user prompt semantic
        self.user_prompt.set("scene", f"This is the current setting: {scene_descr}")
        self.user_prompt.set("character", f"This is the character(s) you should simulate: {character_descr}")
        history_content = "This is the conversation so far, continue it:\n"
        self.user_prompt.set("history", history_content)

    def _get_user_prompt(self) -> str:
        return self.user_prompt.__str__()

    def query(self, user_input):
        # append current player question
        history = self.user_prompt.get("history")
        history += f"Player: {user_input}\n"
        self.user_prompt.set("history", history)
        response = self._query()
        if "response" in response.keys():
            # append model response
            history += f"Character: {response}\n"
            self.user_prompt.set("history", history)
        return response


class ActionLLM(LLM):
    def __init__(
            self,
            config: Configuration,
            scene_descr: str,
            actions_descr: str,
    ):
        super().__init__(LLMType.ACTION, config.gpt_version, config.openai_api_key, config.save_traffic)
        # user prompt syntax
        self.user_prompt_structure = [
            "scene",
            "actions",
            "user",
        ]
        self.user_prompt = Prompt(self.user_prompt_structure)
        # user prompt semantic
        self.user_prompt.set("scene", f"This is where the player currently is: {scene_descr}")
        actions_content = f"The player can take the following actions:\n{actions_descr}"
        self.user_prompt.set("actions", actions_content)

    def _get_user_prompt(self) -> str:
        return self.user_prompt.__str__()

    def query(self, user_input: str):
        self.user_prompt.set("user", f"This was the input of the user: {user_input}")
        result: dict = self._query()
        return result["action"]
