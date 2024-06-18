# adapt from https://github.com/real-stanford/reflect
import shutil
from abc import ABC, abstractmethod
import os
from enum import Enum
from typing import Union, Tuple

import openai
import json

from src.config import Configuration
from src.memory import VectorStore
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
            required_parameters = [
                x for x in required_parameters if x not in not_required
            ]
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

    def add_string_array_parameter(self, parameter_name: str, parameter_description: str):
        items = {"type": "string"}
        self.parameters["properties"][parameter_name] = LLMFunction._array_parameter(
            items, parameter_description
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
    def _array_parameter(items: dict, description: str) -> dict:
        return {"type": "array", "items": items, "description": description}


# interface class managing communication with OpenAI api through query method
class LLM(ABC):
    def __init__(
            self,
            llm_type: str,
            vector_store: VectorStore,
            prompt_structure: list,
            gpt_version,
            api_key,
            save: bool,
            temperature,
    ) -> None:
        assert "memory" in prompt_structure
        self.memory = vector_store
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(
                os.path.join(script_dir, f"prompts/{llm_type}_system.txt"), "r"
        ) as file:
            self.system_prompt = SystemPrompt(file.read())
        self.user_prompt = Prompt(prompt_structure)
        self.llm_function = LLMFunction()
        self.gpt_version = gpt_version
        openai.api_key = api_key
        self.temperature = temperature
        self._save = save
        self._counter = 0
        # TODO configure this path if you want to save the prompts
        self._save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
        )
        shutil.rmtree(self._save_dir)
        os.makedirs(self._save_dir, exist_ok=True)

    @abstractmethod
    def _retrieve_memory(self, user_input: str) -> list[Tuple[str, str]]:
        pass

    # send <messages> to the OpenAI api
    def _query(self, user_input="", not_required: Union[list, None] = None):
        matches = self._retrieve_memory(user_input)
        print(f"matches: {len(matches)}")
        if len(matches) > 0:
            matches = [f"Player: {a} You: {b}" for a, b in matches]
            final = "\n".join(matches)
            memory_content = (
                f"Here are some previous interactions between the player and yourself which might be "
                f"helpful: {final} "
            )
            self.user_prompt.set("memory", memory_content)

        messages = [
            {"role": "system", "content": self.system_prompt.__str__()},
            {
                "role": "user",
                "content": self.user_prompt.__str__(),
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
                model=self.gpt_version, messages=messages  # Use the appropriate model
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
        text_filename_result = f"result_{identifier}_{self._counter}.txt"
        text_filename_prompt = f"prompt_{identifier}_{self._counter}.txt"
        self._counter += 1
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
            with open(
                    os.path.join(txt_save_dir, text_filename_result), "w"
            ) as txt_file:
                txt_file.write(content_json)
        else:
            with open(
                    os.path.join(txt_save_dir, text_filename_result), "w"
            ) as txt_file:
                for value in content_json.values():
                    if isinstance(value, str):
                        txt_file.write(value + "\n")
                    elif isinstance(value, list):
                        for item in value:
                            txt_file.write(json.dumps(item) + "\n")

    # helper function to save both prompts and
    def _save_iteration_as_json(self, messages, content_json):
        identifier = self.__class__.__name__
        json_filename_result = f"result_{identifier}_{self._counter}.json"
        json_filename_prompt = f"prompt_{identifier}_{self._counter}.json"
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


class DescriberLLM(LLM):
    def _retrieve_memory(self, user_input: str):
        question_match = set(self.memory.retrieve(user_input, n=1, min_similarity=0.0))
        object_matches = self.memory.retrieve(
            self.object_description, n=2, min_similarity=0.0
        )
        question_match.intersection(set(object_matches))
        return list(question_match)

    def __init__(self, memory: VectorStore, config: Configuration, scene_description: str, object_description: str):
        structure = [
            "scene",
            "object",
            "memory",
            "user",
        ]
        super().__init__(
            "describer",
            memory,
            structure,
            config.gpt_version,
            config.openai_api_key,
            config.save_traffic,
            config.temperature,
        )
        # user prompt semantic
        self.object_description = object_description
        self.user_prompt.set(
            "scene", f"This is the current scene: {scene_description}"
        )
        self.user_prompt.set(
            "object", f"And this is the object: {object_description}"
        )
        self.llm_function = None

    def query(self, user_input: str):
        self.user_prompt.set(
            "user",
            f"This is the player's input: {user_input}",
        )
        return self._query(user_input=user_input)


# creates immersion by adding details to a story part
class ExplorerLLM(LLM):
    def _retrieve_memory(self, user_input: str):
        return []

    def __init__(
            self,
            memory: VectorStore,
            config: Configuration,
            scene_descr: str,
            scene_layout: str,
            object_names: str,
    ):
        structure = [
            "scene",
            "objects",
            "memory",
            "user",
        ]
        super().__init__(
            "explorer",
            memory,
            structure,
            config.gpt_version,
            config.openai_api_key,
            config.save_traffic,
            config.temperature,
        )
        self.llm_function.add_string_parameter("description", "describe what the player sees")
        self.llm_function.add_string_array_parameter("objects", "objects seen by the player")
        scene_content = f"The scene is characterized by this: {scene_descr}\n"
        scene_content += f"From the player's perspective, the objects are positioned like this: {scene_layout}"
        names = " ".join(object_names)
        object_content = f"You can choose from the following set of objects. Please use exactly these names: ({names})"
        self.user_prompt.set("scene", scene_content)
        self.user_prompt.set("objects", object_content)

    def query(self, user_input: str):
        self.user_prompt.set(
            "user",
            f"This is the player's input: {user_input}",
        )
        return self._query(user_input=user_input)


class TalkerLLM(LLM):
    def _retrieve_memory(self, user_input: str):
        return self.memory.retrieve(
            user_input, n=1, min_similarity=0.0
        )

    def __init__(
            self,
            memory: VectorStore,
            config: Configuration,
            scene_descr: str,
            character_descr: str,
    ):
        structure = [
            "scene",
            "character",
            "memory",
            "history",
        ]
        super().__init__(
            "talker",
            memory,
            structure,
            config.gpt_version,
            config.openai_api_key,
            config.save_traffic,
            config.temperature,
        )
        self.llm_function.add_bool_parameter("continue", "should the conversation continue?")
        self.llm_function.add_string_parameter("response", "dialogue response")
        self.scene_description = scene_descr
        self.character_description = character_descr
        # user prompt semantic
        self.user_prompt.set("scene", f"This is the current setting: {scene_descr}")
        self.user_prompt.set(
            "character",
            f"This is the character(s) you simulate: {character_descr}",
        )
        history_content = "This is the conversation so far, continue it:\n"
        self.user_prompt.set("history", history_content)

    def query(self, user_input):
        # append current player question
        history = self.user_prompt.get("history")
        history += f"Player: {user_input}\n"
        self.user_prompt.set("history", history)
        response = self._query(user_input=user_input)
        if "response" in response.keys():
            # append model response
            history += f"Character: {response}\n"
            self.user_prompt.set("history", history)
        return response


class InteracterLLM(LLM):
    def _retrieve_memory(self, user_input: str):
        return []

    def __init__(
            self,
            memory: VectorStore,
            config: Configuration,
            object_descr: str,
            effect_descr: str,
    ):
        structure = [
            "object",
            "effect",
            "memory",
            "user",
        ]
        super().__init__(
            "interacter",
            memory,
            structure,
            config.gpt_version,
            config.openai_api_key,
            config.save_traffic,
            config.temperature,
        )
        self.llm_function = None
        self.user_prompt.set("object", f"This is object the user wants to interact with: {object_descr}")
        self.user_prompt.set("effect",
                             f"The effects which an interaction could have are described here: {effect_descr}")

    def query(self, user_input):
        self.user_prompt.set("user", f"This is what the player wants to do: {user_input}")
        return self._query(user_input=user_input)


class ThinkerLLM(LLM):
    def _retrieve_memory(self, user_input: str):
        return self.memory.retrieve(
            user_input, n=3, min_similarity=0.0
        )

    def __init__(
            self,
            memory: VectorStore,
            config: Configuration,
            scene_descr: str,
            scene_layout: str,
            self_descr: str,
    ):
        structure = [
            "scene",
            "self",
            "memory",
            "user",
        ]
        super().__init__(
            "thinker",
            memory,
            structure,
            config.gpt_version,
            config.openai_api_key,
            config.save_traffic,
            config.temperature,
        )
        self.llm_function = None
        self.user_prompt.set("scene", f"This is the current scene: {scene_descr} {scene_layout}")
        self.user_prompt.set("self", f"Here is some information about the player's character: {self_descr}")

    def query(self, user_input):
        self.user_prompt.set(
            "user",
            f"This is the player's input: {user_input}",
        )
        return self._query(user_input=user_input)


class ActorLLM(LLM):
    def _retrieve_memory(self, user_input: str):
        return []

    def __init__(
            self,
            memory: VectorStore,
            config: Configuration,
            scene_descr: str,
            object_names: list[str],
            character_names: list[str],
    ):
        structure = [
            "memory",
            "scene",
            "actions",
            "user",
        ]
        super().__init__(
            "actor",
            memory,
            structure,
            config.gpt_version,
            config.openai_api_key,
            config.save_traffic,
            config.temperature
        )
        self.llm_function.add_string_parameter("action", "chosen action")
        characters = " ".join(character_names)
        objects = " ".join(object_names)
        scene_content = f"This is the current scene: {scene_descr}"
        scene_content += f"These are the names of all characters which the player already knows of: ({characters})\n"
        scene_content += f"And these are the names of all object which he already knows of: ({objects})\n"
        self.user_prompt.set(
            "scene",
        )

        action_content = "What follows now is a list of actions which the player can take:\n"
        action_content += "When the player wants to find out more about his surroundings, you respond with" \
                          "\"explore\"\n"
        action_content += "When the player wants to know more about a specific object/character called <name>, " \
                          "you respond with \"describe_<name>\"\n "
        action_content += "When the player wants to talk to some character, you respond with " \
                          "\"talk_<name>\"\n"
        action_content += "When the player wants to interact with a specific object, you respond with " \
                          "\"interact_<name>\"\n"
        action_content += "If none of the previous actions match with the player's intent you can always respond with " \
                          "\"failure\"\n"
        self.user_prompt.set("actions", action_content)

    def query(self, user_input: str):
        self.user_prompt.set("user", f"This was the player's input, now determine his action: {user_input}")
        result: dict = self._query(user_input=user_input)
        return result["action"]
