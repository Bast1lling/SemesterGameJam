# adapt from https://github.com/real-stanford/reflect
from datetime import datetime
import os
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


# A class representing an OpenAI api function call
class LLMFunction:
    def __init__(self):
        # initialize function parameters with summary array
        function_parameters_dict = {}

        self.parameters: dict = LLMFunction._object_parameter(function_parameters_dict)
        self.parameters["type"] = "object"

    # transforms the function into a form required by the OpenAI api
    def get_function_as_list(self):
        parameters = self.parameters["properties"]
        self.parameters["required"] = list(parameters.keys())
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


# interface class managing communication with OpenAI api through query method
class LLM:
    def __init__(
        self,
        gpt_version,
        api_key,
        save: bool,
        system_prompt: Prompt,
        user_prompt: Prompt,
        llm_function: LLMFunction,
        temperature=0.2,
    ) -> None:

        self.gpt_version = gpt_version
        openai.api_key = api_key
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.llm_function = llm_function
        self._save = save
        self._save_dir = "/home/sebastian/Documents/gamejam/Data/"

    # send <messages> to the OpenAI api
    def _query(
        self,
    ):
        messages = [
            {"role": "system", "content": self.system_prompt.__str__()},
            {
                "role": "user",
                "content": self.user_prompt.__str__(),
            },
        ]
        if self.llm_function:
            functions = self.llm_function.get_function_as_list()
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
class ImmersionLLM(LLM):
    def __init__(self, gpt_version, api_key, scene_descr: str, story_part: str):
        super().__init__(gpt_version, api_key, False, None, None, None)
        script_dir = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(script_dir, "prompts/immersion_system.txt"), "r") as file:
            self.system_prompt = SystemPrompt(file.read())

        self.user_prompt_structure = [
            "task",
            "scene",
            "story",
            "history",
        ]
        self.user_prompt = Prompt(self.user_prompt_structure)
        task_content = ("First you will be provided with a description of the current setting/ the current state of "
                        "the story. Then there will be a sentence describing another part of the story. Please create "
                        "a captivating description of this story part, but do not use more than three sentences!")
        self.user_prompt.set("task", task_content)
        self.user_prompt.set("scene", scene_descr)
        self.user_prompt.set("story", story_part)
        history_content = "These were the previous player messages to you:\n"
        self.user_prompt.set("history", history_content)

    def query(self, user_input):
        history = self.user_prompt.get("history")
        history += user_input + "\n"
        self.user_prompt.set("history", history)
        return self._query()


class ConversationLLM(LLM):
    def __init__(self, gpt_version, api_key, character_descr, setting):
        super().__init__(gpt_version, api_key, False, None, None, None)
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.llm_function = LLMFunction()
        self.llm_function.add_string_parameter("state", "conversation state")
        self.llm_function.add_string_parameter("response", "response to user")

        with open(os.path.join(script_dir, "prompts/conversation_system.txt"), "r") as file:
            self.system_prompt = SystemPrompt(file.read())

        self.user_prompt_structure = [
            "task",
            "setting",
            "character",
            "history",
            "user",
        ]
        self.user_prompt = Prompt(self.user_prompt_structure)
        with open(os.path.join(script_dir, "prompts/conversation.txt"), "r") as file:
            self.user_prompt.set("task", file.read())
        self.user_prompt.set("setting", setting)
        self.user_prompt.set("character", character_descr)
        history_content = "This is the conversation so far:\n"
        self.user_prompt.set("history", history_content)

    def query(self, user_input):
        history = self.user_prompt.get("history")
        user_content = f"This is the new response by the player: {user_input}"
        self.user_prompt.set("user", user_content)
        response = self._query()
        if "response" in response.keys():
            history += f"Player: {user_input}\n"
            history += f"Character: {response}\n"
            self.user_prompt.set("history", history)
        return response
