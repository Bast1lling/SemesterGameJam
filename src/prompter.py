class Prompt:
    def __init__(self, structure: list):
        # standard structure of a prompt
        self._content = {}
        for key in structure:
            self._content[key] = ""

    def __str__(self) -> str:
        prompt = ""
        for value in self._content.values():
            if isinstance(value, str):
                if not value:
                    continue
                prompt += value + "\n"
            elif isinstance(value, Prompt):
                prompt += value.__str__()
        return prompt

    def get_structure(self):
        return self._content.keys()

    def get(self, key) -> str:
        if key not in self._content.keys():
            return ""
        return self._content[key]

    def set(self, key, value):
        if key not in self._content.keys():
            return
        self._content[key] = value


class SystemPrompt(Prompt):
    def __init__(self, content: str):
        super().__init__(["system"])
        self.set("system", content)
