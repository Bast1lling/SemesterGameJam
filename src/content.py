from abc import ABC, abstractmethod
from typing import Union


class Describer(ABC):
    def __init__(self, description: str):
        self._description = description

    @abstractmethod
    def describe(self) -> str:
        pass


class Character(Describer):
    def __init__(self, name: str, description: str):
        super().__init__(description)
        self.name = name

    def describe(self) -> str:
        return f"The character is called {self.name}. {self._description}"


class Object(Describer):
    def __init__(
            self, name: str, effect: str, target_scene: Union[str, None], description: str
    ):
        super().__init__(description)
        self.name = name
        self.effect = effect
        self.target_scene = target_scene
        self.prerequisites = []

    def set_prerequisites(self, prerequisites: list[str]):
        self.prerequisites = prerequisites

    def describe(self) -> str:
        return f"The {self.name} can be described like this: {self._description}"


class Item(Describer):
    def __init__(self, name: str, description: str):
        super().__init__(description)
        self.name = name

    def describe(self) -> str:
        return f'The Item "{self.name}" can be described like this: {self._description}'
