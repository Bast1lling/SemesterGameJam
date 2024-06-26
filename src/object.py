from typing import Union


class Fact:
    def __init__(self, fact_text: str, reveals=None):
        self.description = fact_text
        self.reveals: list = reveals

    def __str__(self):
        return self.description


class Interaction:
    def __init__(self, description: str, reveals=None):
        self.description = description
        self.reveals: list = reveals

    def __str__(self):
        return self.description


class Object:
    def __init__(
        self,
        name: str,
        summary: str,
        description: str,
        facts: list[Fact],
        interactions: list[Interaction],
        parent,
    ):
        self.name = name
        self.summary = summary
        self.description = description
        self.facts = facts
        self.revealed_facts = []
        self.interactions = interactions
        self.triggered_interaction = []
        self.parent = parent
        self.revealed = False

    def describer_prompt(self) -> str:
        result = f"Description of {self.name} for player: {self.description}\n"

        result += f"Discoverable facts:\n"
        for i, fact in enumerate(self.facts):
            result += f"Fact {i}: {fact}\n"
        return result[:-1]

    def interacter_prompt(self) -> str:
        result = f"Description of {self.name} for player: {self.description}\n"

        result += f"Possible interactions:\n"
        for i, interaction in enumerate(self.interactions):
            result += f"Interaction {i}: {interaction}\n"
        return result[:-1]

    def reveal_fact(
        self, updated_description: str, fact_indices: list[int]
    ) -> list[str]:
        self.description = updated_description
        revealed_objects = []
        for i in fact_indices:
            fact = self.facts.pop(i)
            if fact.reveals:
                revealed_objects.extend(fact.reveals)
            self.revealed_facts.append(fact)
        return revealed_objects

    def trigger_interaction(
        self, updated_description: str, interaction_indices: list[int]
    ) -> list[str]:
        self.description = updated_description
        revealed_objects = []
        for i in interaction_indices:
            interaction = self.interactions.pop(i)
            if interaction.reveals:
                revealed_objects.extend(interaction.reveals)
            self.triggered_interaction.append(interaction)
        return revealed_objects

    def summary_prompt(self) -> str:
        return f"{self.name}: {self.summary}"


class Character(Object):
    def __init__(
        self,
        name: str,
        summary: str,
        description: str,
        personality: str,
        facts: list[Fact],
        interactions: list[Interaction],
        parent,
    ):
        super().__init__(name, summary, description, facts, interactions, parent)
        self.personality = personality

    def talker_prompt(self) -> str:
        result = f"Description of {self.name} for player: {self.description}\n"

        result += f"{self.name}'s personality: {self.personality}"
        return result


def parse_object(object_data: dict, parent=None) -> Object:
    name = object_data["name"]
    summary = object_data["summary"]

    if "description" in object_data.keys():
        description = object_data["description"]
    else:
        description = summary

    fact_data: list[dict] = object_data["facts"]
    facts = []
    for fact in fact_data:
        fact_descr = fact["description"]
        if "reveals" in fact.keys():
            facts.append(Fact(fact_descr, reveals=fact["reveals"]))
        else:
            facts.append(Fact(fact_descr))

    interaction_data: list[dict] = object_data["interactions"]
    interactions = []
    for interaction in interaction_data:
        interaction_descr = interaction["description"]
        if "reveals" in interaction.keys():
            interactions.append(
                Interaction(interaction_descr, reveals=interaction["reveals"])
            )
        else:
            interactions.append(Interaction(interaction_descr))

    if "personality" in object_data.keys():
        personality = object_data["personality"]
        result = Character(
            name,
            summary,
            description,
            personality,
            facts,
            interactions,
            parent,
        )
    else:
        result = Object(name, summary, description, facts, interactions, parent)
    return result
