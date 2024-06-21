class Rule:
    def __init__(self, rule: str):
        self.description = rule

    def __str__(self):
        return self.description


class Object:
    def __init__(
        self,
        name: str,
        summary: str,
        description: str,
        description_rules: list[Rule],
        interaction_rules: list[Rule],
        parent=None,
        target=None,
        children: list = None,
        explored: set = None,
    ):
        self.name = name
        self.summary = summary
        self.description = description
        self.description_rules = description_rules
        self.interaction_rules = interaction_rules
        self.parent = parent
        self.target = target

        self.children: dict[str, Object] = {}
        for child in children:
            self.children[child.name] = child

        self.explored = set()
        for obj_name in explored:
            if obj_name in self.children.keys():
                self.explored.add(obj_name)
            else:
                # todo debug msg
                print(
                    f"{obj_name} can not be added to explored, since it does not exist!"
                )

        self.not_explored = set(self.children.keys())
        self.not_explored = self.not_explored.difference(self.explored)

    def add_explored(self, obj_names: list[str]):
        for obj_name in obj_names:
            if obj_name in self.not_explored:
                self.not_explored.remove(obj_name)
                self.explored.add(obj_name)
            else:
                # todo debug msg
                print(f"{obj_name} is either explored already or does not exist!")

    def describer_prompt(self) -> str:
        result = f"The current description of {self.name} is: {self.description}\n"
        if self.parent:
            result += f"This is the parent-object of {self.name}: {self.parent.summary_prompt()}\n"
        if len(self.not_explored) > 0:
            result += f"The player does not yet know about the following children-objects of {self.name}:\n"
            for obj_name in self.not_explored:
                obj = self.children[obj_name]
                result += obj.summary_prompt()
        result += "It can be extended according to these rules:\n"
        for rule in self.description_rules:
            result += f"{rule}\n"
        return result[:-1]

    def interacter_prompt(self) -> str:
        result = f"The current description of {self.name} is: {self.description}"
        result += f"A player can interact with {self.name} according to these rules:\n"
        for rule in self.description_rules:
            result += f"{rule}\n"
        return result[:-1]

    def summary_prompt(self) -> str:
        return f"{self.name}: {self.summary}"


class Character(Object):
    def __init__(
        self,
        name: str,
        summary: str,
        description: str,
        personality: str,
        description_rules: list[Rule],
        interaction_rules: list[Rule],
        parent=None,
        children: list = None,
        explored: set = None,
    ):
        super().__init__(
            name,
            summary,
            description,
            description_rules,
            interaction_rules,
            parent=parent,
            children=children,
            explored=explored,
        )
        self.personality = personality

    def talker_prompt(self) -> str:
        result = f"This is a description of character {self.name}:\n"
        result += f"{self.description}\n"
        result += f"And this is {self.name}'s personality: {self.personality}"
        return result


def parse_object(name: str, object_data: dict, parent=None) -> Object:
    summary = object_data["summary"]
    description = object_data["description"]
    if "description_rules" in object_data.keys():
        description_rules = object_data["description_rules"]
    else:
        description_rules = ["DO NOT CHANGE ANYTHING OF THIS DESCRIPTION!"]
    description_rules = [Rule(s) for s in description_rules]

    if "interaction_rules" in object_data.keys():
        interaction_rules = object_data["interaction_rules"]
    else:
        interaction_rules = [
            f"The player can not interact with {name}. Tell him that it is senseless!"
        ]
    interaction_rules = [Rule(s) for s in interaction_rules]

    target = None
    if "target" in object_data.keys():
        target = object_data["target"]

    explored = set()
    if "explored" in object_data.keys():
        explored = set(object_data["explored"])

    children = []
    if "children" in object_data.keys():
        for child_name, child_data in object_data["children"].items():
            children.append(parse_object(child_name, child_data))

    if "personality" in object_data.keys():
        personality = object_data["personality"]
        result = Character(
            name,
            summary,
            description,
            personality,
            description_rules,
            interaction_rules,
            parent=parent,
            children=children,
            explored=explored,
        )
    else:
        result = Object(
            name,
            summary,
            description,
            description_rules,
            interaction_rules,
            parent=parent,
            target=target,
            children=children,
            explored=explored,
        )

    for c in children:
        c.parent = result

    return result
