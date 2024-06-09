import re

from src.config import Configuration
from src.game import Story
from textual.app import App
from textual.widgets import Header, Footer, Input


def print_text(s: str, color: str):
    pattern = r'[.!?]'
    colors = {
        "red": "\033[31m",
        "green": "\033[32m",
        "blue": "\033[34m",
        "end": "\033[0m",
    }
    sentences = re.split(pattern, s)
    delimiters = re.findall(pattern, s)
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) <= 0:
            continue
        if i < len(delimiters):
            delimiter = delimiters[i]
        else:
            delimiter = "."
        print(colors[color] + sentence.strip() + delimiter + colors["end"])


class Game:
    def __init__(self):
        self.config = Configuration()
        self.story = Story(self.config)

    def run(self):
        # run question-answer cycle:
        while True:
            question = self.story.question()
            print_text(question, "blue")
            print("\n")
            user_input = input()
            answer = self.story.answer(user_input)
            print_text(answer, "green")
            print("\n")


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
