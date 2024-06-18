import re
from src.game import Game


def print_text(s: str, color: str):
    pattern = r"[.!?]"
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


def main():
    print_text("loading...", "blue")
    game = Game()
    # run question-answer cycle:
    intro = "Welcome to TextAdventureGPT! Just write what you want to do, ask for information or troll."
    print_text(intro, "blue")
    while True:
        user_input = input()
        answer = game.next(user_input)
        print_text(answer, "green")
        print("\n")


if __name__ == "__main__":
    main()
