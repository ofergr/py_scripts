from random import choice
import string

MAX_INCORRECT_GUESSES = 6

def select_word() -> str:
    with open("words.txt", mode='r') as words:
        word_list = words.readlines()
    return choice(word_list).strip()

def get_player_input(guessed_letters:str) -> str:
    while True:
        player_input = input("guess a letter: ").lower()
        if _validate_input(player_input,guessed_letters):
            return player_input

def _validate_input(player_input:str, guessed_letters:str) -> bool:
    return (
        len(player_input) == 1
        and player_input in string.ascii_lowercase
        and player_input not in guessed_letters
    )

def join_guessed_letters(guessed_letters:str) -> str:
    return " ".join(sorted(guessed_letters))

def build_guessed_word(target_word:str, guessed_letters:set[str])->str:
    current_letters = []
    for letter in target_word:
        if letter in guessed_letters:
            current_letters.append(letter)
        else:
            current_letters.append("_")
    return " ".join(current_letters)

def game_over(wrong_guesses:int, target_word:str, guessed_letters:str)->bool:
    if wrong_guesses == MAX_INCORRECT_GUESSES:
        return True
    if set(target_word) <= guessed_letters:
        return True
    return False

def draw_hanged_man(wrong_guesses:str)->str:
    hanged_man = [
        r"""
  -----
  |   |
      |
      |
      |
      |
      |
      |
      |
      |
-------
""",
        r"""
  -----
  |   |
  O   |
      |
      |
      |
      |
      |
      |
      |
-------
""",
        r"""
  -----
  |   |
  O   |
 ---  |
  |   |
  |   |
      |
      |
      |
      |
-------
""",
        r"""
  -----
  |   |
  O   |
 ---  |
/ |   |
  |   |
      |
      |
      |
      |
-------
""",
        r"""
  -----
  |   |
  O   |
 ---  |
/ | \ |
  |   |
      |
      |
      |
      |
-------
""",
        r"""
  -----
  |   |
  O   |
 ---  |
/ | \ |
  |   |
 ---  |
/     |
|     |
      |
-------
""",
        r"""
  -----
  |   |
  O   |
 ---  |
/ | \ |
  |   |
 ---  |
/   \ |
|   | |
      |
-------
""",
    ]

    print(hanged_man[wrong_guesses])

if __name__ == "__main__":
    # Initial setup
    target_word = select_word()
    guessed_letters = set()
    guessed_word = build_guessed_word(target_word, guessed_letters)
    wrong_guesses = 0
    print("Welcome to Hangman!")

    while not game_over(wrong_guesses, target_word, guessed_letters):
        draw_hanged_man(wrong_guesses)
        print(f"Your word is: {guessed_word}")
        print(
            "Current guessed letters: "
            f"{join_guessed_letters(guessed_letters)}\n"
        )

        player_guess = get_player_input(guessed_letters)
        if player_guess in target_word:
            print(f"letter {player_guess} is part of the word!")
        else:
            print(f"Sorry, {player_guess} is not there.")
            wrong_guesses += 1

        guessed_letters.add(player_guess)
        guessed_word = build_guessed_word(target_word, guessed_letters)
    print (f"The word was \'{target_word}\'")