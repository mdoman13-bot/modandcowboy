import random

def calculate_likeness(guess, password):
    return sum(g == p for g, p in zip(guess, password))

def main():
    # Load the words from the words.txt file
    with open('words.txt', 'r') as file:
        words = [word.strip() for word in file.read().split(',')]

    # Randomly select a password from the list
    password = random.choice(words)
    attempts = 5
    print("Welcome to Terminal Hacking")
    
    # Display the words
    for word in words:
        print(word)

    print(f"\nYou have {attempts} attempts left.")

    while attempts > 0:
        guess = input("Enter guess: ").strip()
        if guess not in words:
            print("Word not in list. Try again.")
            continue

        likeness = calculate_likeness(guess, password)
        print(f"Likeness: {likeness}")

        if guess == password:
            print("Access Granted!")
            break

        attempts -= 1
        print(f"You have {attempts} attempts left.")

    if attempts == 0:
        print("Terminal locked. Please contact an administrator.")

if __name__ == "__main__":
    main()
