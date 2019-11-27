num_to_char = {}
alphabet = 'abcdefghijklmnopqrstuvwxyz'
for i in range(0, 10):
    num_to_char[i] = str(i)
for i in range(0, 26):
    num_to_char[i+10] = alphabet[i].upper()
for i in range(0, 26):
    num_to_char[i+10+26] = alphabet[i].lower()


if __name__ == "__main__":
    print(num_to_char)
