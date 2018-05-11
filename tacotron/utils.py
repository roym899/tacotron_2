# define the basic vocabulary
VOCAB = " ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvwxyz-?!.,;:\'"

def text_to_sequence(text, length):
    sequence = []
    for _ in range(length):
        sequence.append(0)
    for i, c in enumerate(text):
        idx = VOCAB.find(c)
        assert idx != -1
        sequence[i] = idx
    return sequence
