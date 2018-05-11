# define the basic vocabulary
VOCAB = "ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvwxyz -?!.,;:\'"

def text_to_sequence(text):
    sequence = []
    for c in text:
        idx = VOCAB.find(c)
        assert idx != -1
        sequence.append(idx)
        return sequence
