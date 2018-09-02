import pickle
import os.path


class Vocabulary(object):

    def __init__(self,
                 vocab_threshold,
                 vocab_file,
                 start_word="<start>",
                 end_word="<end>",
                 unk_word="<unk>",
                 force_rebuild=False):
        """Initialize the vocabulary.
        Args:
          vocab_threshold: Minimum word count threshold.
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          force_rebuild: If True, reinit vocab to build it from scratch & override any existing vocab_file
                        If False, load vocab from from existing vocab_file, if it exists
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.force_rebuild = force_rebuild
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) and self.force_rebuild is False:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file!')
        else:
            self.init_vocab()

    def init_vocab(self):
        """Prepare new build of dictionary"""
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)

    def build_vocab(self, words_counter):
        words = [word for word, cnt in words_counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

        with open(self.vocab_file, 'wb') as f:
            pickle.dump(self, f)

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)