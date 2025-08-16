from collections import defaultdict
import math


class NaiveBayes:
    def __init__(self):
        # Initialize dictionaries for spam and not spam word frequencies
        self.spam_words = defaultdict(int)
        self.not_spam_word = defaultdict(int)
        self.spam_count = 0
        self.not_spam_count = 0

    def fit(self, X, y):
        # word counts for spam and not spam emails
        for text, label in zip(X, y):
            for word in text.split():
                if label == 1:
                    self.spam_words[word] += 1
                    self.spam_count += 1
                else:
                    self.not_spam_word[word] += 1
                    self.not_spam_count += 1

        # Calculate priors
        self.p_spam = y.sum() / y.shape[0]
        self.p_not_spam = 1 - self.p_spam

        vocab = set(self.spam_words.keys()).union(set(self.not_spam_word.keys()))
        self.vocab_size = len(vocab)
    
    def word_prob(self, word, label):
        if label == 1:  # spam
            return (self.spam_words[word] + 1) / (self.spam_count + self.vocab_size)
        else:  # ham
            return (self.not_spam_word[word] + 1) / (self.not_spam_count + self.vocab_size)
    
    def predict(self, emails):
        predictions = []
        for email in emails:
            emial_words = email.split()
            result = math.log(self.p_spam/self.p_not_spam)
            for word in emial_words:
                p1 = self.word_prob(word, 1)
                p0 = self.word_prob(word, 0)
                result += (math.log((p1*(1 - p0))/((1 - p1)*p0))) + math.log((1-p1)/(1-p0))
            if result >= 0:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions