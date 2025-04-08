import re

import pandas as pd


# 映射类
class Word2Index:

    def __init__(self, filename):
        df = pd.read_csv(filename)

        x = df['review']
        y = df['sentiment']
        x = x.apply(self.clean_html)
        x = x.apply(self.convert_lower)
        x = x.apply(self.remove_special)

        reviews = list(map(lambda s: s.split(), x))
        self.words = list(set(w for r in reviews for w in r))
        self.word2index = {w: self.words.index(w) for w in self.words}
        self.features = [list(set(self.word2index[w] for w in r)) for r in reviews]
        self.labels = list(y.map({'positive': 1, 'negative': 0}))

    @staticmethod
    def clean_html(text):
        p = re.compile(r'<.*?>')
        return p.sub('', text)

    @staticmethod
    def convert_lower(text):
        return text.lower()

    @staticmethod
    def remove_special(text):
        x = ''
        for t in text:
            x = x + t if t.isalnum() else x + ' '
        return x


# 加载数据（特征数据，标签数据，词汇表，映射表）
data = Word2Index('imdb.csv')
print(f'评论：{len(data.labels)}')
print(f'单词：{len(data.words)}')
