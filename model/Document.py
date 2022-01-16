class Document:
    def __init__(self, title, text, author, popularity):
        self.title = title
        self.text = text
        self.author = author
        self.popularity = popularity

    def format(self):
        return [self.author + "\n" + self.title, self.text[:200] + ' ...']

    def __str__(self):
        return f"{self.author} {self.title} {self.popularity}: {self.text}"
