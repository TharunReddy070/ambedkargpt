class BufferMerger:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size

    def merge(self, sentences: list[str]) -> list[str]:
        merged = []
        for i in range(len(sentences)):
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            merged.append(" ".join(sentences[start:end]))
        return merged
