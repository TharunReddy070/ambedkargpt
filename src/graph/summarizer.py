class CommunitySummarizer:
    def summarize(self, entities: list[str]) -> str:
        return "This community focuses on: " + ", ".join(entities[:10])
