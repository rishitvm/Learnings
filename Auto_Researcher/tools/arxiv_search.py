import arxiv

def fetch_papers(topic: str, max_results: int = 3):
    """
        Searches arXiv for recent papers on a given topic.
        Returns a list of dictionaries with title and summary.
    """

    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.SubmittedDate
        )
    
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "url": result.entry_id
        })
    
    return papers