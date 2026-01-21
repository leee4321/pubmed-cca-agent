"""Tools for the ReAct agent (Wikipedia search and lookup)."""
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError


def search_wikipedia(query: str, search_history: list, search_urls: list, model, clean_func):
    """Performs search on query via Wikipedia API and returns its summary.

    Args:
        query: Search parameter to query the Wikipedia API with.
        search_history: List to track search history.
        search_urls: List to track URLs visited.
        model: Gemini model instance for summarization.
        clean_func: Function to clean text.

    Returns:
        observation: Summary of Wikipedia search for query if found else
        similar search results.
    """
    observation = None
    query = query.strip()
    
    try:
        # Try to get the summary for requested query from Wikipedia
        observation = wikipedia.summary(query, sentences=4, auto_suggest=False)
        wiki_url = wikipedia.page(query, auto_suggest=False).url
        observation = clean_func(observation)

        # If successful, return the first 2-3 sentences from the summary as model's context
        observation = model.generate_content(
            f'Return the first 2 or 3 sentences from the following text: {observation}'
        )
        observation = observation.text

        # Keep track of the model's search history
        search_history.append(query)
        search_urls.append(wiki_url)
        print(f"Information Source: {wiki_url}")

    # If the page is ambiguous/does not exist, return similar search phrases
    except (DisambiguationError, PageError) as e:
        observation = f'Could not find ["{query}"].'
        # Get a list of similar search topics
        search_results = wikipedia.search(query)
        observation += f' Similar: {search_results}. You should search for one of those instead.'

    return observation


def lookup_in_page(phrase: str, search_history: list, search_urls: list, clean_func, context_length=200):
    """Searches for the phrase in the latest Wikipedia search page.

    Args:
        phrase: Lookup phrase to search for within a page.
        search_history: List of search history.
        search_urls: List of URLs visited.
        clean_func: Function to clean text.
        context_length: Number of words to consider while looking for the answer.

    Returns:
        result: Context related to the phrase within the page.
    """
    # Get the last searched Wikipedia page and find phrase in it
    page = wikipedia.page(search_history[-1], auto_suggest=False)
    page = page.content
    page = clean_func(page)
    start_index = page.find(phrase)

    # Extract sentences considering the context length defined
    result = page[max(0, start_index - context_length):start_index + len(phrase) + context_length]
    print(f"Information Source: {search_urls[-1]}")
    return result
