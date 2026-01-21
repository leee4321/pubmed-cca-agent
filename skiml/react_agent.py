"""ReAct Agent implementation using Gemini API."""
import os
import re
from google import genai
from google.genai import types

from config import get_api_key, setup_retry_policy
from prompts import get_react_prompt
from tools import search_wikipedia, lookup_in_page


class ReAct:
    """ReAct agent that combines reasoning and acting for question answering."""
    
    def __init__(self, model: str, react_prompt: str | os.PathLike = None):
        """Prepares Gemini to follow a Few-shot ReAct prompt.

        Args:
            model: Name of the model.
            react_prompt: ReAct prompt OR path to the ReAct prompt file.
                         If None, uses default prompt.
        """
        setup_retry_policy()
        
        api_key = get_api_key()
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.model = self.client.models.get(model)
        self.chat = self.client.chats.create(model=self.model_name, history=[])
        self.should_continue_prompting = True
        self._search_history: list[str] = []
        self._search_urls: list[str] = []

        if react_prompt is None:
            # Use default prompt
            self._prompt = get_react_prompt()
        else:
            try:
                # Try to read the file
                with open(react_prompt, 'r') as f:
                    self._prompt = f.read()
            except FileNotFoundError:
                # Assume that the parameter represents prompt itself
                self._prompt = react_prompt

    @property
    def prompt(self):
        """Get the ReAct prompt."""
        return self._prompt

    @staticmethod
    def clean(text: str) -> str:
        """Helper function to clean text by removing newlines."""
        return text.replace("\n", " ")

    def search(self, query: str) -> str:
        """Perform Wikipedia search.

        Args:
            query: Search parameter to query the Wikipedia API with.

        Returns:
            observation: Summary of Wikipedia search for query.
        """
        return search_wikipedia(
            query, 
            self._search_history, 
            self._search_urls, 
            self.model,
            self.clean
        )

    def lookup(self, phrase: str, context_length: int = 200) -> str:
        """Look up a phrase in the latest Wikipedia page.

        Args:
            phrase: Lookup phrase to search for within a page.
            context_length: Number of words to consider while looking for the answer.

        Returns:
            result: Context related to the phrase within the page.
        """
        return lookup_in_page(
            phrase, 
            self._search_history, 
            self._search_urls, 
            self.clean,
            context_length
        )

    def finish(self, _) -> None:
        """Finish the conversation.
        
        Sets the should_continue_prompting flag to False to stop the loop.
        """
        self.should_continue_prompting = False
        print(f"Information Sources: {self._search_urls}")

    def __call__(self, user_question: str, max_calls: int = 8, **generation_kwargs):
        """Start multi-turn conversation with function calling.

        Args:
            user_question: The question to answer.
            max_calls: Maximum calls made to the model to get the final answer.
            **generation_kwargs: Generation configuration parameters:
                - candidate_count: (int | None) = None
                - stop_sequences: (Iterable[str] | None) = None
                - max_output_tokens: (int | None) = None
                - temperature: (float | None) = None
                - top_p: (float | None) = None
                - top_k: (int | None) = None

        Raises:
            AssertionError: If max_calls is not between 1 and 8.
        """
        # Hyperparameter fine-tuned according to the paper
        assert 0 < max_calls <= 8, "max_calls must be between 1 and 8"

        if len(self.chat.get_history(curated=True)) == 0:
            model_prompt = self.prompt.format(question=user_question)
        else:
            model_prompt = user_question

        # Stop sequences for the model to imitate function calling
        callable_entities = ['</search>', '</lookup>', '</finish>']
        generation_kwargs.update({'stop_sequences': callable_entities})

        self.should_continue_prompting = True
        
        for idx in range(max_calls):
            self.response = self.chat.send_message_stream(
                model_prompt,
                config=types.GenerateContentConfig(**generation_kwargs)
            )

            for chunk in self.response:
                print(chunk.text, end=' ')

            response_cmd = self.chat.get_history(curated=True)[-1].parts[-1].text

            try:
                # Regex to extract function name written in between angular brackets
                cmd = re.findall(r'<(.*)>', response_cmd)[-1]
                print(f'</{cmd}>')
                
                # Regex to extract param
                query = response_cmd.split(f'<{cmd}>')[-1].strip()
                
                # Call to appropriate function
                observation = getattr(self, cmd)(query)

                if not self.should_continue_prompting:
                    break

                stream_message = f"\nObservation {idx + 1}\n{observation}"
                print(stream_message)
                
                # Send function's output as user's response
                model_prompt = f"<{cmd}>{query}</{cmd}>'s Output: {stream_message}"

            except (IndexError, AttributeError) as e:
                model_prompt = (
                    "Please try to generate thought-action-observation traces "
                    "as instructed by the prompt."
                )
