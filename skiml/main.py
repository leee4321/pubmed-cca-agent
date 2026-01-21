"""Main entry point for the Gemini ReAct Agent."""
import argparse
from react_agent import ReAct


def main():
    """Run the ReAct agent with a user question."""
    parser = argparse.ArgumentParser(description='Gemini ReAct Agent for Question Answering')
    parser.add_argument(
        'question',
        type=str,
        nargs='?',
        default="What are the total of ages of the main trio from the new Percy Jackson and the Olympians TV series in real life?",
        help='Question to ask the agent'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.0-flash-exp',
        help='Gemini model to use (default: gemini-2.0-flash-exp)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.2,
        help='Temperature for generation (default: 0.2)'
    )
    parser.add_argument(
        '--max-calls',
        type=int,
        default=8,
        help='Maximum number of reasoning steps (default: 8)'
    )
    parser.add_argument(
        '--prompt-file',
        type=str,
        default=None,
        help='Path to custom ReAct prompt file (optional)'
    )
    
    args = parser.parse_args()
    
    # Create the ReAct agent
    print(f"Initializing ReAct agent with model: {args.model}\n")
    agent = ReAct(model=args.model, react_prompt=args.prompt_file)
    
    # Run the agent
    print(f"Question: {args.question}\n")
    print("=" * 80)
    agent(
        args.question,
        max_calls=args.max_calls,
        temperature=args.temperature
    )
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
