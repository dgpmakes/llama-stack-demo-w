#!/usr/bin/env python3
"""
CLI tool for managing RAG operations with LlamaStack.

Usage:
    python run.py load
    python run.py search --query "your query" [--vector-store-id ID] [--max-results N]
"""

import argparse
import logging
import os
import sys
import traceback

from commands.load_command import load_command
from commands.search_command import search_command
from commands.agent_command import agent_command
from commands.model_command import list_command as model_list_command, info_command as model_info_command
from commands.tool_command import groups_command as tool_groups_command, list_tools_command as tool_list_command

# Function to find tool groups by name using a regular expression
# def find_tool_groups(client: LlamaStackClient, tool_name_pattern: str) -> List[ToolGroup]:
#     """
#     Find tool groups by name using a regular expression
#     """
#     return [tool_group for tool_group in client.toolgroups.list() if re.match(tool_name_pattern, tool_group.name)]

def main() -> None:
    # Get log level from environment variable, default to INFO if not set
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    # Configure logging at the module level or at the start of your script
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    # Set specific log level for httpx
    httpx_log_level = os.getenv('HTTPX_LOG_LEVEL', 'WARNING').upper()
    logging.getLogger('httpx').setLevel(getattr(logging, httpx_log_level))

    # Set specific log level for llama-stack-client
    llama_stack_client_log_level = os.getenv('LLAMA_STACK_CLIENT_LOG_LEVEL', log_level)
    logging.getLogger('llama_stack_client').setLevel(getattr(logging, llama_stack_client_log_level))

    logger = logging.getLogger(__name__)
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="CLI tool for managing RAG operations with LlamaStack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load documents into vector store
  python run.py load
  
  # Search the vector store
  python run.py search --query "Tell me about taxes"
  python run.py search --query "What is the capital?" --max-results 5
  python run.py search --query "Investment info" --vector-store-id my-store-id
  
  # Run agent commands
  python run.py agent --agent-type default --input "What is the weather?"
  python run.py agent --agent-type lang_chain --input "Analyze this data"
  python run.py agent --agent-type lang_graph --input "Complex workflow task"
  
  # List models
  python run.py model list
  python run.py model list --verbose
  python run.py model list --provider meta --type llm
  python run.py model info --model "meta-llama/Llama-3.2-3B-Instruct"
  
  # List tool groups and tools
  python run.py tool groups
  python run.py tool groups --verbose
  python run.py tool list --all
  python run.py tool list --group "my-tool-group"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True
    
    # Load command
    load_parser = subparsers.add_parser(
        "load",
        help="Load documents into the vector store"
    )
    
    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Search the vector store for relevant documents"
    )
    search_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The search query"
    )
    search_parser.add_argument(
        "--vector-store-id",
        type=str,
        default=None,
        help="ID of the vector store to search (if not provided, uses the latest)"
    )
    search_parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10)"
    )
    search_parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.8,
        help="Minimum score threshold for results (default: 0.8)"
    )
    search_parser.add_argument(
        "--ranker",
        type=str,
        default="default",
        help="Ranker to use for scoring (default: 'default')"
    )
    
    # Agent command
    agent_parser = subparsers.add_parser(
        "agent",
        help="Run an agent command with different agent types"
    )
    agent_parser.add_argument(
        "--agent-type",
        type=str,
        required=True,
        choices=["default", "lang_chain", "lang_graph"],
        help="Type of agent to use"
    )
    agent_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input text/prompt for the agent"
    )
    agent_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (default: from MODEL_NAME env var)"
    )
    agent_parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="System instructions (default: from SYSTEM_INSTRUCTIONS env var)"
    )
    agent_parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Skip loading MCP tools (agent will use model knowledge only)"
    )
    
    # Model command with subcommands
    model_parser = subparsers.add_parser(
        "model",
        help="Manage and list models"
    )
    model_subparsers = model_parser.add_subparsers(dest="model_subcommand", help="Model subcommands")
    model_subparsers.required = True
    
    # Model list subcommand
    model_list_parser = model_subparsers.add_parser(
        "list",
        help="List all available models"
    )
    model_list_parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Filter by provider ID (e.g., 'meta', 'ollama')"
    )
    model_list_parser.add_argument(
        "--type",
        type=str,
        default=None,
        help="Filter by model type (e.g., 'llm', 'embedding')"
    )
    model_list_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information about each model"
    )
    
    # Model info subcommand
    model_info_parser = model_subparsers.add_parser(
        "info",
        help="Show detailed information about a specific model"
    )
    model_info_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier to get information about"
    )
    
    # Tool command with subcommands
    tool_parser = subparsers.add_parser(
        "tool",
        help="Manage and list tool groups and tools"
    )
    tool_subparsers = tool_parser.add_subparsers(dest="tool_subcommand", help="Tool subcommands")
    tool_subparsers.required = True
    
    # Tool groups subcommand
    tool_groups_parser = tool_subparsers.add_parser(
        "groups",
        help="List all tool groups"
    )
    tool_groups_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information about each tool group"
    )
    
    # Tool list subcommand
    tool_list_parser = tool_subparsers.add_parser(
        "list",
        help="List tools from tool groups"
    )
    tool_list_parser.add_argument(
        "--all",
        action="store_true",
        help="List tools from all tool groups"
    )
    tool_list_parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="List tools from a specific tool group (by identifier)"
    )
    tool_list_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information about each tool"
    )
    
    args = parser.parse_args()
    
    # Execute the appropriate command
    try:
        if args.command == "load":
            logger.debug("Loading documents into the vector store...")
            load_command()
        elif args.command == "search":
            logger.debug("Searching the vector store for relevant documents...")
            search_command(
                query=args.query,
                vector_store_id=args.vector_store_id,
                max_results=args.max_results,
                score_threshold=args.score_threshold,
                ranker=args.ranker
            )
        elif args.command == "agent":
            # Handle --no-tools flag
            tools = [] if args.no_tools else None
            
            logger.debug("Running agent command...")
            response = agent_command(
                agent_type=args.agent_type,
                input_text=args.input,
                model_name=args.model,
                system_instructions=args.instructions,
                tools=tools
            )
            print(f"\nFinal Response:\n{response}")
        elif args.command == "model":
            if args.model_subcommand == "list":
                logger.debug("Listing models...")
                model_list_command(
                    filter_provider=args.provider,
                    filter_type=args.type,
                    verbose=args.verbose
                )
            elif args.model_subcommand == "info":
                logger.debug("Getting model information...")
                model_info_command(model_identifier=args.model)
            else:
                model_parser.print_help()
                sys.exit(1)
        elif args.command == "tool":
            if args.tool_subcommand == "groups":
                logger.debug("Listing tool groups...")
                tool_groups_command(verbose=args.verbose)
            elif args.tool_subcommand == "list":
                logger.debug("Listing tools...")
                tool_list_command(
                    group_name=args.group,
                    all_groups=args.all,
                    verbose=args.verbose
                )
            else:
                tool_parser.print_help()
                sys.exit(1)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        logger.debug("Operation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

