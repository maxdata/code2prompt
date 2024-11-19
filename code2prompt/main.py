"""Main module for the code2prompt CLI tool."""

import logging
from pathlib import Path
from code2prompt.commands.analyze import AnalyzeCommand
from code2prompt.commands.generate import GenerateCommand
from code2prompt.utils.logging_utils import setup_logger
from code2prompt.commands.interactive_selector import InteractiveFileSelector
from code2prompt.core.file_path_retriever import retrieve_file_paths


def generate(ctx, path=(), syntax_map=None, output=None, **options):
    """Generate markdown from code files"""

    # Parse the syntax_map option into a dictionary
    if syntax_map:
        syntax_map_dict = {'.' + ext.strip(): syntax.strip() for ext, syntax in 
                           (mapping.split(':') for mapping in syntax_map.split(','))}
        options['syntax_map'] = syntax_map_dict  # Replace the string with the dictionary

    config = ctx['config'].merge(options)
    logger = setup_logger(level=config.log_level)

    selected_paths = [Path(p) for p in config.path]

    # Check if selected_paths is empty before proceeding
    if not selected_paths:
        logging.error("No file paths provided. Please specify valid paths.")
        return  # Exit the function if no paths are provided

    filter_patterns = config.filter.split(",") if config.filter else []
    exclude_patterns = config.exclude.split(",") if config.exclude else []
    case_sensitive = config.case_sensitive
    gitignore = config.gitignore

    # Handle both directory and file inputs
    filtered_paths = []
    for path in selected_paths:
        if path.is_dir():
            filtered_paths.extend(retrieve_file_paths(
                file_paths=[path],
                gitignore=gitignore,
                filter_patterns=filter_patterns,
                exclude_patterns=exclude_patterns,
                case_sensitive=case_sensitive,
            ))
        elif path.is_file():
            filtered_paths.append(path)

    if filtered_paths and config.interactive:
        file_selector = InteractiveFileSelector(filtered_paths, filtered_paths)
        filtered_selected_path = file_selector.run()
        config.path = filtered_selected_path
    else:
        config.path = filtered_paths

    command = GenerateCommand(config, logger)
    command.execute()

    logger.info("Markdown generation completed.")
    

def analyze(ctx, path=(), format='flat'):
    """Analyze codebase structure"""
    config = ctx['config'].merge({'path': path, 'format': format})
    logger = setup_logger(level=config.log_level)
    logger.info("Analyzing codebase with options: %s", {'path': path, 'format': format})

    command = AnalyzeCommand(config, logger)
    command.execute()

    logger.info("Codebase analysis completed.")
