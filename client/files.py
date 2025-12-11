"""
File operations for LlamaStack.
"""

import logging

from pathlib import Path
from typing import List

from llama_stack_client import LlamaStackClient
from llama_stack_client.types.file import File
from typing_extensions import Literal

logger = logging.getLogger(__name__)

def list_files_in_folder(
    folder_path: str, 
    file_extensions: List[str] = ['.txt', '.md']
) -> List[Path]:
    """List files in a local folder and return file paths"""
    file_paths: List[Path] = []
    folder: Path = Path(folder_path)
    
    if not folder.exists():
        logger.warning(f"Folder {folder_path} does not exist")
        return file_paths
    
    logger.debug(f"Listing files in: {folder_path}")
    
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in file_extensions:
            logger.debug(f"Found file: {file_path.name}")
            file_paths.append(file_path)
            
    logger.debug(f"Successfully listed {len(file_paths)} files")
    return file_paths


def upload_file(
    client: LlamaStackClient,
    file: Path,
    purpose: Literal["assistants", "batch"] = "assistants",
) -> File:
    """
    Upload a file into the vector store.
    Args:
        client: The LlamaStack client
        file: The file to upload
        purpose: The purpose of the file
    Returns:
        The file object
    """
    if not file.exists():
        raise ValueError(f"File {file} does not exist")
    
    # For each file (FilePath) create a file object and upload it to the vector store
    file_response: File = client.files.create(
        file=file,
        purpose=purpose
    )
    return file_response

def upload_files(
    client: LlamaStackClient, 
    files: List[Path], 
) -> List[File]:
    """
    Upload files into the vector database.
    Args:
        client: The LlamaStack client
        files: The list of files to upload
    Returns:
        The list of file IDs
    """
    if not files:
        print("No files to upload")
        return
    
    file_ids: List[File] = [upload_file(client, file) for file in files]

    # If the number of files is not the same as the number of file IDs, raise an error
    if len(files) != len(file_ids):
        raise ValueError(f"Number of files {len(files)} is not the same as the number of file IDs {len(file_ids)}")

    return file_ids