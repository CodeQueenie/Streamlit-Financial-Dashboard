"""
GitHub Copilot Chat History Export Tool

This module provides functionality to locate, export, and archive GitHub Copilot chat logs
and related files from VS Code installations. It searches common installation locations,
creates detailed reports of found files, and copies them to a designated archive location.

Features:
- Recursive directory searching for Copilot-related files
- Detailed Markdown report generation with file metadata
- Automated file copying with directory structure preservation
- Comprehensive error handling and logging
- Support for both chat logs and completion logs
- Timestamp-based organization of exports

Usage:
    python export_history.py

Dependencies:
    - Python 3.6+
    - pathlib
    - logging

Generated by Nicole LeGuern
"""

import json
import os
from datetime import datetime
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sanitize_path(path_str):
    """
    Sanitize a file system path to ensure it's valid for Windows.
    
    Args:
        path_str (str): The path string to sanitize
        
    Returns:
        str or None: The sanitized path string, or None if sanitization fails
        
    Example:
        >>> sanitize_path("C:\\Path\\With\\Invalid//Characters")
        "C:\\Path\\With\\Invalid\\Characters"
    """
    try:
        return str(Path(path_str))
    except Exception as e:
        logger.error(f"Path sanitization error: {e}")
        return None

def find_copilot_directories():
    """
    Search common installation locations for GitHub Copilot related directories.
    
    Searches through standard Windows installation paths including:
    - %APPDATA% (User's Roaming profile)
    - %LOCALAPPDATA% (User's Local profile)
    - %PROGRAMFILES%
    - %PROGRAMFILES(X86)%
    
    Returns:
        list: List of paths (str) where Copilot-related directories were found
        
    Notes:
        - Prints progress to console during search
        - Includes both Copilot and Copilot Chat directories
        - Lists contents of found directories
    """
    possible_locations = [
        os.path.expandvars("%APPDATA%"),  # AppData/Roaming
        os.path.expandvars("%LOCALAPPDATA%"),  # AppData/Local
        os.path.expandvars("%PROGRAMFILES%"),  # Program Files
        os.path.expandvars("%PROGRAMFILES(X86)%"),  # Program Files (x86)
    ]
    
    found_dirs = []
    
    for base in possible_locations:
        print(f"\nSearching in: {base}")
        for root, dirs, files in os.walk(base):
            if "copilot" in root.lower():
                print(f"Found Copilot-related directory: {root}")
                found_dirs.append(root)
                # List contents
                print("Contents:")
                for item in os.listdir(root):
                    print(f"  - {item}")
    
    return found_dirs

def generate_markdown_report(found_dirs):
    """
    Generate a detailed Markdown report of all found Copilot directories and their contents.
    
    Args:
        found_dirs (list): List of directory paths to document
        
    Returns:
        Path: Path object pointing to the generated report file
        
    Raises:
        PermissionError: If unable to create report directory or file
        Exception: For other IO or system errors
        
    Notes:
        Report includes:
        - Timestamp of generation
        - Full path of each directory
        - List of files with sizes and modification times
        - List of subdirectories
        - Error notifications for inaccessible items
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"Github_Copilot_Chat_Log_Locations_{timestamp}.md"
        
        # Use Path to handle path creation properly
        base_path = Path.cwd().parent if Path.cwd().name == "copilot_logging" else Path.cwd()
        report_path = base_path / "copilot_logging" / "reports"
        
        try:
            report_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created reports directory at: {report_path}")
        except PermissionError:
            logger.error("Permission denied when creating reports directory")
            raise
        except Exception as e:
            logger.error(f"Error creating reports directory: {e}")
            raise

        report_file = report_path / report_name
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# GitHub Copilot Chat Log Locations Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if not found_dirs:
                f.write("**No Copilot directories found in common locations.**\n")
                return report_file
                
            f.write("## Found Directories\n\n")
            for dir_path in found_dirs:
                f.write(f"### {dir_path}\n\n")
                try:
                    contents = os.listdir(dir_path)
                    if contents:
                        f.write("Contents:\n")
                        for item in contents:
                            try:
                                full_path = os.path.join(dir_path, item)
                                if os.path.isfile(full_path):
                                    try:
                                        size = os.path.getsize(full_path)
                                        modified = datetime.fromtimestamp(os.path.getmtime(full_path))
                                        f.write(f"- `{item}` (Size: {size:,} bytes, Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')})\n")
                                    except OSError as e:
                                        f.write(f"- `{item}` (Error reading file info: {e})\n")
                                else:
                                    f.write(f"- `{item}/` (Directory)\n")
                            except OSError as e:
                                f.write(f"- Error processing item '{item}': {e}\n")
                    else:
                        f.write("*(Empty directory)*\n")
                    f.write("\n")
                except PermissionError:
                    f.write(f"*Access denied to directory*\n\n")
                except Exception as e:
                    f.write(f"*Error reading directory: {str(e)}*\n\n")
        
        logger.info(f"Successfully generated report at: {report_file}")
        return report_file
    
    except Exception as e:
        logger.error(f"Failed to generate markdown report: {str(e)}")
        raise

def copy_chat_files(found_dirs, timestamp):
    """
    Copy all Copilot chat and completion log files to an exports directory.
    
    Args:
        found_dirs (list): List of directory paths containing files to copy
        timestamp (str): Timestamp string for naming the export directory
        
    Returns:
        tuple: (Path, list) containing:
            - Path: Export directory path where files were copied
            - list: List of Path objects for successfully copied files
        
    Notes:
        - Creates timestamped subdirectories for each export
        - Preserves original directory structure
        - Copies both .log and .json files
        - Skips files that can't be accessed
        - Logs all operations and errors
        
    Example:
        >>> export_path, copied_files = copy_chat_files(dirs, "20240213_151010")
        >>> print(f"Copied {len(copied_files)} files to {export_path}")
    """
    try:
        # Setup export directory
        base_path = Path.cwd().parent if Path.cwd().name == "copilot_logging" else Path.cwd()
        export_path = base_path / "copilot_logging" / "exports" / f"copilot_logs_{timestamp}"
        export_path.mkdir(parents=True, exist_ok=True)
        
        copied_files = []
        
        for dir_path in found_dirs:
            dir_path = Path(dir_path)
            try:
                # Look for Copilot log files
                log_files = list(dir_path.glob("*.log")) + \
                          list(dir_path.glob("*.json"))
                
                if log_files:
                    # Create subdirectory using parent folder name
                    sub_dir = export_path / dir_path.parent.name
                    sub_dir.mkdir(exist_ok=True)
                    
                    for log_file in log_files:
                        try:
                            # Copy file with its original name
                            dest_file = sub_dir / log_file.name
                            shutil.copy2(log_file, dest_file)
                            copied_files.append(dest_file)
                            logger.info(f"Copied {log_file.name} to {dest_file}")
                        except PermissionError:
                            logger.error(f"Permission denied accessing {log_file}")
                        except Exception as e:
                            logger.error(f"Error copying {log_file}: {e}")
                            
            except PermissionError:
                logger.error(f"Permission denied accessing directory {dir_path}")
            except Exception as e:
                logger.error(f"Error processing directory {dir_path}: {e}")
        
        return export_path, copied_files
    
    except Exception as e:
        logger.error(f"Error in copy_chat_files: {e}")
        return None, []

def export_chat_history():
    """
    Main function to export GitHub Copilot chat history from VS Code.
    
    This function orchestrates the entire export process:
    1. Searches for Copilot directories
    2. Copies all relevant files to exports directory
    3. Generates detailed markdown report
    
    Returns:
        None
        
    Notes:
        - Creates timestamped exports
        - Handles keyboard interrupts gracefully
        - Logs all operations
        - Provides console feedback
        
    Example:
        >>> export_chat_history()
        Starting Copilot directory search...
        Exported 15 files to /path/to/exports
        Report saved successfully to /path/to/report.md
    """
    try:
        logger.info("Starting Copilot directory search...")
        found_dirs = find_copilot_directories()
        
        if not found_dirs:
            logger.warning("No Copilot directories were found")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Copy log files
        export_path, copied_files = copy_chat_files(found_dirs, timestamp)
        if export_path:
            logger.info(f"Exported {len(copied_files)} files to {export_path}")
        
        # Generate and save markdown report
        try:
            report_file = generate_markdown_report(found_dirs)
            logger.info(f"Report saved successfully to: {report_file}")
        except PermissionError:
            logger.error("Permission denied when saving report. Try running with administrator privileges.")
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            
    except KeyboardInterrupt:
        logger.info("Search cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error during execution: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        export_chat_history()
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)
