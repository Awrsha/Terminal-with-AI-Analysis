import re
import shlex
import os
from typing import Tuple, List, Set

default_directory = r"C:\Users\pc\Desktop"
if os.path.exists(default_directory):
    app.config['CURRENT_DIRECTORY'] = default_directory
else:
    app.config['CURRENT_DIRECTORY'] = os.getcwd()
    logger.warning(f"Default directory {default_directory} not found, using {os.getcwd()} instead.")

class CommandSanitizer:
    def __init__(self, 
                 blocked_commands: List[str] = None, 
                 allowed_paths: List[str] = None,
                 max_arguments: int = 10,
                 blocked_patterns: List[str] = None):
        """
        Initialize the command sanitizer with security rules.
        
        Args:
            blocked_commands: List of commands that are completely blocked
            allowed_paths: List of paths that are allowed for command execution
            max_arguments: Maximum number of arguments allowed in a command
            blocked_patterns: Regular expression patterns that are blocked
        """
        self.blocked_commands = set(blocked_commands or [])
        self.allowed_paths = allowed_paths or []
        self.max_arguments = max_arguments
        self.blocked_patterns = [re.compile(pattern) for pattern in (blocked_patterns or [])]
        
        # Add default dangerous patterns if none provided
        if not blocked_patterns:
            self._add_default_patterns()
    
    def _add_default_patterns(self):
        """Add default dangerous command patterns."""
        dangerous_patterns = [
            r'rm\s+(-rf?|/s)\s+[/\\]',  # Recursive delete from root
            r'>(>?)\s*/dev/(null|zero)',  # Redirecting to special devices
            r'mkfs',  # Formatting drives
            r'dd\s+if=.*\s+of=',  # Direct disk operations
            r'chmod\s+777',  # Overly permissive permissions
            r'^sudo\s+',  # Commands with sudo
            r'passwd',  # Password changing
            r'useradd|userdel',  # User management
            r'(wget|curl)\s+.*\s*\|\s*bash',  # Piping downloaded content to bash
            r'>(>?)\s*[/\\]',  # Writing to root directories
            r'\|\s*(bash|sh|csh|ksh|zsh)',  # Piping to shell
            r'eval\s+',  # Eval commands
            r'source\s+',  # Source commands
            r':\(\)\s*{\s*:\|\:',  # Fork bomb pattern
            r'exec\s+',  # Exec commands
            r'chown',  # Change ownership
            r'shutdown|reboot|halt|poweroff',  # System control
            r'iptables|firewall',  # Firewall manipulation
            r'sysctl',  # Kernel parameter manipulation
            r'modprobe|insmod|rmmod',  # Kernel module manipulation
        ]
        
        self.blocked_patterns = [re.compile(pattern) for pattern in dangerous_patterns]
    
    def is_command_safe(self, command: str) -> Tuple[bool, str]:
        """
        Check if a command is safe to execute.
        
        Args:
            command: The command string to check
            
        Returns:
            Tuple of (is_safe, reason)
            - is_safe: Boolean indicating if the command is safe
            - reason: String describing why the command is not safe (if applicable)
        """
        # Validate empty command
        if not command or not command.strip():
            return False, "Empty command"
        
        # Split the command into arguments
        try:
            args = shlex.split(command)
        except ValueError as e:
            return False, f"Invalid command syntax: {str(e)}"
        
        # No arguments provided
        if not args:
            return False, "No command specified"
        
        # Check number of arguments
        if len(args) > self.max_arguments:
            return False, f"Too many arguments (max {self.max_arguments})"
        
        # Get the base command (first word)
        base_command = args[0].lower()
        
        # Check if the command is in the blocked list
        if base_command in self.blocked_commands:
            return False, f"Command '{base_command}' is blocked"
        
        # Check against blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(command.lower()):
                return False, f"Command matches blocked pattern: {pattern.pattern}"
        
        # Check allowed paths if specified
        if self.allowed_paths:
            command_path = self._find_command_path(base_command)
            if command_path and not self._is_path_allowed(command_path):
                return False, f"Command from path '{command_path}' is not allowed"
        
        # If we get here, the command is considered safe
        return True, ""
    
    def _find_command_path(self, command: str) -> str:
        """
        Find the full path of a command.
        
        Args:
            command: The command name
            
        Returns:
            The full path of the command or empty string if not found
        """
        # If the command contains a path separator, it's already a path
        if os.path.sep in command:
            return command
        
        # Otherwise, check if it's in the PATH
        path_env = os.environ.get('PATH', '')
        paths = path_env.split(os.pathsep)
        
        for path in paths:
            full_path = os.path.join(path, command)
            if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
                return full_path
        
        return ""
    
    def _is_path_allowed(self, path: str) -> bool:
        """
        Check if a path is in the allowed paths list.
        
        Args:
            path: The path to check
            
        Returns:
            True if the path is allowed, False otherwise
        """
        path = os.path.normpath(path)
        
        for allowed_path in self.allowed_paths:
            allowed_path = os.path.normpath(allowed_path)
            
            # Check if the path is the allowed path or is under it
            if path == allowed_path or path.startswith(allowed_path + os.path.sep):
                return True
        
        return False
    
    def sanitize_command(self, command: str) -> Tuple[str, bool, str]:
        """
        Sanitize a command by removing dangerous parts.
        
        Args:
            command: The command to sanitize
            
        Returns:
            Tuple of (sanitized_command, is_modified, reason)
            - sanitized_command: The sanitized command
            - is_modified: Boolean indicating if the command was modified
            - reason: Reason for modification, if any
        """
        original_command = command
        
        # Check if the command is safe
        is_safe, reason = self.is_command_safe(command)
        if is_safe:
            return command, False, ""
        
        # Try to sanitize by removing shell operators
        sanitized = re.sub(r'[;&|><`$]', ' ', command)
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        # If after sanitization the command is empty, return error
        if not sanitized:
            return "", True, "Command contains only unsafe operators"
        
        # Check if the sanitized command is safe
        is_safe, new_reason = self.is_command_safe(sanitized)
        if is_safe:
            return sanitized, True, f"Removed unsafe operators: {reason}"
        
        # If the command is still not safe, try to extract just the base command
        try:
            args = shlex.split(command)
            base_command = args[0]
            
            is_base_safe, _ = self.is_command_safe(base_command)
            if is_base_safe:
                return base_command, True, f"Removed all arguments due to security concerns: {reason}"
        except:
            pass
        
        # If we couldn't sanitize the command, return empty with reason
        return "", True, f"Could not sanitize command: {reason}"