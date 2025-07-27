import os 
import logging
import base64
from typing import (Any,
                    Dict,
                    Optional,
                    Type,
                    TypeVar,
                    Union,
                    cast)
from functools import lru_cache
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import dotenv
from pydantic import BaseModel,ValidationError
from config.app_config import get_config,ConfigManager,EnvConfigSource
from pathlib import Path
import re

logger = logging.getLogger(__name__)

T = TypeVar("T",bound=BaseModel)


class EnvironmentVariableError(Exception):
    """Exception raised for environment variable errors"""
    pass

class EnhancedEnvConfigSource(EnvConfigSource):
    """
    Enhanced version of EnvConfigSource that adds secure handling of environment variables.
    
    This class extends the functionality in app_config.py by adding:
    - Support for direct platform environment variables
    - Secure handling of sensitive values
    - Better error handling for missing variables
    - Automatic .env file management with encrypted values
    - No prefix requirement for environment variables
    """

    def __init__(self):
        super().__init__()
        # Override the ENV_PREFIX to be empty - no prefix required
        self.ENV_PREFIX = ""
        self._encryption_key = None
        self._cache = {}
        self._env_file_path = Path(".env")
        self._init_encryption()
        self._load_dotenv()
        
    def _init_encryption(self) ->None:
        """Initialize encryption for sensitive data"""
        secret = os.environ.get("ENCRYPTION_SECRET","wizard-secure-default-key")
        salt = b'wizard-salt'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        key = base64.urlsafe_b64encode(kdf.derive(secret.encode()))
        self._encryption_key = key

    def _load_dotenv(self,dotenv_path: Union[str,Path,None] = None, override : bool = False):
        """
        Load environment variables from .env file if available

        Args:
            dotenv_path: Path to .env file or None to search in default locations
            override: Whether to override existing environment variables
        
        Returns:
            True if .env file was loaded, False otherwise
        """
        if dotenv_path:
            path = Path(dotenv_path) if isinstance(dotenv_path,str) else dotenv_path
            self._env_file_path = path
            if not path.exists():
                logger.warning(f".env file not found at {path}, using platform environment variables")
                return False
        elif not os.path.exists(".env"):
            logger.info("No .env file found, using platform environment variables")
            return False

        return dotenv.load_dotenv(dotenv_path=dotenv_path,override=override)

    def _read_env_file(self) -> Dict[str, str]:
        """
        Read the current .env file and return its contents as a dictionary
        
        Returns:
            Dict[str, str]: Dictionary of environment variables from .env file
        """
        env_vars = {}
        if not self._env_file_path.exists():
            return env_vars
            
        try:
            with open(self._env_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Match KEY=VALUE pattern (handles quoted values too)
                    match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$', line)
                    if match:
                        key, value = match.groups()
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        env_vars[key] = value
                    else:
                        logger.warning(f"Invalid line format in .env file at line {line_num}: {line}")
        except Exception as e:
            logger.error(f"Error reading .env file: {e}")
            
        return env_vars

    def _write_env_file(self, env_vars: Dict[str, str]) -> None:
        """
        Write environment variables to the .env file
        
        Args:
            env_vars: Dictionary of environment variables to write
        """
        try:
            # Create directory if it doesn't exist
            self._env_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self._env_file_path, 'w', encoding='utf-8') as f:
                # Write header comment
                f.write("# Environment Variables\n")
                f.write("# Generated by JOLT Transformation Engine\n")
                f.write("# Sensitive values are encrypted\n\n")
                
                # Sort keys for consistent output
                for key in sorted(env_vars.keys()):
                    value = env_vars[key]
                    # Quote values that contain spaces or special characters
                    if ' ' in value or any(char in value for char in ['$', '`', '"', "'"]):
                        f.write(f'{key}="{value}"\n')
                    else:
                        f.write(f'{key}={value}\n')
                        
            logger.info(f"Updated .env file at {self._env_file_path}")
        except Exception as e:
            logger.error(f"Error writing .env file: {e}")
            raise EnvironmentVariableError(f"Failed to write .env file: {e}")

    def _update_env_file(self, key: str, value: str) -> None:
        """
        Update a single environment variable in the .env file
        
        Args:
            key: Environment variable name
            value: Environment variable value (already encrypted if sensitive)
        """
        env_vars = self._read_env_file()
        env_vars[key] = value
        self._write_env_file(env_vars)

    def encrypt_value(self,value:str) -> str:
        """Encrypt a sensitive value"""
        if not self._encryption_key:
            self._init_encryption()
        
        f = Fernet(self._encryption_key)
        return f.encrypt(value.encode()).decode()
    
    def decrypt_value(self,encrypted_value:str) -> str:
        """Decrypt sensitive data

        Args:
            encrypted_value (str): encrypted value which is wanted to decrypt

        Returns:
            str: decrypted value
        """
        if not self._encryption_key:
            self._init_encryption()
        
        try:
            f = Fernet(self._encryption_key)
            return f.decrypt(encrypted_value.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            raise EnvironmentVariableError("Failed to decrypt sensitive data")
    
    @lru_cache(maxsize=128)
    def get_env(self, key : str , default: Any = None, is_sensitive: bool = False ) -> Any:
        """Get an environment variable with caching

        Args:
            key (str): Environment variable name
            default (Any, optional): Default value if environment variables is not set
            is_sensitive (bool, optional): Whether this is a sensitive value (like API key)

        Returns:
            Any: The environment variable value or default
        """
        if key in self._cache:
            value = self._cache[key]
            if is_sensitive and value and isinstance(value,str) and value.startswith("encrypted:"):
                return self.decrypt_value(value[10:])
            return value
        
        value = os.environ.get(key, default)
        
        if value is not None:
            if is_sensitive and not str(value).startswith("encrypted:"):
                encrypted = f"encrypted:{self.encrypt_value(str(value))}"
                self._cache[key] = encrypted
            else:
                self._cache[key] = value
        
        if is_sensitive and value is not None and isinstance(value,str) and value.startswith("encrypted:"):
            return self.decrypt_value(value[10:])
        
        return value
    
    def set_env(self, key : str, value : Any, is_sensitive : bool = False, save_to_file: bool = True) -> None:
        """
        Set an environment variable

        Args:
            key(str) : Environment variable name
            value(Any): Value to set
            is_sensitive: Whether this is a sensitive value
            save_to_file: Whether to save the value to .env file (default: True)
        """        
        if is_sensitive:
            # Encrypt the value
            encrypted_value = f"encrypted:{self.encrypt_value(str(value))}"
            self._cache[key] = encrypted_value
            
            # Save encrypted value to .env file
            if save_to_file:
                self._update_env_file(key, encrypted_value)
                
            # Also set in current environment
            os.environ[key] = encrypted_value
            
            logger.info(f"Set encrypted environment variable {key}")
        else:
            # Store plain value
            str_value = str(value)
            self._cache[key] = str_value
            
            # Save to .env file
            if save_to_file:
                self._update_env_file(key, str_value)
                
            # Also set in current environment
            os.environ[key] = str_value
            
            logger.info(f"Set environment variable {key}")

    def remove_env(self, key: str, remove_from_file: bool = True) -> None:
        """
        Remove an environment variable
        
        Args:
            key: Environment variable name
            remove_from_file: Whether to remove from .env file (default: True)
        """        
        # Remove from cache
        if key in self._cache:
            del self._cache[key]
            
        # Remove from current environment
        if key in os.environ:
            del os.environ[key]
            
        # Remove from .env file
        if remove_from_file:
            env_vars = self._read_env_file()
            if key in env_vars:
                del env_vars[key]
                self._write_env_file(env_vars)
                logger.info(f"Removed environment variable {key} from .env file")

    def list_env_vars(self, include_sensitive: bool = False, pattern: str = None) -> Dict[str, str]:
        """
        List environment variables
        
        Args:
            include_sensitive: Whether to include (encrypted) sensitive values
            pattern: Optional pattern to filter variable names (e.g., "API_KEY" to find all API keys)
            
        Returns:
            Dict[str, str]: Dictionary of environment variables
        """
        env_vars = {}
        
        for key, value in os.environ.items():
            # Apply pattern filter if specified
            if pattern and pattern.upper() not in key.upper():
                continue
                
            if not include_sensitive and isinstance(value, str) and value.startswith("encrypted:"):
                env_vars[key] = "[ENCRYPTED]"
            else:
                env_vars[key] = value
                    
        return env_vars

    def backup_env_file(self, backup_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Create a backup of the current .env file
        
        Args:
            backup_path: Path for backup file (default: .env.backup)
            
        Returns:
            Path: Path to the backup file
        """
        if backup_path is None:
            backup_path = self._env_file_path.with_suffix('.env.backup')
        else:
            backup_path = Path(backup_path)
            
        if self._env_file_path.exists():
            import shutil
            shutil.copy2(self._env_file_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
        else:
            logger.warning("No .env file exists to backup")
            
        return backup_path

    def validate_required(self,keys:list[str]) -> None:
        """Validate that required environment variables are set
        Args:
            keys (list[str]): List of required environment variable names
        
        Raises: EnvironmentVariableError: If any required variable is missing
        """
        missing = []

        for key in keys:
            if key not in os.environ:
                missing.append(key)
                    
        if missing:
            raise EnvironmentVariableError(
                f"Missing required environment variables: {','.join(missing)}"
            )
            
    def load_config(self) -> Dict[str,Any]:
        """
        Load configuration from environment variables.
        Uses the ENV_MAPPING to find environment variables.

        Returns:
            Dict[str,Any]: Dictionary of configuration values
        """
        config = {}

        for env_var, config_path in self.ENV_MAPPING.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                self._set_nested_value(config,config_path,value)

        return config
    
    def clear_cache(self) -> None:
        """Clear the environment variable cache."""
        self._cache.clear()
        self.get_env.cache_clear()


class EnvironmentService:
    """
    Service for enhanced environment variable management.

    This service integrates with the existing config system from app_config.py
    while providing additional functionality for handling environment variables
    in both development and production environments.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EnvironmentService,cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._env_source = EnhancedEnvConfigSource()
        config_manager = ConfigManager()

        for i,source in enumerate(config_manager.sources):
            if isinstance(source,EnvConfigSource):
                config_manager.sources[i] = self._env_source
                break
        else:
            config_manager.add_source(self._env_source)

        self._initialized = True

        if os.path.exists(".env") : 
            logger.info("Environment service initialized in development mode (with .env file)")
        else:
            logger.info("Environment service initialized in production mode (using platform environment variables)")
        
    def get(self,key: str, default: Any = None, is_sensitive: bool = False) -> Any:
        """
        Get environment variables

        Args:
            key (str): variable key 
            default (Any, optional): Default value if exists. Defaults to None.
            is_sensitive (bool, optional): Whether it is sensitive. Defaults to False.

        Returns:
            Any: Environment variable
        """
        return self._env_source.get_env(key,default,is_sensitive)

    def set(self,key: str, value: Any, is_sensitive: bool= False, save_to_file: bool = True) -> None:
        """
        Set environment variable

        Args:
            key (str): Environment variable name
            value (Any): Value to set for key
            is_sensitive (bool, optional): Whether to contains sensitive data. Defaults to False.
            save_to_file (bool, optional): Whether to save to .env file. Defaults to True.
        """
        self._env_source.set_env(key,value,is_sensitive, save_to_file)

    def remove(self, key: str, remove_from_file: bool = True) -> None:
        """
        Remove an environment variable
        
        Args:
            key (str): Environment variable name
            remove_from_file (bool): Whether to remove from .env file
        """
        self._env_source.remove_env(key, remove_from_file)

    def list_variables(self, include_sensitive: bool = False, pattern: str = None) -> Dict[str, str]:
        """
        List environment variables
        
        Args:
            include_sensitive: Whether to show encrypted sensitive values
            pattern: Optional pattern to filter variable names
            
        Returns:
            Dict[str, str]: Dictionary of environment variables
        """
        return self._env_source.list_env_vars(include_sensitive, pattern)

    def backup_env_file(self, backup_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Create a backup of the .env file
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path: Path to the backup file
        """
        return self._env_source.backup_env_file(backup_path)

    def load_dotenv(self,dotenv_path: Union[str,Path,None] = None, override: bool = False) -> bool:
        """Load environment variables from .env file

        Args:
            dotenv_path (Union[str,Path,None], optional): '.env' file path. Defaults to None.
            override (bool, optional): Whether to override .env file . Defaults to False.

        Returns:
            bool: True, if loaded. False, if not loaded
        """
        return self._env_source._load_dotenv(dotenv_path,override)

    def validate_required(self,keys : list[str]) -> None:
        """
        Validate that required environment variables are set

        Args:
            keys (list[str]): List of environment variable names to validate
        """
        self._env_source.validate_required(keys)
        
    def reload_config(self) -> None:
        """
        Reload the configuration from environment variables.
        This is useful after setting new environment variables.
        """
        config_manager = ConfigManager()
        config_manager._load_config()

    def get_deployment_info(self) -> Dict[str,str]:
        """
        Get information about the deployment environment

        Returns:
            Dict[str,str]: Dictionary with deployment information 
        """
        config = get_config()

        return {
            "environment" : config.environment,
            "app_name" : config.app_name
        }


# Singleton instance
env_service = EnvironmentService()

def get_env(key: str, default: Any = None, is_sensitive: bool = False) -> Any:
    """Get an environment variable."""
    return env_service.get(key, default, is_sensitive)


def set_env(key: str, value: Any, is_sensitive: bool = False, save_to_file: bool = True) -> None:
    """
    Set an environment variable.
    
    Args:
        key: Environment variable name
        value: Value to set
        is_sensitive: Whether this is sensitive data (will be encrypted)
        save_to_file: Whether to save to .env file (default: True)
    """
    env_service.set(key, value, is_sensitive, save_to_file)
    env_service.reload_config()


def remove_env(key: str, remove_from_file: bool = True) -> None:
    """Remove an environment variable."""
    env_service.remove(key, remove_from_file)
    env_service.reload_config()


def list_env_vars(include_sensitive: bool = False, pattern: str = None) -> Dict[str, str]:
    """
    List environment variables.
    
    Args:
        include_sensitive: Whether to show encrypted values
        pattern: Optional pattern to filter variable names
    """
    return env_service.list_variables(include_sensitive, pattern)


def backup_env_file(backup_path: Optional[Union[str, Path]] = None) -> Path:
    """Create a backup of the .env file."""
    return env_service.backup_env_file(backup_path)


def load_env_file(path: Union[str, Path, None] = None, override: bool = False) -> bool:
    """
    Load environment variables from .env file if available.
    
    This is typically used in development. In production, environment
    variables are expected to be set on the platform directly.
    """
    return env_service.load_dotenv(path, override)


def validate_required_env(keys: list[str]) -> None:
    """Validate that required environment variables are set."""
    env_service.validate_required(keys)


def get_config_from_env() -> Dict[str, Any]:
    """
    Get configuration from environment variables.
    
    This uses the app_config.py configuration system.
    
    Returns:
        The application configuration
    """
    return get_config()