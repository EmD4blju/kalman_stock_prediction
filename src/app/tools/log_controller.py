from logging.config import dictConfig
import json
from pathlib import Path



class LogController():
    """Module to initialize all necessary logging tools.
    """

    def __init__(self, config_path: Path = Path('src','app','config','logging_config.json'), dirs:Path = Path('logs')):
        if not config_path.exists():
            raise FileNotFoundError(f'Config path: {config_path} does not exist')
        
        if not dirs.exists():
            raise FileNotFoundError(f'Log directory path: {dirs} does not exist')
        
        self._config_path = config_path
        self._dirs = dirs
        
    def start(self):
        """Starts the LogController by initializing log directories, configuring loggers and handlers
        from the provided configuration, and starting a QueueListener to handle log records.
        """
        self._init_log_dirs()
        config = self._load_loggers_configuration() 
        dictConfig(config)
        
        
    def _init_log_dirs(self) -> None:
        """
        Creates directories with given paths.
        
        Args:
            dirs (Iterable): Iterable of required directories.
        """
        self._dirs.mkdir(parents=True, exist_ok=True)
    
    

    def _load_loggers_configuration(self) -> dict:
        """Loads yaml file and configurates logging.
        
        Args:
            conf_path (Path): Path to .yaml configuration file.

        Returns:
            dict: Loaded configuration dictionary.
        """
        with open(self._config_path, 'r') as file:
            return json.load(file)
    
