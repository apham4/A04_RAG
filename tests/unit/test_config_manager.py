import pytest
import json
from pathlib import Path
from classes.config_manager import ConfigManager

# A pytest fixture to create a temporary config file
@pytest.fixture
def temp_config_file(tmp_path):
    config_data = {"llm_model_name": "test-model", "log_level": "DEBUG"}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))
    return config_file

# A pytest fixture to create an empty config file
@pytest.fixture
def temp_empty_config_file(tmp_path):
    config_data = {}
    config_file = tmp_path / "empty_config.json"
    config_file.write_text(json.dumps(config_data))
    return config_file

def test_config_manager_loads_successfully(temp_config_file):
    """
    Tests that the ConfigManager can successfully load a valid JSON file.
    """
    # WHEN the ConfigManager is initialized with the path to the temp file
    config_manager = ConfigManager(temp_config_file)
    
    # THEN it should correctly retrieve the values
    assert config_manager.get("llm_model_name") == "test-model"
    assert config_manager.get("log_level") == "DEBUG"

def test_config_manager_get_with_default(temp_empty_config_file):
    """
    Tests that the get() method returns a default value for a missing key.
    """
    # GIVEN a config manager with no file
    config_manager = ConfigManager(temp_empty_config_file)
    
    # WHEN we get a key that doesn't exist, with a default
    result = config_manager.get("a_missing_key", "default_value")
    
    # THEN the default value should be returned
    assert result == "default_value"

def test_config_manager_handles_missing_file():
    """
    Tests that the application exits if the config file is not found.
    """
    # GIVEN a path to a file that does not exist
    non_existent_file = Path("surely/this/does/not/exist.json")
    
    # WHEN ConfigManager is initialized with this path
    # THEN it should raise a SystemExit (because of the exit(1) call)
    with pytest.raises(SystemExit):
        ConfigManager(non_existent_file)