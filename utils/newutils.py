import sys
import datetime

def get_local_time() -> str:
    """
    Get the current local time in a specific format.

    Returns:
        str: Current time formatted as "Month-Day-Year_Hour-Minute-Second".
    """
    return datetime.datetime.now().strftime("%b-%d-%Y_%H-%M-%S")

def get_command_line_args_str() -> str:
    """
    Get the command line arguments as a single string, with '/' replaced by '|'.

    Returns:
        str: The command line arguments.
    """
    return '_'.join(sys.argv).replace('/', '|')

def get_file_name(config: dict, suffix: str = ''):
    import hashlib
    config_str = "".join([str(value) for key, value in config.items() if key != 'accelerator'])
    md5 = hashlib.md5(config_str.encode(encoding="utf-8")).hexdigest()[:6]
    command_line_args = get_command_line_args_str()
    logfilename = "{}-{}-{}-{}{}".format(
        config["run_id"], command_line_args[:50], get_local_time(), md5, suffix
    )
    return logfilename