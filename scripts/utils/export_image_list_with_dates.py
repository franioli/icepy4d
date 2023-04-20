"""
MIT License

Copyright (c) 2022 Francesco Ioli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import argparse
import sys
import logging

from pathlib import Path

from icepy4d.classes.images import ImageDS, Image


def parse_command_line():
    """
    parse_command_line Parse command line input

    Returns:
        Tuple[str, dict]: Tuple containing the path of the configuration file and a dictionary containing parameters to setup the logger
    """
    parser = argparse.ArgumentParser(
        description="""Read image exif info and store them as a csv file \
            Check -h or --help for options.
        Usage: ./main.py """
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path of to the configuration file",
    )
    parser.add_argument(
        "--log_folder",
        default="logs",
        type=str,
        help="Folder for storing logs (default: 'logs')",
    )
    parser.add_argument(
        "--log_name",
        default="log",
        type=str,
        help="Base name of the log file",
    )
    parser.add_argument(
        "--log_file_level",
        default="info",
        type=str,
        help="Set log level for logging to file \
            (possible options are: 'debug', 'info', \
            'warning', 'error', 'critical')",
    )
    parser.add_argument(
        "--log_console_level",
        default="info",
        type=str,
        help="Set log level for logging to stdout \
            (possible options are: 'debug', 'info', \
            'warning', 'error', 'critical')",
    )
    args = parser.parse_args()

    if not len(sys.argv) > 1:
        raise ValueError(
            "Not enough input arguments. Specify at least the configuration file. Use --help (or -h) for help."
        )

    cfg_file = Path(args.config)
    log_cfg = {
        "log_folder": args.log_folder,
        "log_name": args.log_name,
        "log_file_level": args.log_file_level,
        "log_console_level": args.log_console_level,
    }

    return cfg_file, log_cfg


if __name__ == "__main__":

    ROOT_DIR = "/mnt/labmgf/Belvedere/belvedereStereo/img"  # "data/img"
    IM_DIR_LIST = ["pi2_jpg"]
    IM_EXT = None

    logging.info("")

    ROOT_DIR = Path(ROOT_DIR)
    for dir in IM_DIR_LIST:
        folder = ROOT_DIR / dir
        images = ImageDS(folder, ext=IM_EXT)
        images.write_exif_to_csv(ROOT_DIR / f"image_list_{dir}.csv")

    print("Done")
