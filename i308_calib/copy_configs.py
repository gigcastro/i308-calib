import argparse
import os
import shutil
#


def get_config_dir():
    try:
        import pkg_resources
        config_dir = pkg_resources.resource_filename("i308_calib", "cfg")
    except ImportError:
        from importlib.resources import files
        config_dir = str(files("i308_calib").joinpath("cfg"))

    return config_dir
def copy_configs(args):
    """Copy bundled config files to the current directory."""

    config_dir = get_config_dir()
    dest_dir = args.target  # os.getcwd()

    if dest_dir != "" and not os.path.isdir(dest_dir):
        print("Creating destination directory {}".format(dest_dir))
        os.makedirs(dest_dir)

    for filename in os.listdir(config_dir):
        src = os.path.join(config_dir, filename)
        dest = os.path.join(dest_dir, filename)

        if not os.path.exists(dest):
            shutil.copy(src, dest)
            print(f"Copied {filename} to {dest_dir}")
        else:
            print(f"Skipping {filename}, already exists.")


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="copies base configuration files",
        formatter_class=argparse.RawTextHelpFormatter
    )

    arg_parser.add_argument(
        "-t", "--target",
        default="cfg",
        help=f"target configuration directory "
    )

    args = arg_parser.parse_args()

    return args


def run():
    args = parse_args()
    copy_configs(args)


if __name__ == "__main__":
    run()
