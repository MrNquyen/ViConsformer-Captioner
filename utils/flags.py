import argparse
class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_core_args()

    def get_parser(self):
        return self.parser.parse_args()

    def add_core_args(self):
        # TODO: Update default values
        self.parser = argparse.ArgumentParser(description='DEVICE Captioner')

        self.parser.add_argument(
            "--config", type=str, default=None, required=False, help="config yaml file"
        )

        self.parser.add_argument(
            "--run_type", type=str, default="train", help="Which run type you prefer ['train', 'test']"
        )

        self.parser.add_argument(
            "--save_dir", type=str, default=None, help="Where to save the model"
        )

        self.parser.add_argument(
            "--resume_file", type=str, default=None, help="Model file to resume training or testing"
        )

        self.parser.add_argument(
            "--device", type=str, default="cuda:0", help="Set device"
        )