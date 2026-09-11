import unittest

import main


class MainTests(unittest.TestCase):
    def test_cli_exposes_only_registration_and_recognition(self) -> None:
        parser = main.build_parser()

        self.assertEqual(parser.parse_args(["register"]).command, "register")
        self.assertEqual(parser.parse_args(["recognize"]).command, "recognize")
        command_action = next(
            action for action in parser._actions if action.dest == "command"
        )
        self.assertEqual(set(command_action.choices), {"register", "recognize"})


if __name__ == "__main__":
    unittest.main()
