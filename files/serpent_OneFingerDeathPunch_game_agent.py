import offshoot
import pyautogui
# import re
import time
import numpy as np

from serpent.sprite import Sprite
from serpent.game_agent import GameAgent
from serpent.sprite_locator import SpriteLocator
from serpent.input_controller import MouseButton, KeyboardKey
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier

plugin_path = offshoot.config["file_paths"]["plugins"]


class SerpentOneFingerDeathPunchGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.analytics_client = None

        self.sprite_locator = SpriteLocator()

        self.play_mode = "bot"
        self.username_entered = False

    def setup_play(self):

        context_classifier_path = f"{plugin_path}/SerpentOneFingerDeathPunchGameAgentPlugin/files/ml_models/context_classifier.model"

        context_classifier = CNNInceptionV3ContextClassifier(input_shape=(360, 640, 3))

        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)

        self.machine_learning_models["context_classifier"] = context_classifier

    def handle_play(self, game_frame):
        context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)

        print(context)

        # You will see a lot of pyautogui.mouseX(). The click_screen_region
        # doesn't work on Windows for now so this is my temporary fix.

        if context is None:
            print("There's nothing there... Waiting...")
            return

        if context == "splash_screen":
            print("Boring part, just click on \"Play\"")
            pyautogui.click(button="left", x=640, y=460)
            time.sleep(5)

        if context == "main_menu":
            print("Boring part 2, just click on \"Play\"... again")
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="MAIN_MENU_CLICK_MOUSE_PLAY"
            )
            time.sleep(0.5)
            time.sleep(1)

        if context == "mode_menu":
            print("What to choose ? Oh ! Survival !")
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="MODE_MENU_SURVIVAL"
            )
            time.sleep(1)

        if context == "survival_menu":
            print("I need to start a game :D")

            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="SURVIVAL_MENU_BUTTON_TOP"
            )
            time.sleep(1)

        if context == "survival_pre_game":
            print("I don't need skills. Let's play !")

            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="SURVIVAL_PRE_GAME_START_BUTTON"
            )
            time.sleep(1)

        if context == "game":
            print("\033c")
            print()
            print("I'M PLAYING !")

            if self.play_mode == "bot":
                self.handle_play_bot()

        if context == "game_paused":
            print("I'M PAUSING !")
            time.sleep(1)

        if context == "game_end_score":
            # TODO: check score + nb of enemies killed.
            print("I'M... dead.")
            print("Waiting for button...")
            time.sleep(3)
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="GAME_OVER_SCORE_BUTTON"
            )
            time.sleep(1)

        if context == "game_end_highscore":
            print("I'M... dead. And i have an highscore")

            if not self.username_entered:
                for letter in ["HS_LETTER_A", "HS_LETTER_I", "HS_LETTER_K", "HS_LETTER_L", "HS_LETTER_E", "HS_LETTER_M"]:
                    self.input_controller.click_screen_region(
                        button=MouseButton.LEFT,
                        screen_region=letter
                    )
                    pyautogui.mouseDown()
                    pyautogui.mouseUp()

                self.username_entered = True

            print("Done !")

            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="HS_OK"
            )

    def handle_play_bot(self):
        # TODO: add data about life points, nb of killed enemies, etc..

        # Cropping game_frame into 2 zones (the click ones)
        # frame_left = game_frame.frame[350:366, 536:608]
        # frame_right = game_frame.frame[350:366, 672:744]

        frame_left = {
            "image_data": game_frame.frame[350:358, 600:608],
            "key": KeyboardKey.KEY_LEFT,
            "msbtn": MouseButton.LEFT
        }
        frame_right = {
            "image_data": game_frame.frame[350:358, 672:680],
            "key": KeyboardKey.KEY_RIGHT,
            "msbtn": MouseButton.RIGHT
        }

        for frame_to_check in [frame_left, frame_right]:
            check_one = frame_to_check["image_data"][1, 1][0] < 100
            check_two = frame_to_check["image_data"][1, 1][1] > 135
            check_three = frame_to_check["image_data"][1, 1][2] > 150

            if check_one and check_two and check_three:
                print(frame_to_check["msbtn"])
                # print(self.input_controller.tap_key(frame_to_check["key"]))
                self.input_controller.click(button=frame_to_check["msbtn"])
                break
