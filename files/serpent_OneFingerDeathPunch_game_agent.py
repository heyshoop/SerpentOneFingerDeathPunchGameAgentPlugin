import offshoot
import pyautogui
# import re
import time
import numpy as np

from serpent.sprite import Sprite
from serpent.game_agent import GameAgent
from serpent.sprite_locator import SpriteLocator
from serpent.input_controller import MouseButton
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier

plugin_path = offshoot.config["file_paths"]["plugins"]


class SerpentOneFingerDeathPunchGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.analytics_client = None

        self.sprite_locator = SpriteLocator()

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
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            time.sleep(5)

        if context == "main_menu":
            print("Boring part 2, just click on \"Play\"... again")
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="MAIN_MENU_CLICK_MOUSE_PLAY"
            )
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            time.sleep(0.5)
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            time.sleep(1)

        if context == "mode_menu":
            print("What to choose ? Oh ! Survival !")
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="MODE_MENU_SURVIVAL"
            )
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            time.sleep(1)

        if context == "survival_menu":
            print("I need to start a game :D")

            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="SURVIVAL_MENU_BUTTON_TOP"
            )
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            time.sleep(1)

        if context == "survival_pre_game":
            print("I don't need skills. Let's play !")

            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="SURVIVAL_PRE_GAME_START_BUTTON"
            )
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            time.sleep(1)

        if context == "game":
            print("\033c")
            print()
            print("I'M PLAYING !")
            # TODO: add data about life points, nb of killed enemies, etc..

            # Cropping game_frame into 2 zones (the click ones)
            frame_right = game_frame.frame[350:414, 672:744]
            frame_left = game_frame.frame[350:414, 536:608]

            # These are only for checking the sprite name we found
            sprite_leftpunch = self.game.sprites["SPRITE_LEFT-PUNCH"]
            sprite_rightpunch = self.game.sprites["SPRITE_RIGHT-PUNCH"]

            for frame_to_check in [frame_left, frame_right]:
                sprite_frame = Sprite(
                    "QUERY",
                    image_data=frame_to_check[..., np.newaxis]
                )
                # I've done some test and the CONSTELLATION_OF_PIXELS is far
                # more better FOR MY CASE ONLY !
                sprite_name = self.sprite_identifier.identify(
                    sprite_frame, mode="CONSTELLATION_OF_PIXELS"
                )

                # DEBUG
                print(sprite_name)

                if sprite_name != "UNKNOWN":
                    if sprite_name == sprite_leftpunch.name:
                        key = "left"
                    elif sprite_name == sprite_rightpunch.name:
                        key = "right"

                    # We can use the left & right mouse buttons too.
                    self.input_controller.tap_key(key)

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
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            time.sleep(1)

        if context == "game_end_highscore":
            print("I'M... dead. And i have an highscore")
