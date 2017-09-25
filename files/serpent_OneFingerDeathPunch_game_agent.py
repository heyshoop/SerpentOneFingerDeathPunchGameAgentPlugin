import offshoot
import pyautogui
import re

from serpent.game_agent import GameAgent
from serpent.input_controller import MouseButton
from serpent.sprite_locator import SpriteLocator
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier


class SerpentOneFingerDeathPunchGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.analytics_client = None

        self.allPunches = []

    def setup_play(self):
        plugin_path = offshoot.config["file_paths"]["plugins"]

        context_classifier_path = f"{plugin_path}/SerpentOneFingerDeathPunchGameAgentPlugin/files/ml_models/context_classifier.model"

        context_classifier = CNNInceptionV3ContextClassifier(input_shape=(360, 640, 3))

        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)

        self.machine_learning_models["context_classifier"] = context_classifier

    def handle_play(self, game_frame):
        context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)

        print(context)

        if context is None:
            print("There's nothing there... Waiting...")
            return

        if context == "splash_screen":
            print("Boring part, just click on \"Play\"")
            pyautogui.click(button="left", x=640, y=460)
            pyautogui.mouseDown()
            pyautogui.mouseUp()

        if context == "main_menu":
            print("Boring part 2, just click on \"Play\"... again")
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="MAIN_MENU_CLICK_MOUSE_PLAY"
            )
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            pyautogui.mouseDown()
            pyautogui.mouseUp()

        if context == "mode_menu":
            print("What to choose ? Oh ! Survival !")
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="MODE_MENU_SURVIVAL"
            )
            pyautogui.mouseDown()
            pyautogui.mouseUp()

        if context == "survival_menu":
            print("I need to start a game :D")

            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="SURVIVAL_MENU_BUTTON_TOP"
            )
            # Temporary fix for click
            pyautogui.mouseDown()
            pyautogui.mouseUp()

        if context == "survival_pre_game":
            print("I don't need skills. Let's play !")

            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="SURVIVAL_PRE_GAME_START_BUTTON"
            )
            pyautogui.mouseDown()
            pyautogui.mouseUp()

        if context == "game":
            # print("\033c")
            print()
            print("I'M PLAYING !")

            sprite_leftone = self.game.sprites["SPRITE_LEFT-1"]
            sprite_lefttwo = self.game.sprites["SPRITE_LEFT-2"]
            sprite_rightone = self.game.sprites["SPRITE_RIGHT-1"]
            sprite_righttwo = self.game.sprites["SPRITE_RIGHT-2"]

            sprites_left = [sprite_leftone, sprite_lefttwo]
            sprites_right = [sprite_rightone, sprite_righttwo]

            sprite_locator = SpriteLocator()

            print()
            try:
                for sprites_list in [sprites_left, sprites_right]:
                    for sprite_to_check in sprites_list:
                        print("Checking for:", sprite_to_check.name)
                        check_sprite = sprite_locator.locate(
                            sprite=sprite_to_check,
                            game_frame=game_frame
                        )

                        if check_sprite:
                            print("Got the sprite on screen !")
                            print("Checking side...")
                            if sprite_to_check in sprites_left:
                                btnClick = "left"
                            else:
                                btnClick = "right"

                            print("Sprite on", btnClick, "side.")
                            match_two = re.search("TWO", sprite_to_check.name)
                            print("Checking if 1 or 2 (for now)...")
                            if match_two:
                                nbClick = 2
                            else:
                                nbClick = 1

                            print(nbClick, "hit(s) to do")

                            for i in range(0, nbClick):
                                print("Hit", i + 1)
                                pyautogui.mouseDown(button=btnClick)
                                pyautogui.mouseUp(button=btnClick)
            except TypeError as error:
                print("CheckError:", error)

        if context == "game_paused":
            print("I'M PAUSING !")

        if context == "game_end_score":
            print("I'M... dead.")
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="GAME_OVER_SCORE_BUTTON"
            )
            pyautogui.mouseDown()
            pyautogui.mouseUp()

        if context == "game_end_highscore":
            print("I'M... dead. And i have an highscore")
