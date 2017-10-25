import os
import time
import offshoot
import pyautogui
import collections
import numpy as np

from skimage.color import rgb2gray

import serpent.utilities
from serpent.game_agent import GameAgent
from serpent.sprite_locator import SpriteLocator
from serpent.input_controller import MouseButton, KeyboardKey
from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier

plugin_path = offshoot.config["file_paths"]["plugins"]

# Constants used for zoom level
ZOOM_MAIN = "main"
ZOOM_BRAWLER = "brawler"
ZOOM_KILL_MOVE = "kill_move"


class SerpentOneFingerDeathPunchGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handlers["PLAY_BOT"] = self.handle_play_bot

        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_setups["PLAY_BOT"] = self.setup_play

        self.analytics_client = None

        self.sprite_locator = SpriteLocator()

        self.username_entered = False

    def setup_play(self):
        self.reset_game_state()
        self.setup_play_bot()

        # self.my_ddqn = DDQN(
        #     input_shape=
        # )

    def setup_play_bot(self):
        context_classifier_path = f"{plugin_path}/SerpentOneFingerDeathPunchGameAgentPlugin/files/ml_models/context_classifier.model"
        context_classifier = CNNInceptionV3ContextClassifier(input_shape=(360, 640, 3))
        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)
        self.machine_learning_models["context_classifier"] = context_classifier

    def reset_game_state(self):
        self.game_state = {
            "health": collections.deque(np.full((8,), 10), maxlen=8),
            "nb_ennemies_hit": 0,
            "zoom_level": ZOOM_MAIN,
            "bonus_mode": False,
            "bonus_hits": 4,
            "nb_miss": 0,
            "miss_failsafe": 2
        }

    def handle_play(self, game_frame):
        # TODO: find a way to check enemies without the left&right click sprite
        serpent.utilities.clear_terminal()
        print("DATA WILL BE HERE...")

        # This order is important
        self.update_zoom_level(game_frame)
        self.update_health_counter(game_frame)
        # These can be in any order
        self.update_miss_counter(game_frame)
        self.update_bonus_mode_and_hits(game_frame)

    def handle_play_bot(self, game_frame):
        context = self.machine_learning_models["context_classifier"].predict(game_frame.frame)
        print(context)
        if context is None:
            print("There's nothing there... Waiting...")
            return

        self.do_splash_screen_action(context)
        self.do_main_menu_actions(context)
        self.do_mode_menu_action(context)
        self.do_survival_menu_action(context)
        self.do_survival_pre_game_action(context)
        self.do_game_paused_action(context)
        self.do_game_end_highscore_action(context)
        self.do_game_end_score_action(context)

        if context == "ofdp_game":
            serpent.utilities.clear_terminal()
            print("I'M PLAYING !")
            # TODO: add data about life points, nb of killed enemies, etc..

            # This order is important
            self.update_zoom_level(game_frame)
            self.update_health_counter(game_frame)
            # These can be in any order
            self.update_miss_counter(game_frame)
            self.update_bonus_mode_and_hits(game_frame)

            # TODO: find a new way to get enemies
            pixel_left = {
                "image_data": game_frame.frame[350:351, 600:601],
                "key": KeyboardKey.KEY_LEFT,
                "msbtn": MouseButton.LEFT
            }
            pixel_right = {
                "image_data": game_frame.frame[350:351, 672:673],
                "key": KeyboardKey.KEY_RIGHT,
                "msbtn": MouseButton.RIGHT
            }

            for pixel_to_check in [pixel_left, pixel_right]:
                pixel = pixel_to_check["image_data"][0, 0]
                if sum(pixel) == 72:
                    # self.input_controller.tap_key(key=pixel_to_check["key"])
                    self.input_controller.click(button=pixel_to_check["msbtn"])

            self.display_game_data()
            print("")

    def display_game_data(self):
        print(
            "Health:",
            self.game_state["health"][0],
            "| LAST HIT!!!" if self.game_state["health"][0] == 1 else "",
            "| Ded" if self.game_state["health"][0] == 0 else ""
        )
        print("NB miss:", self.game_state["nb_miss"])
        print("Zoom level:", self.game_state["zoom_level"])
        if self.game_state["bonus_mode"]:
            print(
                "BONUS ROUND - Bonus round end in",
                self.game_state["bonus_hits"],
                "hits"
            )

    def update_health_counter(self, game_frame):
        zoom_level = self.game_state["zoom_level"]

        if zoom_level == ZOOM_MAIN:
            first_x = 553
            first_y = 554
            last_health_x = 569
            last_health_y = 570
        elif zoom_level == ZOOM_BRAWLER:
            first_x = 606
            first_y = 607
            last_health_x = 622
            last_health_y = 623
        elif zoom_level == ZOOM_KILL_MOVE:
            # Can't get any new modification on health here
            # return
            first_x = 553
            first_y = 554
            last_health_x = 569
            last_health_y = 570

        current_health = 0

        for nb_health in range(0, 9):
            region_health = game_frame.frame[first_x:first_y, 786 - (35 * nb_health):787 - (35 * nb_health)]
            if region_health[0, 0, 0] > 200:
                current_health += 1

        health_last = game_frame.frame[last_health_x:last_health_y, 475:476]
        # "REGION": (569, 475, 570, 476)

        if health_last[0, 0, 0] > 200:
            current_health += 1

        if -1 <= self.game_state["health"][0] - current_health <= 1:
            self.game_state["health"].appendleft(current_health)

    def update_miss_counter(self, game_frame):
        miss_region = rgb2gray(game_frame.frame[357:411, 570:710])
        self.game_state["miss_failsafe"] -= 1
        # print(sum(sum(miss_region)))
        if 3400 < sum(sum(miss_region)) < 3500 and self.game_state["miss_failsafe"] < 0 and self.game_state["zoom_level"] is ZOOM_MAIN:
            self.game_state["nb_miss"] += 1
            self.game_state["miss_failsafe"] = 2

    def update_bonus_mode_and_hits(self, game_frame):
        for nb_hits in range(0, 4):
            region_hit = game_frame.frame[618:619, 714 - (50 * nb_hits):715 - (50 * nb_hits)]
            if sum(region_hit[0, 0]) == 306:
                self.game_state["bonus_hits"] += 1

        if self.game_state["bonus_hits"] > 0:
            self.game_state["bonus_mode"] = True
        self.game_state["bonus_mode"] = False

    # Check the zoom on game screen. "main" is the normal game, "brawler" when
    # a brawler is fighting, "kill_move" when the character does a kill move
    def update_zoom_level(self, game_frame):
        check_zoom_mode = game_frame.frame[563:564, 639:640]
        sum_pixels = sum(check_zoom_mode[0, 0])
        if sum_pixels > 300:
            self.game_state["zoom_level"] = ZOOM_MAIN
        elif sum_pixels == 300:
            self.game_state["zoom_level"] = ZOOM_BRAWLER
        elif sum_pixels < 300:
            self.game_state["zoom_level"] = ZOOM_KILL_MOVE

    def do_splash_screen_action(self, context):
        if context == "ofdp_splash_screen":
            print("Boring part, just click on \"Play\"")
            self.input_controller.move(x=650, y=460)
            self.input_controller.click()
            time.sleep(5)

    def do_main_menu_actions(self, context):
        if context == "ofdp_main_menu":
            print("Boring part 2, just click on \"Play\"... again")
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="MAIN_MENU_CLICK_MOUSE_PLAY"
            )
            time.sleep(0.5)
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="MAIN_MENU_CLICK_MOUSE_PLAY"
            )
            time.sleep(1)

    def do_mode_menu_action(self, context, game_mode="MODE_MENU_SURVIVAL"):
        if context == "ofdp_mode_menu":
            print("What to choose ? Oh ! Survival !")
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region=game_mode
            )
            time.sleep(1)

    def do_survival_menu_action(self, context, game_mode="SURVIVAL_MENU_BUTTON_TOP"):
        if context == "ofdp_survival_menu":
            print("I need to start a game :D")

            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region=game_mode
            )
            time.sleep(1)

    def do_survival_pre_game_action(self, context):
        if context == "ofdp_survival_pre_game":
            print("I don't need skills. Let's play !")

            self.reset_game_state()
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="SURVIVAL_PRE_GAME_START_BUTTON"
            )
            time.sleep(1)


    def do_game_paused_action(self, context):
        # TODO: add click for quitting or resume.
        if context == "ofdp_game_paused":
            print("I'M PAUSING !")
            time.sleep(1)

    def do_game_end_highscore_action(self, context):
        if context == "ofdp_game_end_highscore":
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

    def do_game_end_score_action(self, context):
        if context == "ofdp_game_end_score":
            # TODO: check score + nb of enemies killed.
            print("I'M... dead.")
            print("Waiting for button...")
            time.sleep(3)
            self.input_controller.click_screen_region(
                button=MouseButton.LEFT,
                screen_region="GAME_OVER_SCORE_BUTTON"
            )
            time.sleep(1)
