import os
import time
import offshoot
import pyautogui

from serpent.game_agent import GameAgent
from serpent.sprite_locator import SpriteLocator
from serpent.input_controller import MouseButton, KeyboardKey
from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier

plugin_path = offshoot.config["file_paths"]["plugins"]


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
        self.setup_play_bot()

        input_mapping = {
            "LEFT": [MouseButton.LEFT, KeyboardKey.KEY_LEFT],
            "RIGHT": [MouseButton.RIGHT, KeyboardKey.KEY_RIGHT]
        }

        direction_action_space = KeyboardMouseActionSpace(
            default_keys=[None, "LEFT", "RIGHT"]
        )

        direction_model_file_path = "datasets/ofdp_direction_dqn_0_1_.hp5".replace("/", os.sep)
        self.dqn_direction = DDQN(
            model_file_path=direction_model_file_path if os.path.isfile(direction_model_file_path) else None,
            input_shape=(100, 100, 4),
            input_mapping=input_mapping,
            action_space=direction_action_space,
            replay_memory_size=5000,
            max_steps=1000000,
            observe_steps=1000,
            batch_size=32,
            model_learning_rate=1e-4,
            initial_epsilon=1,
            final_epsilon=0.01,
            override_epsilon=False
        )

    def setup_play_bot(self):
        context_classifier_path = f"{plugin_path}/SerpentOneFingerDeathPunchGameAgentPlugin/files/ml_models/context_classifier.model"
        context_classifier = CNNInceptionV3ContextClassifier(input_shape=(360, 640, 3))
        context_classifier.prepare_generators()
        context_classifier.load_classifier(context_classifier_path)
        self.machine_learning_models["context_classifier"] = context_classifier

        self.last_seen_health = 10

    def handle_play(self, game_frame):
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
        self.do_game_end_highscore(context)
        self.do_game_end_score_action(context)

        if context == "ofdp_game":
            print("\033c")
            print("DATA WILL BE HERE...")

            self.display_health(game_frame)

            #TODO: find a way to check enemies without the left&right click sprite

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
        self.do_game_end_highscore(context)
        self.do_game_end_score_action(context)

        if context == "ofdp_game":
            print("\033c")
            print()
            print("I'M PLAYING !")
            # TODO: add data about life points, nb of killed enemies, etc..

            self.display_health(game_frame)

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

            action = "Nothing"

            for frame_to_check in [frame_left, frame_right]:
                check_one = frame_to_check["image_data"][1, 1][0] < 100
                check_two = frame_to_check["image_data"][1, 1][1] > 135
                check_three = frame_to_check["image_data"][1, 1][2] > 150

                if check_one and check_two and check_three:
                    action = frame_to_check["msbtn"]
                    # print(self.input_controller.tap_key(frame_to_check["key"]))
                    self.input_controller.click(button=frame_to_check["msbtn"])
                    break

            print("Actions:", action)
            print("")

    def display_health(self, game_frame):
        if self.check_zoomed_to_level_one(game_frame):
            first_x = 553
            first_y = 554
            last_health_x = 569
            last_health_y = 570
        else:
            first_x = 606
            first_y = 607
            last_health_x = 622
            last_health_y = 623

        current_health = 0

        for nb_health in range(0, 9):
            region_health = game_frame.frame[first_x:first_y, 786 - (35 * nb_health):787 - (35 * nb_health)]
            if region_health[0, 0, 0] > 200:
                current_health += 1

        health_last = game_frame.frame[last_health_x:last_health_y, 475:476]
        #"REGION": (569, 475, 570, 476)

        if health_last[0, 0, 0] > 200:
            current_health += 1

        if -1 <= self.last_seen_health - current_health <= 1:
            self.last_seen_health = current_health

        print(
            "Health:",
            self.last_seen_health,
            "LAST HIT!!!" if self.last_seen_health == 1 else ""
        )


    # "Zoomed level one" is the brawler mode.
    # "Zoomed level two" is the killmove zoom.
    def check_zoomed_to_level_one(self, game_frame):
        #"REGION": (579, 961, 580, 962)
        # THIS IS ONLY THE FIRST ZOOM. THE SECOND ONE NEED TO BE TESTED
        check_zoomed_mode = game_frame.frame[579:580, 961:962]
        if sum(check_zoomed_mode[0, 0]) < 150:
            return True
        return False

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

    def do_game_end_highscore(self, context):
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
