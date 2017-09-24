import offshoot
import pyautogui

from serpent.game_agent import GameAgent
from serpent.input_controller import MouseButton
from serpent.machine_learning.context_classification.context_classifiers.cnn_inception_v3_context_classifier import CNNInceptionV3ContextClassifier


class SerpentOneFingerDeathPunchGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.analytics_client = None

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

        # print(self.game.window_geometry['y_offset'])

        if context == "splash_screen":
            print("Boring part, just click on \"Play\"")
            pyautogui.click(button="left", x=640, y=460)
            pyautogui.mouseDown()
            pyautogui.mouseUp()
        if context == "main_menu":
            print("Boring part 2, just click on \"Play\"... again")
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            pyautogui.mouseDown()
            pyautogui.mouseUp()
        if context == "mode_menu":
            print("What to choose ? Oh ! Survival !")
            pyautogui.click(button="left", x=905, y=355)
            pyautogui.mouseDown()
            pyautogui.mouseUp()
        if context == "survival_menu":
            print("I need to start a game :D")

            # self.input_controller.click_screen_region(
            #     button=MouseButton.LEFT,
            #     screen_region="SURVIVAL_MENU_BUTTON_TOP"
            # )
            # Temporary fix
            # pyautogui.click()
            pyautogui.click(button="left", x=640, y=320)
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            # pyautogui.press("left")
        if context == "survival_pre_game":
            print("I don't need skills. Let's play !")

            # pyautogui.click(button="left", x=990, y=100)
            pyautogui.click(button="left", x=990, y=131)
            pyautogui.mouseDown()
            pyautogui.mouseUp()
        if context == "game":
            print("I'M PLAYING !")
        if context == "game_paused":
            print("I'M PAUSING !")
        if context == "game_end_score":
            print("I'M... dead.")
        if context == "game_end_highscore":
            print("I'M... dead. And i have an highscore")

    def handle_game(self):
        pass
