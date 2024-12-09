from tkinter import Frame, Label, CENTER
from game_functions import CELL_COUNT
import torch
from model import DQN
from game_env import Game2048Env
from tkinter import messagebox

import game_ai
import numpy as np
import game_functions

EDGE_LENGTH = 400

CELL_PAD = 10

UP_KEY = "'w'"
DOWN_KEY = "'s'"
LEFT_KEY = "'a'"
RIGHT_KEY = "'d'"
AI_KEY = "'q'"
AI_PLAY_KEY = "'p'"

LABEL_FONT = ("Verdana", 40, "bold")

GAME_COLOR = "#92877d"

EMPTY_COLOR = "#9e948a"

LABEL_COLORS = {
    2: "#776e65",
    4: "#776e65",
    8: "#f9f6f2",
    16: "#f9f6f2",
    32: "#f9f6f2",
    64: "#f9f6f2",
    128: "#f9f6f2",
    256: "#f9f6f2",
    512: "#f9f6f2",
    1024: "#f9f6f2",
    2048: "#f9f6f2",
    4096: "#f9f6f2",
}

TILE_COLORS = {
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
    4096: "#3c3a32",
}


class Display(Frame):
    def __init__(self, model_path=None):
        Frame.__init__(self)

        self.grid()
        self.master.title("2048")

        self.commands = {
            UP_KEY: game_functions.move_up,
            DOWN_KEY: game_functions.move_down,
            LEFT_KEY: game_functions.move_left,
            RIGHT_KEY: game_functions.move_right,
            AI_PLAY_KEY: game_ai.ai_move,
            # AI_KEY: self.ai_single_move,
            # AI_PLAY_KEY: self.ai_continuous_play,
        }

        self.master.bind("<Key>", self.key_press)

        self.env = Game2048Env()

        self.model = DQN(input_shape=(4, 4))
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.grid_cells = []
        self.build_grid()
        self.init_matrix()
        self.draw_grid_cells()

        self.mainloop()

    def ai_single_move(self):
        state = np.array(self.matrix).reshape(1, 4, 4)

        valid_actions = self.env.get_valid_actions()

        if not valid_actions:
            return False

        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze()

        valid_q_values = q_values[valid_actions]
        action = valid_actions[torch.argmax(valid_q_values).item()]

        move_functions = [
            game_functions.move_left,
            game_functions.move_up,
            game_functions.move_down,
            game_functions.move_right,
        ]
        self.matrix, move_made, _ = move_functions[action](self.matrix)
        if move_made:
            self.matrix = game_functions.add_new_tile(self.matrix)
            self.draw_grid_cells()
            
        return move_made

    def ai_continuous_play(self):
        move_count = 0
        while True:
            move_made = self.ai_single_move()
            if not move_made:
                break
            move_count += 1
            # Optional: add a small delay or update to see moves
            self.update_idletasks()
        
        print(f"AI played {move_count} moves")
    
    def build_grid(self):
        background = Frame(self, bg=GAME_COLOR, width=EDGE_LENGTH, height=EDGE_LENGTH)
        background.grid()

        for row in range(CELL_COUNT):
            grid_row = []
            for col in range(CELL_COUNT):
                cell = Frame(
                    background,
                    bg=EMPTY_COLOR,
                    width=EDGE_LENGTH / CELL_COUNT,
                    height=EDGE_LENGTH / CELL_COUNT,
                )
                cell.grid(row=row, column=col, padx=CELL_PAD, pady=CELL_PAD)
                t = Label(
                    master=cell,
                    text="",
                    bg=EMPTY_COLOR,
                    justify=CENTER,
                    font=LABEL_FONT,
                    width=5,
                    height=2,
                )
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def init_matrix(self):
        self.matrix = game_functions.initialize_game()

    def draw_grid_cells(self):
        for row in range(CELL_COUNT):
            for col in range(CELL_COUNT):
                tile_value = self.matrix[row][col]
                if not tile_value:
                    self.grid_cells[row][col].configure(text="", bg=EMPTY_COLOR)
                else:
                    self.grid_cells[row][col].configure(
                        text=str(tile_value),
                        bg=TILE_COLORS[tile_value],
                        fg=LABEL_COLORS[tile_value],
                    )
        self.update_idletasks()

    def key_press(self, event):
        valid_game = True
        key = repr(event.char)
        if key in self.commands:
            if key == AI_PLAY_KEY:
                # Direct call to ai_continuous_play without passing matrix
                move_count = 0
                while valid_game:
                    self.matrix, valid_game = game_ai.ai_move(self.matrix,40, 30)
                    if valid_game:
                        self.matrix = game_functions.add_new_tile(self.matrix)
                        self.draw_grid_cells()
                    move_count += 1
                    self.ai_continuous_play()
            else:
                # For other commands, proceed as before
                self.matrix, move_made, _ = self.commands[key](self.matrix)
                if move_made:
                    self.matrix = game_functions.add_new_tile(self.matrix)
                    self.draw_grid_cells()
                    
                if game_functions.game_over(self.matrix):
                    messagebox.showwarning("Game over", "No more moves available!")
                    self.init_matrix()
                    self.draw_grid_cells()


gamegrid = Display(model_path="./snapshots/2048_dqn_model_ep3500.pth")
