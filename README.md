# 2048 AI

## Installation

### Requirements
I assume you have python and pip installed. 

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Enter the virtual environment:
On Linux/macos:
  ```bash
  source venv/bin/activate
  ```
On windows:
  ```bash
  \venv\Scripts\activate
  ```

3. Install the packages
  ```bash
  pip install -r requirements.txt
  ```

## Running the application

To run the training portion, please run 

  ```bash
  python game_env.py
  ```

To run the training portion, please run 

  ```bash
  python game_display.py
  ```

## Playing

In the actual game itself, you can press `p` to watch the DQN model play the game.
You can also press `m` to watch Monte Carlo play the game.
You can also just play the game yourself using the `WASD` keys. Sometimes the program 
might hang when the models are playing so just be patient.


