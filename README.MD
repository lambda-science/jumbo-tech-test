# Installation

- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install stable-baselines3[extra] numpy gymnasium pygame`

# Checking the environment

`python jumbo_gym.py`

```
.  .  .  .  .  .  .  .  .  .  .  .
.  .  .  .  .  .  .  .  A  .  .  .
G  ■  .  .  .  .  .  .  .  .  .  .
G  ■  .  .  .  .  .  .  .  .  .  .
.  .  .  .  ■  ■  .  .  ■  .  .  .
.  .  .  .  .  .  .  .  .  .  .  .
.  .  .  .  .  .  .  .  .  .  .  .
.  P  .  .  .  .  .  .  .  .  .  .
.  .  .  .  .  .  .  .  .  .  .  .
.  .  .  .  .  .  .  .  .  .  .  .
.  .  .  .  .  .  .  .  .  .  .  .
G  .  .  .  .  .  .  .  .  .  .  .
```

(A is the agent, G is a good hiding spot, P is the player guard and ■ are pillars)

# Training and evaluating the models

/!\ Please make sure your computer can open a graphic windows for PyGame (typically if using Windows and WSL you will need an X-Serv)  
`python model.py`

```
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 84.7     |
|    ep_rew_mean      | 14.5     |
|    exploration_rate | 0.05     |
| time/               |          |
|    episodes         | 368      |
|    fps              | 8517     |
|    time_elapsed     | 3        |
|    total_timesteps  | 29896    |
----------------------------------
Mean reward:  10.184999999999999
```

And then a PyGame video will open with the agent (blue square) will try to find the good hiding spot (green square, hidden from the line of sight and has at least two adjacent walls) that is hidden from the guard (Red Square). Black squares represent the pillars.  
![pygame](assets/image.png)
