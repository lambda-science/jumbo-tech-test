import gymnasium as gym
import numpy as np
import math
import pygame
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
MATRIX_SIZE = 12


class JumboEnv(gym.Env):
    """Custom Gym Environment for Jumbo tech-test. Describe a 12x12 hide and seek environnement"""

    metadata = {"render_modes": ["human", "rgb_array", "cli"], "render_fps": 4}

    def __init__(self, render_mode=None):
        # Flatten observation space 12x12 -> 144
        self.observation_space = spaces.flatten_space(
            spaces.Box(low=0, high=4, shape=(MATRIX_SIZE, MATRIX_SIZE), dtype=np.uint8)
        )
        # 4 possible actions
        self.action_space = gym.spaces.Discrete(4)
        # 12x12 matrix with posiiton of Agent, player, pillars and good hiding spots
        self.matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
        self.agent_position = (0, 0)
        self.guard_position = (0, 1)
        self.pillars = self._generate_random_pillars()
        self.good_hiding_spots = self._hiding_spots()
        self.visited_positions = []
        # Placing these elements on the 12x12 matrix
        self.matrix[self.agent_position] = 1
        self.matrix[self.guard_position] = 2
        for pillar in self.pillars:
            self.matrix[pillar] = 3
        for hiding_spot in self.good_hiding_spots:
            self.matrix[hiding_spot] = 4

        # PyGames Settings
        self.window_size = 512
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """Reset the environment. Random position for AI, agent and pillars."""
        super().reset(seed=seed)
        self.matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
        self.pillars = self._generate_random_pillars()
        self.agent_position = self._get_random_start_position()
        self.good_hiding_spots = self._hiding_spots()
        self.visited_positions = []

        # Perform checks to make sure the agent is not spawning on a pillar or on the guard
        while self.agent_position in self.pillars:
            self.agent_position = self._get_random_start_position()

        self.guard_position = self._get_random_start_position()

        while (
            self.guard_position in self.pillars
            or self.guard_position == self.agent_position
        ):
            self.guard_position = self._get_random_start_position()

        # Place these elements on the 12x12 matrix as before
        self.matrix[self.agent_position] = 1
        self.matrix[self.guard_position] = 2
        for pillar in self.pillars:
            self.matrix[pillar] = 3
        for hiding_spot in self.good_hiding_spots:
            self.matrix[hiding_spot] = 4

        if self.render_mode == "human":
            self._render_frame()
        return self._get_observation(), self._get_info()

    def step(self, action):
        """Perform one step in the environment. Moving in one of the four directions. Calculate the reward and check if the game is done."""

        # Reset the previous position of the agent
        self.matrix[self.agent_position] = 0
        done = False
        if action == 0:  # Move up
            new_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif action == 1:  # Move down
            new_position = (self.agent_position[0] + 1, self.agent_position[1])
        elif action == 2:  # Move left
            new_position = (self.agent_position[0], self.agent_position[1] - 1)
        elif action == 3:  # Move right
            new_position = (self.agent_position[0], self.agent_position[1] + 1)

        # Check if the new position is valid (not colliding with pillars or out of bounds)
        if (
            0 <= new_position[0] < MATRIX_SIZE
            and 0 <= new_position[1] < MATRIX_SIZE
            and new_position not in self.pillars
        ):
            self.agent_position = new_position

        # Mark the new position
        self.matrix[self.agent_position] = 1
        # Calculate the reward
        reward, done = self._custom_reward_function()
        # Add the new position to the list of visited positions
        self.visited_positions.append(self.agent_position)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), reward, done, False, self._get_info()

    def _custom_reward_function(self):
        """Custom reward function for the environment. Encourage exploration and stop the game if the agent is in a  good hiding spot."""
        reward_good_spot = 50.0  # Positive reward for finding a good hiding spot
        reward_explore = 0.05  # Reward for exploring new positions
        done = False
        total_reward = 0

        if self.agent_position in self.good_hiding_spots:
            done = True
            return reward_good_spot, done

        # Encourage exploration
        if tuple(self.agent_position) not in self.visited_positions:
            total_reward += reward_explore

        return total_reward, done

    def _get_observation(self):
        obs = np.array(self.matrix, dtype=np.uint8)
        return obs.flatten()

    def render(self):
        """Render the environment. Print the matrix in the CLI (ASCII) or show the environment in a PyGame window."""

        if self.render_mode == "cli":
            render_matrix = np.full((MATRIX_SIZE, MATRIX_SIZE), ".")
            render_matrix[self.agent_position] = "A"
            render_matrix[self.guard_position] = "P"
            for good_hiding_spot in self.good_hiding_spots:
                row, col = good_hiding_spot
                render_matrix[row, col] = "G"
            for pillar in self.pillars:
                row, col = pillar
                render_matrix[row, col] = "■"
            for row in render_matrix:
                print(" ".join(str(item).ljust(2) for item in row))
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """Render the environment in a PyGame window, from official documentation."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        cell_size = self.window_size / MATRIX_SIZE
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)

        for i in range(MATRIX_SIZE):
            for j in range(MATRIX_SIZE):
                pygame.draw.rect(
                    canvas,
                    WHITE,
                    (j * cell_size, i * cell_size, cell_size, cell_size),
                    1,
                )
                if self.agent_position == (i, j):
                    pygame.draw.rect(
                        canvas,
                        BLUE,
                        (j * cell_size, i * cell_size, cell_size, cell_size),
                    )
                elif self.guard_position == (i, j):
                    pygame.draw.rect(
                        canvas,
                        RED,
                        (j * cell_size, i * cell_size, cell_size, cell_size),
                    )
                elif (i, j) in self.pillars:
                    pygame.draw.rect(
                        canvas,
                        BLACK,
                        (j * cell_size, i * cell_size, cell_size, cell_size),
                    )
                elif (i, j) in self.good_hiding_spots:
                    pygame.draw.rect(
                        canvas,
                        GREEN,
                        (j * cell_size, i * cell_size, cell_size, cell_size),
                    )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """Close the PyGame window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _generate_random_pillars(self):
        """Generate a random number of pillars with random positions and sizes. They are rectangular."""

        num_pillars = np.random.randint(3, 4)
        pillars = []  # List of positions with a pillar
        for _ in range(num_pillars):
            width = np.random.randint(1, 4)
            height = np.random.randint(1, 4)
            row = np.random.randint(
                0, MATRIX_SIZE - 1 - height
            )  # Ensure (row + height) is within 12
            col = np.random.randint(
                0, MATRIX_SIZE - 1 - width
            )  # Ensure (col + width) is within 12
            for i in range(width):
                for j in range(height):
                    pillars.append((row + i, col + j))
        return pillars

    def _get_random_start_position(self):
        return np.random.randint(0, MATRIX_SIZE), np.random.randint(0, MATRIX_SIZE)

    def _get_info(self):
        return {}

    def _hiding_spots(self):
        """Return a list of possible hiding spots. A hiding spot is a position that is not visible from the guard and that has at least 2 adjacent walls (or pillards), typically a corner."""
        good_hiding_spots = []
        coordinates_to_remove = [(0, 0), (11, 11), (0, 11), (12, 0)]
        for i in range(MATRIX_SIZE):
            for j in range(MATRIX_SIZE):
                if (i, j) in self.pillars or (i, j) == self.guard_position:
                    continue
                elif (
                    not self._is_visible(self.guard_position, (i, j))
                    and self._number_adjacent_walls((i, j)) >= 2
                ):
                    good_hiding_spots.append((i, j))
        filtered_coordinates = [
            coord for coord in good_hiding_spots if coord not in coordinates_to_remove
        ]
        return filtered_coordinates

    def _is_visible(self, src, tgt):
        """Check if a target position is visible from a source position. This is done by checking if there is a line of sight between the two positions, without any obstacles (pillars) in between. To check if the player can see the agent."""
        # Check if the target is visible from the source position
        if src == tgt:
            return False

        src_row, src_col = src
        tgt_row, tgt_col = tgt
        diff_row, diff_col = tgt_row - src_row, tgt_col - src_col
        steps = max(abs(diff_row), abs(diff_col))

        # Use integer division to calculate step size
        row_step, col_step = diff_row // steps, diff_col // steps

        for i in range(steps + 1):
            check_row, check_col = src_row + i * row_step, src_col + i * col_step

            # Check if the line of sight goes out of the grid boundaries
            if check_row < 0 or check_row >= 12 or check_col < 0 or check_col >= 12:
                return False

            # Check for obstacles (pillars) along the line of sight
            if (check_row, check_col) in self.pillars:
                return False

        return True

    def _number_adjacent_walls(self, agent_position):
        """Return the number of adjacent walls (or pillars) to a given position."""
        row, col = agent_position
        adjacent_positions = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]

        num_adjacent_walls = 0

        for position in adjacent_positions:
            if (
                position[0] < 0
                or position[0] >= MATRIX_SIZE
                or position[1] < 0
                or position[1] >= MATRIX_SIZE
                or position in self.pillars
            ):
                num_adjacent_walls += 1

        return num_adjacent_walls


if __name__ == "__main__":
    """ Check if the environment is working."""
    env = JumboEnv(render_mode="cli")
    check_env(env)
    env.reset()
    env.render()
