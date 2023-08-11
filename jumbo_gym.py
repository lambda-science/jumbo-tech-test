import random
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
    """Custom Gym Environment for Jumbo tech-test. Describe a 12x12
    hide and seek environnement"""

    metadata = {"render_modes": ["human", "rgb_array", "cli"], "render_fps": 5}

    def __init__(self, render_mode=None, determinist=False):
        # Flatten observation space 12x12 -> 144
        self.observation_space = spaces.flatten_space(
            spaces.Box(low=0, high=4, shape=(MATRIX_SIZE, MATRIX_SIZE), dtype=np.uint8)
        )
        # 4 possible actions
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        # 12x12 matrix with posiiton of Agent, player, pillars and good hiding spots
        self.size = MATRIX_SIZE
        self.determinist = determinist
        if self.determinist:
            # Predefined matrix and pillars
            self.matrix, self.pillars = self._determinist_matrix()

        if not self.determinist:
            # Random pillars positions
            self.matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
            self.pillars = self._generate_random_pillars()

        # Randomly place agent and player on the matrix and check
        # if they are not on a pillar
        self.agent_position = self._get_random_start_position()
        while self.agent_position in self.pillars:
            self.agent_position = self._get_random_start_position()
        self.guard_position = self._get_random_start_position()

        while (
            self.guard_position in self.pillars
            or self.guard_position == self.agent_position
        ):
            self.guard_position = self._get_random_start_position()

        self.matrix[self.agent_position] = 1
        self.matrix[self.guard_position] = 2

        # Calculate good hiding spots and place them on the matrix along with pillars
        self.good_hiding_spots = self._hiding_spots()
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

        # 12x12 matrix with posiiton of Agent, player, pillars and good hiding spots
        if self.determinist:
            self.matrix, self.pillars = self._determinist_matrix()

        if not self.determinist:
            self.matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
            self.pillars = self._generate_random_pillars()

        # Randomly place agent and player on the matrix and check
        # if they are not on a pillar
        self.agent_position = self._get_random_start_position()
        while self.agent_position in self.pillars:
            self.agent_position = self._get_random_start_position()
        self.guard_position = self._get_random_start_position()

        while (
            self.guard_position in self.pillars
            or self.guard_position == self.agent_position
        ):
            self.guard_position = self._get_random_start_position()
        self.matrix[self.agent_position] = 1
        self.matrix[self.guard_position] = 2

        # Calculate good hiding spots and place them on the matrix along with pillars
        self.good_hiding_spots = self._hiding_spots()
        for pillar in self.pillars:
            self.matrix[pillar] = 3
        for hiding_spot in self.good_hiding_spots:
            self.matrix[hiding_spot] = 4

        if self.render_mode == "human":
            self._render_frame()
        return self._get_observation(), self._get_info()

    def step(self, action):
        """Perform one step in the environment. Moving in one of the four directions.
        Calculate the reward and check if the game is done."""

        # Reset the previous position of the agent
        self.matrix[self.agent_position] = 0
        done = False
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        new_position = self.agent_position + direction

        # Check if the new position is valid and move the agent
        if (
            (0 <= new_position[0] < MATRIX_SIZE)
            and (0 <= new_position[1] < MATRIX_SIZE)
            and (self.matrix[new_position[0], new_position[1]] != 2)
            and (self.matrix[new_position[0], new_position[1]] != 3)
        ):
            self.agent_position = (new_position[0], new_position[1])

        # Mark the new position in the matrix/observation space and give rewards
        self.matrix[self.agent_position] = 1
        if self.agent_position in self.good_hiding_spots:
            done = True
            reward = 1
        else:
            reward = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), reward, done, False, self._get_info()

    def _get_observation(self):
        obs = np.array(self.matrix, dtype=np.uint8)
        return obs.flatten()

    def render(self):
        """Render the environment. Print the matrix in the CLI (ASCII) or
        show the environment in a PyGame window."""

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
            # The following line will automatically add a delay to keep
            # the framerate stable.
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
        """Generate a random number of obstacles made of multiple pillars (single tile
        wall) with random positions and sizes. They are rectangular with randomly
        missing spots to create hiding spots."""
        num_pillars = np.random.randint(3, 5)
        # Pre define the top left position of the obstacbles
        top_left_pos = [(1, 2), (7, 1), (2, 7), (7, 7)]
        random.shuffle(top_left_pos)
        pillars = []

        for n in range(num_pillars):
            width = np.random.randint(3, 5)
            height = np.random.randint(4, 5)
            pillar_rect = []  # Store pillar positions for each obstacble
            position = top_left_pos[n]

            # Generate pillars for the obstacble
            for i in range(width):
                for j in range(height):
                    pillar_rect.append((position[0] + i, position[1] + j))

            # Determine the number of pillars to remove from the outer layer
            num_to_remove = min(np.random.randint(3, 5), len(pillar_rect))

            # Remove pillars only from the outer layer of the obstacble
            outer_layer = (
                [(position[0] + i, position[1]) for i in range(width)]
                + [(position[0] + i, position[1] + height - 1) for i in range(width)]
                + [(position[0], position[1] + j) for j in range(1, height - 1)]
                + [
                    (position[0] + width - 1, position[1] + j)
                    for j in range(1, height - 1)
                ]
            )

            valid_removal_positions = list(set(outer_layer) & set(pillar_rect))
            removal_positions = random.sample(valid_removal_positions, num_to_remove)

            for pos in removal_positions:
                pillar_rect.remove(pos)

            pillars.extend(pillar_rect)  # Add the remaining pillars for this obstacble

        return pillars

    def _get_random_start_position(self):
        return np.random.randint(0, MATRIX_SIZE), np.random.randint(0, MATRIX_SIZE)

    def _get_info(self):
        return {}

    def _hiding_spots(self):
        """Return the 3 most distant hiding spots. A hiding spot is a position that
        is not visible from the guard and that has at least 2 adjacent walls (pillars),
          typically a corner."""
        good_hiding_spots = []

        # Iterate over all positions in the grid and calculate the number
        # of adjacent walls + line of sight
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                n_adj_walls = self._number_adjacent_walls((i, j))
                if (i, j) in self.pillars or (i, j) == self.guard_position:
                    continue
                elif (
                    not self._has_line_of_sight(self.guard_position, (i, j))
                    and n_adj_walls >= 2
                    and n_adj_walls < 4
                ):
                    good_hiding_spots.append((i, j))

        # Filter good hiding spot to keep only the 3 furthest from the guard
        if len(good_hiding_spots) > 3:
            good_hiding_spots = sorted(
                good_hiding_spots,
                key=lambda pos: self._distance(pos, self.guard_position),
                reverse=True,
            )
            good_hiding_spots = good_hiding_spots[:3]

        return good_hiding_spots

    def _distance(self, pos1, pos2):
        """Return the distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _bresenham_line(self, x1, y1, x2, y2):
        """Return a list of points in the line between (x1, y1) and (x2, y2) using
        Bresenham's line algorithm."""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            points.append((x1, y1))

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return points

    def _has_line_of_sight(self, guard_position, matrix_position):
        """Check if a given matrix position is visible from the guard position. This is
        done by checking if there is a line of sight between the two positions with
        Bresenham's line algorithm and checking if there is a pillar in the line."""
        x1, y1 = guard_position
        x2, y2 = matrix_position

        line = self._bresenham_line(x1, y1, x2, y2)

        for point in line:
            if point in self.pillars:
                return False

        return True

    def _number_adjacent_walls(self, agent_position):
        """Return the number of adjacent pillars to a given position."""
        row, col = agent_position
        adjacent_positions = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]

        num_adjacent_walls = 0

        for position in adjacent_positions:
            if position in self.pillars:
                num_adjacent_walls += 1

        return num_adjacent_walls

    def set_render_mode(self, render_mode):
        """Set the rendering mode."""
        self.render_mode = render_mode

    def set_determinist_mode(self, determinist):
        """Set the matrix determinism mode."""
        self.determinist = determinist

    def _determinist_matrix(self):
        """Generate the 12x12 pre-defined matrix from the subject."""
        matrix = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0],
                [0, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 0],
                [0, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 0],
                [0, 0, 3, 3, 3, 0, 0, 3, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 3, 3, 0, 0, 3, 3, 3, 3, 0],
                [0, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 0],
                [0, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 0],
                [0, 0, 0, 0, 3, 0, 0, 0, 3, 3, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        pillars = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 3:
                    pillars.append((i, j))
        return matrix, pillars


if __name__ == "__main__":
    """Check if the environment is working."""
    while True:
        env = JumboEnv(render_mode="human", determinist=True)
        check_env(env, warn=True)
        env.reset()
        env.render()
