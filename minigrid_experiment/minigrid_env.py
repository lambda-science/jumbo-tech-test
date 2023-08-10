from __future__ import annotations

import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, Lava
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        determinist=False,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.determinist = determinist
        self.size = 14
        self.agent_start_dir = 0
        self.agent_start_pos = None
        self.guard_position = None
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * self.size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=self.size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "Hide And Seek"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        if self.determinist:
            self.pillars = self._get_predetermined_pillars()
            for pillar in self.pillars:
                self.grid.set(*pillar, Wall())
            self.guard_position = self.place_obj(Lava(), None)
            # Place a goal square in the bottom-right corner
            self.hiding_spots = self._hiding_spots()

        elif self.determinist == False:
            self.pillars = self._get_random_pillars()
            for pillar in self.pillars:
                self.grid.set(*pillar, Wall())
            self.guard_position = self.place_obj(Lava(), None)
            # Place a goal square in the bottom-right corner
            self.hiding_spots = self._hiding_spots(allow_border=True)

        for spot in self.hiding_spots:
            self.put_obj(Goal(), *spot)

        self.place_agent()
        self.mission = "Hide and Seek"

    def _get_predetermined_pillars(self):
        """Generate the 12x12 pre-defined matrix from the subject."""
        matrix = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                [0, 0, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 0, 0],
                [0, 0, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 0, 0],
                [0, 0, 0, 3, 3, 3, 0, 0, 3, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 3, 3, 0, 0, 3, 3, 3, 3, 0, 0],
                [0, 0, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 0, 0],
                [0, 0, 0, 3, 3, 3, 0, 0, 3, 3, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        pillars = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 3:
                    pillars.append((j, i))
        return pillars

    def _get_random_pillars(self):
        """Generate a random number of pillars with random positions and sizes. They are rectangular."""

        num_pillars = 4
        pillars = []  # List of positions with a pillar
        for _ in range(num_pillars):
            width = np.random.randint(2, 4)
            height = np.random.randint(2, 4)
            row = np.random.randint(
                1, self.size - 2 - height
            )  # Ensure (row + height) is within 12
            col = np.random.randint(
                1, self.size - 2 - width
            )  # Ensure (col + width) is within 12
            for i in range(width):
                for j in range(height):
                    pillars.append((row + i, col + j))
        return pillars

    def _distance(self, pos1, pos2):
        """Return the distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _hiding_spots(self, allow_border=False):
        """Return a list of possible hiding spots. A hiding spot is a position that is not visible from the guard and that has at least 2 adjacent walls (or pillars), typically a corner."""
        good_hiding_spots = []
        weak_hiding_spots = []
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                n_adj_walls, at_least_one_pillar = self._number_adjacent_walls(
                    (i, j), allow_border
                )
                if (i, j) in self.pillars or (i, j) == self.guard_position:
                    continue
                elif (
                    not self._has_line_of_sight(self.guard_position, (i, j))
                    and n_adj_walls >= 2
                    and n_adj_walls < 4
                    and at_least_one_pillar
                ):
                    good_hiding_spots.append((i, j))
                elif (
                    not self._has_line_of_sight(self.guard_position, (i, j))
                    and n_adj_walls >= 1
                    and n_adj_walls < 4
                    and at_least_one_pillar
                ):
                    weak_hiding_spots.append((i, j))

        # Filter good hiding spot to keep only the 3 furthest from the guard
        if len(good_hiding_spots) > 3:
            good_hiding_spots = sorted(
                good_hiding_spots,
                key=lambda pos: self._distance(pos, self.guard_position),
                reverse=True,
            )
        # Sort weak_hiding_spots by distance to guard
        weak_hiding_spots = sorted(
            weak_hiding_spots,
            key=lambda pos: self._distance(pos, self.guard_position),
            reverse=True,
        )

        # If there are less than 3 good hiding spots, add weak hiding spots to reach 3
        remaining_spots = 3 - len(good_hiding_spots)
        if remaining_spots > 0:
            good_hiding_spots.extend(weak_hiding_spots[:remaining_spots])

        return good_hiding_spots[:3]

    def _has_line_of_sight(self, guard_position, matrix_position):
        """Check if a given matrix position is visible from the guard position. This is done by checking if there is a line of sight between the two positions with Bresenham's line algorithm and checking if there is a pillar in the line."""
        x1, y1 = guard_position
        x2, y2 = matrix_position

        line = self._bresenham_line(x1, y1, x2, y2)

        for point in line:
            if point in self.pillars:
                return False

        return True

    def _bresenham_line(self, x1, y1, x2, y2):
        """Return a list of points in the line between (x1, y1) and (x2, y2) using Bresenham's line algorithm."""
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

    def _number_adjacent_walls(self, agent_position, allow_border=False):
        """Return the number of adjacent walls (or pillars) to a given position."""
        row, col = agent_position
        adjacent_positions = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]

        num_adjacent_walls = 0
        at_least_one_pillar = False
        for position in adjacent_positions:
            if position in self.pillars:
                at_least_one_pillar = True
                num_adjacent_walls += 1
            elif allow_border and (
                position[0] == 0
                or position[0] == 13
                or position[1] == 0
                or position[1] == 13
            ):
                num_adjacent_walls += 1

        return num_adjacent_walls, at_least_one_pillar

    def set_render_mode(self, mode):
        self.render_mode = mode


def main():
    env = SimpleEnv(render_mode="human", determinist=False)
    # enable manual control for testing
    manual_control = ManualControl(env)
    manual_control.start()


if __name__ == "__main__":
    main()
