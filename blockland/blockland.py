import abc
import pathlib

import gym
import gym.spaces
import numpy as np
import pygame

from blockland.entities.agent import Human
from blockland.entities.agent import Robot
from blockland.entities.block import Brick
from blockland.entities.block import Grass
from blockland.entities.block import Invisible
from blockland.entities.block import Lava
from blockland.entities.block import RoadBlack
from blockland.entities.block import RoadWhite
from blockland.entities.block import Sand
from blockland.entities.block import Water
from blockland.entities.box import Box
from blockland.entities.stall import Stall
from blockland.entities.tree import Tree


class BlocklandEnv(gym.Env):

    metadata = {"render.modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, level):
        """Blockland environment created according to the level configuration.

        Args:
            level: Level configuration.
        """
        self.level = level
        self.level["floors"].reverse()
        self.level["walls"].reverse()

        self.world_width = len(self.level["floors"][0])
        self.world_height = len(self.level["floors"])

        self.time_limit = self.level["time_limit"]

        self.cached_images = {}
        self.screen_margin = 10
        self.screen_xscale = 62
        self.screen_yscale = 44
        self.screen_zscale = 30
        self.screen_width = self.world_width * self.screen_xscale + 2 * self.screen_margin
        self.screen_height = self.world_height * self.screen_yscale + 2 * self.screen_zscale + 2 * self.screen_margin

        self.agent_map = {
            "human": Human,
            "robot": Robot,
        }

        self.entity_map = {
            "box": Box,
            "stall": Stall,
            "tree": Tree,
        }

        self.block_map = {
            "B": Brick,
            "G": Grass,
            "I": Invisible,
            "L": Lava,
            "R": RoadBlack,
            "r": RoadWhite,
            "S": Sand,
            "W": Water,
        }

        pygame.init()
        pygame.display.init()

        self.reset()
        self.observation_space = self.get_observation_space()
        self.action_space = gym.spaces.MultiDiscrete([6] * len(self.level["agents"]))

    def reset(self):
        """Reset the environment.

        """
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.time = 0

        self.agents = []
        self.entities = []

        for agent in self.level["agents"]:
            if agent["type"] in self.agent_map:
                Agent = self.agent_map[agent["type"]]
                x = np.random.uniform(agent["x"][0], agent["x"][1])
                y = np.random.uniform(agent["y"][0], agent["y"][1])
                obj = Agent(self, agent["id"], x, y, 1)
                self.agents.append(obj)
                self.entities.append(obj)

        for entity in self.level["entities"]:
            if entity["type"] in self.entity_map:
                Entity = self.entity_map[entity["type"]]
                x = np.random.uniform(entity["x"][0], entity["x"][1])
                y = np.random.uniform(entity["y"][0], entity["y"][1])
                obj = Entity(self, entity["id"], x, y, 1)
                self.entities.append(obj)

        self.floors = [[None for _ in range(self.world_height)] for _ in range(self.world_width)]
        for i in range(self.world_width):
            for j in range(self.world_height):
                if self.level["floors"][j][i] in self.block_map:
                    Block = self.block_map[self.level["floors"][j][i]]
                    self.floors[i][j] = Block(self, i, j, 0)

        self.walls = [[None for _ in range(self.world_height)] for _ in range(self.world_width)]
        for i in range(self.world_width):
            for j in range(self.world_height):
                if self.level["walls"][j][i] in self.block_map:
                    Block = self.block_map[self.level["walls"][j][i]]
                    self.walls[i][j] = Block(self, i, j, 1)

        states = self.get_states()
        return list(states.values())

    def step(self, actions):
        """Take a step through the environment.

        Args:
            actions: List of actions by all agents in sequential order.
        """
        old_states = self.get_states()
        self.screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        for agent, action in zip(self.agents, actions):
            agent.step(action)

        new_states = self.get_states()
        reward = self.get_reward(old_states, new_states)
        done = self.get_done(new_states)

        self.time += 1
        if self.time >= self.time_limit:
            done = True

        return list(new_states.values()), reward, done, new_states

    def render(self, mode="rgb_array"):
        """Render the current state of the environment.

        Args:
            mode: Render mode. Only supports rgb_array.
        """
        for j in reversed(range(self.world_height)):
            for i in range(self.world_width):
                self.screen.blit(self.floors[i][j].image, self.floors[i][j].rect)

        for j in reversed(range(self.world_height)):
            renderable_entities = []
            for entity in self.entities:
                if j < entity.states["y"] <= j + 1:
                    renderable_entities.append(entity)
            renderable_entities.sort(key=lambda entity: -entity.states["y"])
            for entity in renderable_entities:
                self.screen.blit(entity.image, entity.rect)
            for i in range(self.world_width):
                if self.walls[i][j] is not None:
                    self.screen.blit(self.walls[i][j].image, self.walls[i][j].rect)

        pygame.display.flip()
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        """Close the environment.

        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def load_image(self, filename):
        path = pathlib.Path(pathlib.Path(__file__).parent, "assets", filename)
        if path not in self.cached_images:
            self.cached_images[path] = pygame.image.load(path)
        return self.cached_images[path]

    def convert_coord(self, world_x, world_y, world_z):
        screen_x = self.screen_margin + (world_x * self.screen_xscale)
        screen_y = self.screen_height - self.screen_margin - (world_y * self.screen_yscale) - (world_z * self.screen_zscale)
        return screen_x, screen_y

    def should_include_state(self, state_id):
        return True

    def get_observation_space(self):
        low = []
        high = []
        for entity in self.entities:
            for key in entity.states:
                if self.should_include_state(f"{entity.id}_{key}"):
                    low.append(entity.states_low[key])
                    high.append(entity.states_high[key])
        return gym.spaces.Box(np.array(low), np.array(high))

    def get_states(self):
        states = {}
        for entity in self.entities:
            for key, value in entity.states.items():
                if self.should_include_state(f"{entity.id}_{key}"):
                    states[f"{entity.id}_{key}"] = value
        return states

    @abc.abstractmethod
    def get_reward(self, old_states, new_states):
        return 0

    @abc.abstractmethod
    def get_done(self, states):
        return False
