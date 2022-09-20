import abc

import pygame

from blockland.entities.entity import Entity
from blockland.entities.entity import InteractableEntity


class Agent(Entity):

    def __init__(self, env, id, x, y, z):
        super().__init__(env, id, x, y, z)
        self.states["carrying"] = 0
        self.states_low["carrying"] = 0
        self.states_high["carrying"] = 1
        self.carrying = None

    def step(self, action):
        prev_x = self.states["x"]
        prev_y = self.states["y"]

        if action == 1:
            self.states["x"] = min(self.env.world_width - 0.1, self.states["x"] + 0.1)
        if action == 2:
            self.states["y"] = min(self.env.world_height - 0.1, self.states["y"] + 0.1)
        if action == 3:
            self.states["x"] = max(0.1, self.states["x"] - 0.1)
        if action == 4:
            self.states["y"] = max(0.1, self.states["y"] - 0.1)

        for i in range(self.env.world_width):
            for j in range(self.env.world_height):
                if self.env.walls[i][j] is not None:
                    if self.hitbox.colliderect(self.env.walls[i][j].hitbox):
                        self.states["x"] = prev_x
                        self.states["y"] = prev_y

        for entity in self.env.entities:
            if entity != self and self.hitbox.colliderect(entity.hitbox):
                self.states["x"] = prev_x
                self.states["y"] = prev_y

        if self.carrying is not None:
            self.carrying.states["x"] = self.states["x"]
            self.carrying.states["y"] = self.states["y"]

        if action == 5:
            for entity in self.env.entities:
                if isinstance(entity, InteractableEntity):
                    if entity != self and self.interact_hitbox.colliderect(entity.hitbox):
                        entity.interact(self)

    @abc.abstractproperty
    def image_filenames(self):
        raise NotImplementedError

    @property
    def rect(self):
        rect = self.image.get_rect()
        rect.midbottom = self.env.convert_coord(self.states["x"], self.states["y"], self.states["z"])
        return rect

    @property
    def hitbox(self):
        x_start, y_start = self.env.convert_coord(self.states["x"] - 0.1, self.states["y"] - 0.1, self.states["z"])
        x_end, y_end = self.env.convert_coord(self.states["x"] + 0.1, self.states["y"] + 0.1, self.states["z"])
        return pygame.Rect(x_start, y_end, x_end - x_start, y_start - y_end)

    @property
    def interact_hitbox(self):
        x_start, y_start = self.env.convert_coord(self.states["x"] - 0.5, self.states["y"] - 0.5, self.states["z"])
        x_end, y_end = self.env.convert_coord(self.states["x"] + 0.5, self.states["y"] + 0.5, self.states["z"])
        return pygame.Rect(x_start, y_end, x_end - x_start, y_start - y_end)


class Human(Agent):

    @property
    def image_filenames(self):
        return ["character_man.png"]


class Robot(Agent):

    @property
    def image_filenames(self):
        return ["character_robot.png"]
