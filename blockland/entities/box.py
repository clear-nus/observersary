import pygame

from blockland.entities.agent import Agent
from blockland.entities.entity import InteractableEntity


class Box(InteractableEntity):

    def __init__(self, env, id, x, y, z):
        super().__init__(env, id, x, y, z)
        self.carried_by = None

    def interact(self, by):
        if not isinstance(by, Agent) or by.carrying is not None:
            return
        by.states["carrying"] += 1
        by.carrying = self
        self.carried_by = by

    @property
    def image_filenames(self):
        return ["box_treasure.png"]

    @property
    def rect(self):
        if self.carried_by is not None:
            return pygame.Rect(-1000, -1000, 0, 0)
        rect = self.image.get_rect()
        rect.midbottom = self.env.convert_coord(self.states["x"], self.states["y"], self.states["z"])
        return rect

    @property
    def hitbox(self):
        if self.carried_by is not None:
            return pygame.Rect(-1, -1, 0, 0)
        x_start, y_start = self.env.convert_coord(self.states["x"] - 0.2, self.states["y"] - 0.1, self.states["z"])
        x_end, y_end = self.env.convert_coord(self.states["x"] + 0.2, self.states["y"] + 0.1, self.states["z"])
        return pygame.Rect(x_start, y_start, x_end - x_start, y_end - y_start)
