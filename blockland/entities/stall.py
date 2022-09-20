import pygame

from blockland.entities.agent import Agent
from blockland.entities.entity import InteractableEntity


class Stall(InteractableEntity):

    def __init__(self, env, id, x, y, z):
        super().__init__(env, id, x, y, z)
        self.states["carrying"] = 0
        self.states_low["carrying"] = 0
        self.states_high["carrying"] = 2
        self.carrying = []

    def interact(self, by):
        if not isinstance(by, Agent) or by.carrying is None:
            return
        item = by.carrying
        self.states["carrying"] += 1
        self.carrying.append(item)
        self.frame = self.states["carrying"]
        by.states["carrying"] -= 1
        by.carrying = None
        item.carried_by = self

    @property
    def image_filenames(self):
        return ["market_stallBlue.png", "market_stallBlue1.png", "market_stallBlue2.png"]

    @property
    def rect(self):
        rect = self.image.get_rect()
        rect.midbottom = self.env.convert_coord(self.states["x"], self.states["y"], self.states["z"])
        return rect

    @property
    def hitbox(self):
        x_start, y_start = self.env.convert_coord(self.states["x"] - 0.5, self.states["y"] - 0.3, self.states["z"])
        x_end, y_end = self.env.convert_coord(self.states["x"] + 0.5, self.states["y"] + 0.3, self.states["z"])
        return pygame.Rect(x_start, y_start, x_end - x_start, y_end - y_start)
