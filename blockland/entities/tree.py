import pygame

from blockland.entities.entity import Entity


class Tree(Entity):

    @property
    def image_filenames(self):
        return ["foliageTree_green.png"]

    @property
    def rect(self):
        rect = self.image.get_rect()
        rect.midbottom = self.env.convert_coord(self.states["x"], self.states["y"], self.states["z"])
        return rect

    @property
    def hitbox(self):
        x_start, y_start = self.env.convert_coord(self.states["x"] - 0.1, self.states["y"] - 0.1, self.states["z"])
        x_end, y_end = self.env.convert_coord(self.states["x"] + 0.1, self.states["y"] + 0.1, self.states["z"])
        return pygame.Rect(x_start, y_start, x_end - x_start, y_end - y_start)
