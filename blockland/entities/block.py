import pygame

from blockland.entities.entity import GameObject


class Block(GameObject):

    @property
    def rect(self):
        rect = self.image.get_rect()
        rect.bottomleft = self.env.convert_coord(self.states["x"], self.states["y"], self.states["z"])
        return rect

    @property
    def hitbox(self):
        if self.states["z"] == 0:
            return None
        x_start, y_start = self.env.convert_coord(self.states["x"], self.states["y"], self.states["z"])
        x_end, y_end = self.env.convert_coord(self.states["x"] + 1, self.states["y"] + 1, self.states["z"])
        return pygame.Rect(x_start, y_end, x_end - x_start, y_start - y_end)


class Brick(Block):

    @property
    def image_filenames(self):
        return ["tileWood_flat.png"]


class Grass(Block):

    @property
    def image_filenames(self):
        return ["tileGrass.png"]


class Invisible(Block):

    @property
    def image_filenames(self):
        return ["tileInvisible.png"]


class Lava(Block):

    @property
    def image_filenames(self):
        return ["tileLava_1.png"]


class RoadBlack(Block):

    @property
    def image_filenames(self):
        return ["tileRoad_black.png"]


class RoadWhite(Block):

    @property
    def image_filenames(self):
        return ["tileRoad_white.png"]


class Sand(Block):

    @property
    def image_filenames(self):
        return ["tileSand.png"]


class Water(Block):

    @property
    def image_filenames(self):
        return ["tileWater_1.png"]
