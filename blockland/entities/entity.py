import abc

import pygame


class GameObject(pygame.sprite.Sprite):

    def __init__(self, env, x, y, z):
        """Game object in a Blockland environment.

        Args:
            env: Blockland environment.
            x: Initial x position.
            y: Initial y position.
            z: Initial z position.
        """
        super().__init__()
        self.env = env

        self.states = {}
        self.states_low = {}
        self.states_high = {}

        self.states["x"] = x
        self.states_low["x"] = 0
        self.states_high["x"] = env.world_width

        self.states["y"] = y
        self.states_low["y"] = 0
        self.states_high["y"] = env.world_height

        self.states["z"] = z
        self.states_low["z"] = 0
        self.states_high["z"] = 1

        self.frame = 0

        self.images = []
        for image_filename in self.image_filenames:
            self.images.append(self.env.load_image(image_filename))

    @property
    def image_filenames(self):
        return []

    @property
    def image(self):
        return self.images[self.frame]

    @abc.abstractproperty
    def rect(self):
        raise NotImplementedError

    @abc.abstractproperty
    def hitbox(self):
        raise NotImplementedError


class Entity(GameObject):

    def __init__(self, env, id, x, y, z):
        super().__init__(env, x, y, z)
        self.id = id


class InteractableEntity(Entity):

    @abc.abstractmethod
    def interact(self, by):
        raise NotImplementedError
