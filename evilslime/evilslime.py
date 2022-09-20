import cv2
import gym
import gym.spaces
import numpy as np


class Color:

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (239, 71, 111)
    GREEN = (6, 214, 160)
    BLUE = (17, 138, 178)
    YELLOW = (239, 71, 111)


class EvilSlimeEnv(gym.Env):

    def __init__(self, body_color, background_color):
        """Environment with a slime that is limited to only horizontal movement.

        Args:
            body_color: Color of slime body.
            background_color: Color of background.
        """
        self.threshold = 5.0
        self.position = 0.0
        self.absolute_velocity = 0.025
        self.body_color = body_color
        self.background_color = background_color

        self.metadata["render_modes"] = ["rgb_array"]
        self.observation_space = gym.spaces.Box(-self.threshold, self.threshold, [1])
        self.action_space = gym.spaces.Discrete(3)

    def step(self, action):
        """Take a step through the environment.

        Args:
            action: Slime action. 0 to move left, 1 to do nothing, 2 to move right.
        """
        velocity = (action - 1) * self.absolute_velocity
        self.position = np.clip(self.position + velocity, -self.threshold, self.threshold)
        return self.position, 0.0, False, None

    def reset(self):
        """Reset the environment.

        """
        self.position = 0
        return self.position

    def render(self, mode="rgb_array", render_width=200, render_height=100):
        """Render the current state of the environment.

        Args:
            mode: Render mode. Only supports rgb_array.
            render_width: Width of the render.
            render_height: Height of the render.
        """
        assert mode == "rgb_array"

        radius = 30.0
        center_x = (((self.position + self.threshold) / (self.threshold * 2)) * (render_width - 2 * radius)) + radius

        body_center_x = int(center_x)
        body_center_y = int(0.0)
        body_radius = int(radius)
        eye_center_x = int(center_x)
        eye_center_y = int(radius * 0.5)
        eye_radius = int(radius * 0.4)
        eyeball_center_x = int(center_x)
        eyeball_center_y = int(radius * 0.65)
        eyeball_radius = int(radius * 0.2)

        image = np.full((render_height, render_width, 3), self.background_color, np.uint8)
        cv2.circle(image, (body_center_x, body_center_y), body_radius, self.body_color, -1)
        cv2.circle(image, (eye_center_x, eye_center_y), eye_radius, Color.WHITE, -1)
        cv2.circle(image, (eyeball_center_x, eyeball_center_y), eyeball_radius, Color.BLACK, -1)
        return image

    def seed(self, seed):
        """Set the seed for random number generators.

        Args:
            seed: Seed value for the generators.
        """
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
