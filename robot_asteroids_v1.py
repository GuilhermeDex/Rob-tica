import gymnasium as gym
import numpy as np
import cv2
import time

class AsteroidsRobot:
    def __init__(self):
        self.env = gym.make('ALE/Asteroids-v5', render_mode='human')
        self.observation, _ = self.env.reset()
        self.actions = {
            'NOOP': 0,
            'FIRE': 1,
            'UP': 2,
            'RIGHT': 3,
            'LEFT': 4,
            'DOWN': 5,
            'UPRIGHT': 6,
            'UPLEFT': 7,
            'DOWNRIGHT': 8,
            'DOWNLEFT': 9,
            'UPFIRE': 10,
            'RIGHTFIRE': 11,
            'LEFTFIRE': 12,
            'DOWNFIRE': 13,
            'UPRIGHTFIRE': 14,
            'UPLEFTFIRE': 15,
            'DOWNRIGHTFIRE': 16,
            'DOWNLEFTFIRE': 17
        }
        self.attractive_force = 0.1
        self.repulsive_force = 1.5
        self.repulsive_range = 30
        self.safe_distance = 20
        self.target = (40, 40)  # Centro da tela (imagem processada)

    def process_image(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return resized

    def find_ship(self, processed_image):
        ship_pos = np.unravel_index(np.argmax(processed_image), processed_image.shape)
        return ship_pos

    def find_asteroids(self, processed_image):
        mask = (processed_image > 30) & (processed_image < 120)
        coords = np.argwhere(mask)
        return coords

    def calculate_potential_field(self, processed_image):
        ship_pos = self.find_ship(processed_image)
        asteroids = self.find_asteroids(processed_image)
        force_x, force_y = 0.0, 0.0
        if len(asteroids) > 0:
            dists = np.linalg.norm(asteroids - ship_pos, axis=1)
            idx = np.argsort(dists)[:3]
            for i in idx:
                ast = asteroids[i]
                ady = ship_pos[0] - ast[0]
                adx = ship_pos[1] - ast[1]
                dist = np.sqrt(adx**2 + ady**2) + 1e-5
                if dist < self.repulsive_range:
                    force_x += self.repulsive_force * adx / (dist ** 2)
                    force_y += self.repulsive_force * ady / (dist ** 2)
        dy = self.target[0] - ship_pos[0]
        dx = self.target[1] - ship_pos[1]
        force_y += self.attractive_force * dy
        force_x += self.attractive_force * dx
        margin = 10
        if ship_pos[0] < margin:
            force_y += 2.0
        if ship_pos[0] > 84 - margin:
            force_y -= 2.0
        if ship_pos[1] < margin:
            force_x += 2.0
        if ship_pos[1] > 84 - margin:
            force_x -= 2.0
        return force_x, force_y, ship_pos, asteroids

    def get_action(self, force_x, force_y, fire, ship_pos, asteroids):
        for ast in asteroids:
            dx = abs(ast[1] - ship_pos[1])
            dy = ship_pos[0] - ast[0]
            if dx < 6 and 0 < dy < 12:
                return self.actions['UP']
        return self.actions['LEFTFIRE']

    def run(self):
        total_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            processed_image = self.process_image(self.observation)
            force_x, force_y, ship_pos, asteroids = self.calculate_potential_field(processed_image)
            action = self.get_action(force_x, force_y, False, ship_pos, asteroids)
            self.observation, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            time.sleep(0.03)
        print(f"Jogo finalizado! Recompensa total: {total_reward}")
        self.env.close()

if __name__ == "__main__":
    robot = AsteroidsRobot()
    robot.run() 