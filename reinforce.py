import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

def rl():
    style.use("ggplot")

    size = 0

    HM_EPISODE = 25000
    MOVE_PENALTY = 1
    BULLET_PENALTY = 300
    PLAYER_PENALTY = 50

    epsilon = 0.9

    EPS_DECAY = 0.9998
    SHOW_EVERY = 2500

    start_q_tabel = None

    learning_rate = 0.1
    discount = 0.95

    class Blob:
        def action(self, choice):
            if choice == 0:
                self.move(x=5, y=5)
            elif choice == 1:
                self.move(x=-5, y=-5)
            elif choice == 2:
                self.move(x=-5, y=5)
            elif choice == 3:
                self.move(x=5, y=-5)
            elif choice == 4:
                self.move(x=5, y=0)
            elif choice == 5:
                self.move(x=-5, y=0)
            elif choice == 6:
                self.move(x=0, y=-5)
            elif choice == 7:
                self.move(x=0, y=5)

        def move(self, x=False, y=False):

            if not x:
                self.x += np.random.randint(-1, 2)
            else:
                self.x += x

            if not y:
                self.y += np.random.randint(-1, 2)
            else:
                self.y += y

    if start_q_table is None:
        # initialize the q-table#
        q_table = {}
        for x1 in range(-SIZE + 1, SIZE):
            for y1 in range(-SIZE + 1, SIZE):
                for x2 in range(-SIZE + 1, SIZE):
                    for y2 in range(-SIZE + 1, SIZE):
                        q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(8)]

    else:
        with open(start_q_table, "rb") as f:
            q_table = pickle.load(f)

    def train(self):
        for episode in range(HM_EPISODES):
            # bot = Blob()  #####Location
            # player = Blob()
            # bullet = Blob()
            if episode % SHOW_EVERY == 0:
                print(f"on #{episode}, epsilon is {epsilon}")
                print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
                show = True
            else:
                show = False

            episode_reward = 0
            for i in range(200):
                obs = (bot - player, bot - bullet)  # bot to player//bot to bullet

                if np.random.random() > epsilon:
                    # GET THE ACTION
                    action = np.argmax(q_table[obs])
                else:
                    action = np.random.randint(0, 8)
                # Take the action!
                bot.action(action)

                if bot.x == bullet.x and bot.y == bullet.y:
                    reward = -ENEMY_PENALTY
                elif bot.x == player.x and bot.y == player.y:
                    reward = FOOD_REWARD
                else:
                    reward = -MOVE_PENALTY

                new_obs = (bot - player, bot - bullet)
                max_future_q = np.max(q_table[new_obs])
                current_q = q_table[obs][action]

                if reward == PLAYER_PENALTY:
                    new_q = PLAYER_PENALTY
                elif reward == BULLET_PENALTY:
                    new_q = -BULLET_PENALTY
                else:
                    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                q_table[obs][action] = new_q

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel(f"Reward {SHOW_EVERY}ma")
    plt.xlabel("episode #")
    plt.show()

    with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table, f)