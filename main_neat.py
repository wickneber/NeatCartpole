import os
import neat
import gym
import numpy as np

def run(config):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config)
    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(5))

    winner = pop.run(fitness, 10)

    print('BEST GENOMEEEE  {}'.format(winner))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    env = gym.make("CartPole-v1")
    observation = env.reset()
    done = False
    while not done:
        action = np.argmax(winner_net.activate(observation))
        observation_, reward, done, info = env.step(action)
        env.render()
        observation = observation_


def fitness(genomes, config):
    env = gym.make("CartPole-v1")
    nets = []
    ge = []
    for genome_id, genome in genomes:
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        genome.fitness = 0
        ge.append(genome)

    # iterate through each net
    for x, net in enumerate(nets):
        done = False
        observation = env.reset()
        iterations = 0
        while not done:
            action = np.argmax(net.activate(observation))
            observation_, reward, done, info = env.step(action)
            if done:
                ge[x].fitness -= 200

            else:
                ge[x].fitness += reward

            iterations += 1
            observation = observation_


def play_game(config):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config)
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-49')
    winner = p.run(fitness, 1)
    winner = neat.nn.FeedForwardNetwork.create(winner, config)
    env = gym.make("CartPole-v1")
    observation = env.reset()
    done = False
    while not done:
        action = np.argmax(winner.activate(observation))
        observation_, reward, done, info = env.step(action)
        env.render()
        observation = observation_


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join('config-feedforward.txt')
    run(config_path)