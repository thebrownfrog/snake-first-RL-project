import numpy as np
import torch
from torch import nn
import time
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import string

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("training on {}".format(device))

def print_game_state(game_state):
   for v in range(10):
       print()
   for i in range(len(game_state)):
       for j in range(len(game_state[i])):
           p = int(game_state[i][j])
           if p == 0:
            print("□", end=" ")
           if p == 1:
            print("●", end=" ")
           if p > 1:
            print("■", end=" ")
       print()

def bi(bool):
    if (bool):
        return 1
    return 0

def rawoutputfilter(state, snakelength):
    snake = state[1].detach().clone().to(device)
    apples = state[0].detach().clone().to(device)
    snk = (snake > 0).to(device)
    snake[snk] = torch.add(-snake[snk] / snakelength, 3 * torch.ones_like(snake[snk]))
    return (snake.cpu(),apples.cpu())

def outputfilter(state, snakelength, headpos, direction, viewsize, size):
    snake = state[1].detach().clone().to(device)
    apples = state[0].detach().clone().to(device)
    sizex = size[0]
    sizey = size[1]

    half_viewsize = viewsize // 2

    bounds = torch.zeros(viewsize,viewsize).to(device)
    i, j = torch.meshgrid(torch.arange(viewsize), torch.arange(viewsize))

    i.to(device)
    j.to(device)

    cond1 = headpos[0] - half_viewsize + i < 0
    cond2 = headpos[1] - half_viewsize + j < 0
    cond3 = headpos[0] - half_viewsize + i >= sizex
    cond4 = headpos[1] - half_viewsize + j >= sizey

    condition = cond1 | cond2 | cond3 | cond4

    bounds = torch.where(condition, -1, 0)

    i2, j2 = torch.meshgrid(torch.arange(sizex), torch.arange(sizey))

    i2.to(device)
    j2.to(device)
    
    cond12 = headpos[0] + half_viewsize >= i2
    cond22 = headpos[1] + half_viewsize >= j2
    cond32 = headpos[0] - half_viewsize <= i2
    cond42 = headpos[1] - half_viewsize <= j2
    condition2 = cond12 & cond22 & cond32 & cond42

    snakegrid = torch.zeros(viewsize,viewsize).to(device)
    applesgrid = torch.zeros(viewsize,viewsize).to(device)
    
    condition = ~condition

    snakegrid[condition] = snake[condition2]
    applesgrid[condition] = apples[condition2]

    snk = snakegrid > 0
    snakegrid[snk] = torch.add(-snakegrid[snk] / snakelength, 2 * torch.ones_like(snakegrid[snk]))

    rot = bi(direction[1] == 1) + bi(direction[2] == 1) * 2 + bi(direction[3] == 1) * 3

    bounds = bounds.rot90(rot, [0, 1])
    applesgrid = applesgrid.rot90(rot, [0, 1])
    snakegrid = snakegrid.rot90(rot, [0, 1])

    return (bounds.cpu(), applesgrid.cpu(), snakegrid.cpu())

def nextheadxy(snake, action, direction):
    (x,y) = (snake == 1).nonzero(as_tuple=True)
    x = x.tolist()[0]
    y = y.tolist()[0]
    
    if (int(direction[0]) == 1):
        (nx,ny) = (x - action[1],y + action[2] - action[0])
        if (int(action[2]) == 1):
            direction = [0,1,0,0]
        if (int(action[0]) == 1):
            direction = [0,0,0,1]
    elif (int(direction[1]) == 1):
        (nx,ny) = (x + action[2] - action[0],y + action[1])
        if (int(action[2]) == 1):
            direction = [0,0,1,0]
        if (int(action[0]) == 1):
            direction = [1,0,0,0]
    elif (int(direction[2]) == 1):
        (nx,ny) = (x + action[1],y - action[2] + action[0])
        if (int(action[2]) == 1):
            direction = [0,0,0,1]
        if (int(action[0]) == 1):
            direction = [0,1,0,0]
    elif (int(direction[3]) == 1):
        (nx,ny) = (x - action[2] + action[0],y - action[1])
        if (int(action[2]) == 1):
            direction = [1,0,0,0]
        if (int(action[0]) == 1):
            direction = [0,0,1,0]
    
    nheadpos = (nx, ny)
    nheadpos = tuple(map(lambda x: x.int(), nheadpos)) #remove when testing manually!(when uncommenting from line 252)
    return ((nx < snake.size(0)) and (nx >= 0) and (ny < snake.size(1)) and (ny >= 0)), (x,y), nheadpos, direction

def getsnake(snake, headpos):
    i = 1
    gnew = headpos
    def pickadj():
        nonlocal gnew
        if (gnew[0] < snake.size(0) - 1):
            pos = tuple(map(lambda i, j: i + j, gnew, (1,0)))
            if (snake[pos] == i + 1):
                gnew = pos
                return True
        if (gnew[1] < snake.size(1) - 1):
            pos = tuple(map(lambda i, j: i + j, gnew, (0,1)))
            if (snake[pos] == i + 1):
                gnew = pos
                return True
        if (gnew[0] >= 1):
            pos = tuple(map(lambda i, j: i - j, gnew, (1,0)))
            if (snake[pos] == i + 1):
                gnew = pos
                return True
        if (gnew[1] >= 1):
            pos = tuple(map(lambda i, j: i - j, gnew, (0,1)))
            if (snake[pos] == i + 1):
                gnew = pos
                return True
        return False

    snk = [headpos]
    while (pickadj()):
        snk.append(gnew)
        i += 1
    return snk

def movesnake(snake, snakelist, nheadpos):
    snk = [nheadpos] + snakelist
    for x in range(1, len(snk)):
        snake[snk[x-1]] = snake[snk[x]]
    snake[snk[len(snk)-1]] = 0
    return snake

def randxy(grid):
    lst = []
    for x, row in enumerate(grid):
        for y, val in enumerate(row):
            if (val == 0):
                lst.append((x, y))
    if (len(lst) == 0):
        raise ValueError("error. no space to generate new apple.")
    return lst[np.random.randint(0,len(lst))]

class Env:
    def __init__(self, height, width, ar, dr, wr, Apples=2):
        self.applereward = ar
        self.walkreward = wr
        self.diereward = dr
        self.height = height
        self.width = width
        self.sizexy = (height,width)

        self.size = height * width
        self.viewsize = int(2*np.ceil(np.sqrt(1/3*self.size))-1)
        self.applecount = max(min(Apples, self.size - 1), 1)
        self.apples = torch.zeros(self.sizexy)
        self.snake = torch.zeros(self.sizexy)
        pos = (height//2,width//2)
        self.snake[pos] = 1
        for k in range(self.applecount):
            self.apples[randxy(sum(self.apples,self.snake))] = 1
        self.snakelength = 1
        self.t = 0
        self.maxt = self.size * self.size / 4 + 200
        self.direction = [1,0,0,0]

    def reset(self):
        self.apples = torch.zeros(self.sizexy)
        self.snake = torch.zeros(self.sizexy)
        pos = (self.height//2,self.width//2)
        self.snake[pos] = 1
        for k in range(self.applecount):
            self.apples[randxy(sum(self.apples,self.snake))] = 1
        self.snakelength = 1
        self.t = 0
        self.direction = [1,0,0,0]
    
    def increment_snake(self):
        for idx in (self.snake > 0).nonzero():
            self.snake[(idx[0],idx[1])] += 1

    def act(self, action):
        alive, headpos, nheadpos, self.direction = nextheadxy(self.snake, action, self.direction)
        
        if (not alive):
            paststate = (self.apples,self.snake)
            self.reset()
            return False, self.diereward, outputfilter(paststate, self.snakelength, nheadpos, self.direction, self.viewsize, self.sizexy)
        if (self.apples[nheadpos] == 1):
            self.increment_snake()
            self.snake[nheadpos] = 1
            self.apples[nheadpos] = 0
            self.snakelength+=1
            if (self.size - self.snakelength - self.applecount >= 0):
                self.apples[randxy(sum(self.apples,self.snake))] = 1
            if (self.snakelength == self.size):
                paststate = (self.apples,self.snake)
                self.reset()
                return False, self.applereward * 40, outputfilter(paststate, self.snakelength, nheadpos, self.direction, self.viewsize, self.sizexy)
            return True, self.applereward, outputfilter((self.apples,self.snake), self.snakelength, nheadpos, self.direction, self.viewsize, self.sizexy)
        if (self.snake[nheadpos] > 0):
            paststate = (self.apples,self.snake)
            self.reset()
            return False, self.diereward, outputfilter(paststate, self.snakelength, nheadpos, self.direction, self.viewsize, self.sizexy)
        if (self.snake[nheadpos] == 0):
            self.t += 1
            if (self.t >= self.maxt):
                paststate = (self.apples,self.snake)
                self.reset()
                return False, self.diereward, outputfilter(paststate, self.snakelength, nheadpos, self.direction, self.viewsize, self.sizexy)
            
            self.snake = movesnake(self.snake, getsnake(self.snake, headpos), nheadpos)
            return True, self.walkreward, outputfilter((self.apples,self.snake), self.snakelength, nheadpos, self.direction, self.viewsize, self.sizexy)
    
    def observestate(self):
        return True, 0, outputfilter((self.apples,self.snake), self.snakelength, (self.height//2,self.width//2), self.direction, self.viewsize, self.sizexy)
    def rawstate(self):
        return rawoutputfilter((self.apples, self.snake), self.snakelength)

'''env = Env(height=15, width=20, ar=1, dr=-2, wr=-0.01, Apples=2)

for x in range(800):
    m = input()
    a = [0,0,0]
    if (m == 'a'):
        a = [1,0,0]
    elif (m == 'w'):
        a = [0,1,0]
    elif (m == 'd'):
        a = [0,0,1]
    
    #a[np.random.randint(0,3)] = 1
    _, _, (bounds, apples, snake) = env.act(a)
    print_game_state(bounds+apples+snake)
    #time.sleep(0.3)'''

class Policy(nn.Module):
    def __init__(self, input_s, output_s, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(input_s, input_s * 5)
        self.fc2 = nn.Linear(input_s * 5, input_s * 6)
        self.fc3 = nn.Linear(input_s * 6, input_s * 7)
        self.fc4 = nn.Linear(input_s * 7, output_s)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout()
        self.softmax = nn.Softmax()

    def forward(self, input, training):
        input = self.relu(self.fc1(input))
        input = self.relu(self.fc2(input))
        if (training):
            input = self.dropout(input)
        input = self.relu(self.fc3(input))
        if (training):
            input = self.dropout(input)
        return self.softmax(self.fc4(input))
    
    def act(self, states, training=True):
        input = torch.flatten(torch.concat(states)).to(device)
        probs = Categorical(self.forward(input, training).cpu())
        a = probs.sample()
        action = torch.zeros(3).scatter_(dim=0, index=a, src=torch.ones(1))
        return action, probs.log_prob(a)

def training_loop(policy, optimizer, episodes, gamma, env, ar,
                  print_score_every=1000,print_state=True,evaluate=False,save_every=10000):
    _, _, (bounds, apples, snake) = env.observestate()
    rewards = None
    log_probs = None
    score = 0
    no_apples = 0
    episode_lengths = 0
    for episode in range(1, episodes + 1):
        if (print_state):
            appl, snk = env.rawstate()
            print_game_state(appl + snk)
            time.sleep(0.72)
        cont = True
        rewards = []
        log_probs = []
        e_l = 0
        while (cont):
            action, probs = policy.act((bounds, apples, snake), (not evaluate))
            cont, reward, (bounds, apples, snake) = env.act(action)
            rewards.append(reward)
            if (not evaluate):
                log_probs.append(probs)
            if (print_state and cont):
                appl, snk = env.rawstate()
                print_game_state(appl + snk)
                time.sleep(0.072)
            e_l += 1
        
        score += sum(rewards)
        no_apples += (torch.tensor(rewards) == ar).sum().item()
        if (not evaluate):
            for t in range(0,len(rewards)-1)[::-1]:
                rewards[t] += gamma * rewards[t + 1]

            policy_loss = []
            for log_prob, disc_return in zip(log_probs, rewards):
                policy_loss.append((-log_prob * disc_return).unsqueeze(0))
            policy_loss = torch.cat(policy_loss).sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        episode_lengths += e_l
        if (not evaluate and episode % save_every == 0):
            SAVEPATH = "snake10-10-2-v3"
            EXT = ".pt"
            e = str(episode+41500)
            torch.save(policy.state_dict(), SAVEPATH + "e" + e + EXT)
        if (episode % print_score_every == 0):
            print("average score, n.o. apples and ep. length for episodes {}-{}: {:.3f}, {:.2f}, {:.1f}".format(
                episode - print_score_every + 1, episode, score/print_score_every, no_apples/print_score_every, episode_lengths/print_score_every))
            score = 0
            episode_lengths = 0
            no_apples = 0
        
        

parameters = {
    "width":10,
    "height":10,
    "apples":2,
    "episodes":5,
    "gamma":0.971,
    "print_state":True,
    "evaluate":True,
    "print_every":100,
    "learning_rate": 1e-6,
    "applereward":2,
    "diereward":-150,
    "walkreward":-0.03,
    "saveevery":500
}
env = Env(parameters["width"],parameters["height"],parameters["applereward"],
          parameters["diereward"],parameters["walkreward"],parameters["apples"])
LOADPATH = "snake10-10-2-v3e41500.pt"

policy = Policy(env.viewsize * env.viewsize * 3,3).to(device)
if not LOADPATH == "":
    policy = Policy(env.viewsize * env.viewsize * 3,3).to(device)
    policy.load_state_dict(torch.load(LOADPATH, map_location=device))
    if (parameters["evaluate"] == True):
        policy.eval()

optimizer = optim.Adam(policy.parameters(), lr=parameters["learning_rate"])

training_loop(policy, optimizer, parameters["episodes"], parameters["gamma"], env, parameters["applereward"],
              parameters["print_every"],parameters["print_state"],parameters["evaluate"],parameters["saveevery"])
