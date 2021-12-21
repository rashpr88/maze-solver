import numpy as np
import pylab as plt

location_to_state = {
    (0,0) : 0,
    (0,1) : 1,
    (0,2) : 2,
    (0,3) : 3,
    (1,1) : 4,
    (1,2) : 5,
    (-1,-1) : 6,
    (-1,2) : 7,
    (-1,3) : 8,
    (2,1) : 9,
    (2,2) : 10,
    (2,-1) : 11,
    (3,1) : 12,
    (3,2) : 13,
    (3,3) : 14,
}

direction_to_action = {"up": 0,
"down": 1,
"right": 2,
"left" :3}

up=0
down=1
right=2
left=3

rewards = np.array([[1,-1,-1,-1],
              [2,0,4,-1],
              [3,1,5,7],
              [-1,2,-1,8],
              [5,-1,9,1],
              [-1,4,2,10],
              [7,-1,11,-1],
              [8,6,-1,-1],
              [-1,7,3,-1],
              [10,11,12,4],
              [-1,9,5,13],
              [9,-1,-1,6],
              [-1,-1,-1,9],
              [14,-1,-1,10],
              [-1,13,-1,-1]])

state_to_location = dict((state,location) for location,state in location_to_state.items())
action_to_direction = dict((action,direction) for direction,action in direction_to_action.items())

Q = np.array(np.zeros([15,4]))
for i in range(1000):
    # Pick up a state randomly
    current_state = np.random.randint(0,15)

Rew = np.copy(rewards)
print(Rew)

gamma = 0.7

initial_state = 0

playable_actions = []
def available_actions(s):

	current_state_row = Rew[s,]
	for j in range(4):
		if Rew[current_state, j] >= 0:
			playable_actions.append(j)
	return playable_actions


available_act = playable_actions

available_act = available_actions(initial_state)


def sample_next_action(available_actions_range):
	next_action = int(np.random.choice(available_act))
	return next_action


action = sample_next_action(available_act)


def update(current_state, action, gamma):
	if Rew[current_state, action] == 12:
		Q[current_state, action] = 100
	else:
		max_values = np.max(Q, 1)
		state = Rew[current_state, action]
		max_value = max_values[state]

		Q[current_state, action] = Rew[current_state, action] + gamma * max_value
		print('max_value', Rew[current_state, action] + gamma * max_value)

	if (np.max(Q) > 0):
		return (np.sum(Q / np.max(Q) * 100))

	else:
		return (0)


update(initial_state, action, gamma)

scores = []
for i in range(1000):
	current_state = np.random.randint(0, 15)
	available_act = available_actions(current_state)
	action = sample_next_action(available_act)
	score = update(current_state, action, gamma)
	scores.append(score)
	print('Score:', str(score))

print("Trained Q matrix:")
print(Q / np.max(Q) * 100)

plt.plot(scores)
plt.show()


def get_optimal_route(start_location, end_location):
	ending_state = location_to_state[end_location]
	st_state = location_to_state[start_location]
	route = [start_location]
	max_values = np.max(Q, 1)

	while (st_state != ending_state):

		max_v = max_values[st_state]

		for i in range(4):
			if Q[st_state, i] == max_v:
				n = i
		next_location = Rew[st_state, n]
		route.append(state_to_location[next_location])
		st_state = next_location
	return route


r = get_optimal_route((0, 0), (3, 1))
print("Optimal route : ", r)

seq = np.array([[(0, 0), up, right, right, right],
				[(0, 2), down, right, up, right, down, right],
				[(1, 2), down, right, right]
				])
q = np.array(np.zeros([15, 4]))

for episode in range(len(seq)):
	print("Episode :", episode + 1)
	st = seq[episode]
	state = location_to_state[st[0]]
	print("State : ", st[0])
	i = 1

	while i != len(st):
		action = st[i]
		act = action_to_direction[action]
		print("Action : " + act)

		AllMax = np.max(q, 1)

		next_state = Rew[state, action]

		qMax = AllMax[next_state]
		if Rew[state, action] == 12:
			q[state, action] = 100
		else:
			q[state, action] = Rew[state, action] + gamma * qMax
		state = Rew[state, action]
		i = i + 1
	if (np.max(Q) > 0):
		np.sum(Q / np.max(Q) * 100)
	else:
		0

	print("Q table : \n", q)

