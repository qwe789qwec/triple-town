import gym
from RL_brain import QLearningTable
 
def roundten(number):
	return round(number*10)/10

def make_observcation(state):
	x, x_dot, theta, theta_dot = state
	observation = 0
	if(x > 0):
		observation += 1
	if(x_dot > 0):
		observation += 2
	if(theta > 0):
		observation += 4
	if(theta_dot > 0):
		observation += 8
	
	# return	[roundten(x), roundten(x_dot), roundten(theta), roundten(theta_dot)]
	return [roundten(theta), roundten(theta_dot)]

def train(env):
	RL = QLearningTable(actions=list(range(env.action_space.n)))
	print(list(range(env.action_space.n)))
	state = env.reset()
	max_score = 0
	observation = make_observcation(state)

	for episode in range(1000):
		done = False
		score = 0
		new_reward = 0
		last_reward = 0
		while not done:
			# 渲染當前環境（可選）
			env.render()
			# 隨機選擇一個動作（0 或 1）
			action = RL.choose_action(str(observation))
			# action = env.action_space.sample()

			# 執行選擇的動作，獲得下一步的狀態、獎勵和終止標誌
			next_state, reward, done, _ = env.step(action)

			x, x_dot, theta, theta_dot = next_state
			_observation = [make_observcation(next_state), action]
			# _observation = make_observcation(next_state)
			# r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
			# r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
			# newreward = r1 + r2
			r1 = (env.x_threshold - abs(x))/env.x_threshold
			r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians
			score += reward
			new_reward = (r1 + r2)
			# train_reward = last_reward - new_reward
			# last_reward = new_reward
			RL.learn(str(observation), action, r2, done, str(_observation))
			# if(done):
			# 	print(next_state)
			# 在這裡你可以執行你的代碼，根據當前狀態和獎勵來訓練你的智能體

			# 更新當前狀態
			observation = _observation
		# 關閉環境
		# if(episode%50 == 0):
		# 	print("episode: {}".format(episode))
		# 	print("max_score: {}".format(max_score))
		state = env.reset()
		# print("episode: {}".format(episode))
		# print("score: {}".format(score))
		if(max_score < score):
			max_score = score
			max_episode = episode
			print("max_episode:", max_episode, "max_score:", max_score)
			# 98
	env.close()
	# print("max_episode:", max_episode, "max_score:", max_score)
	return RL.q_table

if __name__ == "__main__":
    # 創建 CartPole-v1 環境
    env = gym.make('CartPole-v1', render_mode = "human")
    qtable = train(env)
    print(qtable)
    # 23519

    # RL = QLearningTable(actions=list(range(env.n_actions)))

    # env.after(100, update)
    # env.mainloop()
    # print(RL.q_table)