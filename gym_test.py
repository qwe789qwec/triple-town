import gym
from pynput import keyboard

key_value = 2

def on_press(key):
    global key_value
    try:
        if key == keyboard.Key.left:
            key_value = 0
        elif key == keyboard.Key.right:
            key_value = 1
        else:
            key_value = 2
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

# 創建 CartPole-v1 環境
env = gym.make('CartPole-v1')

# 重置環境，獲得初始狀態
state = env.reset()
x, x_dot, theta, theta_dot = state
x_error = env.x_threshold - abs(x)
theta_error = 0 - theta
x_error = 0 - x
integral = 0
ki = 0
kd = 40
PIDunmber = 0
pass_theta_error = theta_error
pass_x_error = x_error
score = 0

done = False
while not done:
# while env.x_threshold > abs(x):
    # 渲染當前環境（可選）
    env.render()
    
    # 隨機選擇一個動作（0 或 1）
    if(PIDunmber > 0):
        action = 0
    else:
        action = 1

    if(key_value != 2):
        action = key_value
        key_value = 2
    # action = env.action_space.sample()

    # 執行選擇的動作，獲得下一步的狀態、獎勵和終止標誌
    next_state, reward, done, _ = env.step(action)

    # 在這裡你可以執行你的代碼，根據當前狀態和獎勵來訓練你的智能體
    
    x, x_dot, theta, theta_dot = next_state
    x_error = abs(x)
    theta_error = 0 - theta
    integral += theta_error + x_error
    derivative = (theta_error - pass_theta_error) + (x_error - pass_x_error)
    PIDunmber = theta_error + x_error * 0.1 + ki * integral + kd * derivative
    pass_theta_error = theta_error
    pass_x_error = x_error
    score += reward
    # 更新當前狀態
    state = next_state

# 關閉環境
print(score)
env.close()
listener.join