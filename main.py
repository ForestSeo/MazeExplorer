import csv
import numpy as np
from PIL import Image
import random
import copy
import cv2

# 迷路データ(CSV)の読み込み
filename = "./maze.csv"
maze = []
with open(filename, encoding="utf-8-sig", newline="") as f:
    csvreader = csv.reader(f)
    for i in csvreader:
        maze.append(list(map(int, i)))
maze = np.array(maze)

start = tuple(np.argwhere(maze == 2)[0])
goal = tuple(np.argwhere(maze == 3)[0])

# 画像にする際の準備
temp_maze = np.zeros(maze.shape)
for i in range(len(maze)):
    for j in range(len(maze[0])):
        if maze[i, j] == 0:
            temp_maze[i,j] = 1
        elif maze[i, j] == 2:
            temp_maze[i,j] = 0.3
        elif maze[i, j] == 3:
            temp_maze[i,j] = 0.7
photo = []
photo_size = (maze.shape[1]*10, maze.shape[0]*10)

# Q値の初期化
rows, cols = maze.shape
actions = ['up', 'down', 'left', 'right']
Q = np.zeros((rows, cols, len(actions)))

# パラメーター
alpha = 0.1
gamma = 0.9
epsilon = 0.3
episodes = 1000

# 行動
def move(state, action):
    i, j = state
    if action == 'up':
        i -= 1
    elif action == 'down':
        i += 1
    elif action == 'left':
        j -= 1
    elif action == 'right':
        j += 1
    if 0 <= i < rows and 0 <= j < cols and maze[i, j] != 1:
        return (i, j)
    else:
        return state

# 報酬
def get_reward(state):
    i, j = state
    if maze[i, j] == 3:
        return 100
    elif maze[i, j] == 1:
        return -10
    else:
        return -1

# 1フレームの画像作成
def makephotodata(state):
  temp = copy.copy(temp_maze)
  temp[state[0], state[1]] = 0.5
  temp = temp * 255
  temp_image = Image.fromarray(temp.astype(np.uint8), mode='L')
  temp_image = temp_image.resize(photo_size, Image.NEAREST)
  photo.append(temp_image)

# フレーム画像をまとめて動画に
def makevideo(num):
    width, height = photo[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"/content/output_{num}.mp4", fourcc, 120, (width, height))
    for img_pil in photo:
        img_np = np.array(img_pil)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        out.write(img_cv)
    out.release()
    print(f"動画がoutput_{num}.mp4として保存されました。")

# 実行
count = []
for episode in range(episodes+1):
    c = 0
    state = start
    while state != goal:
        c += 1
        i, j = state
        if episode % 100 == 0:
            makephotodata(state)

        if random.random() < epsilon:
            action_idx = random.randint(0, 3)
        else:
            action_idx = np.argmax(Q[i, j])

        action = actions[action_idx]
        next_state = move(state, action)
        reward = get_reward(next_state)

        ni, nj = next_state
        Q[i, j, action_idx] += alpha * (reward + gamma * np.max(Q[ni, nj]) - Q[i, j, action_idx])
        state = next_state
    count.append(c)

    # 100回ゴールするごとに、その回の行動の様子を動画にする
    if episode % 100 == 0:
      makevideo(episode)
      photo = []

# 最適な道を表示する関数
def print_path():
    state = start
    path = [state]
    while state != goal:
        i, j = state
        action_idx = np.argmax(Q[i, j])
        action = actions[action_idx]
        next_state = move(state, action)
        if next_state == state:
            break
        path.append(next_state)
        state = next_state

    for i in range(rows):
        for j in range(cols):
            if (i, j) == start:
                print('S', end=' ')
            elif (i, j) == goal:
                print('G', end=' ')
            elif (i, j) in path:
                print('*', end=' ')
            elif maze[i, j] == 1:
                print('X', end=' ')
            else:
                print('.', end=' ')
        print()

# 最適な道を表示
print_path()