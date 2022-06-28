import copy
import logging

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)

# OpenAI Gymのインターフェイスに従って環境を実装


class Fruits(gym.Env):
    """
    This is a modified version of "Checkers".
    Apples and lemons are randomly placed every time you reset the enviroment.
    「Checkers」の果物をランダムに配置するバージョン

    ---
    The map contains apples and lemons. The first player (red) is very sensitive and scores 10 for
    the team for an apple (green square) and −10 for a lemon (orange square). The second (blue), less sensitive
    player scores 1 for the team for an apple and −1 for a lemon. There is a wall of lemons between the
    players and the apples. Apples and lemons disappear when collected, and the environment resets
    when all apples are eaten. It is important that the sensitive agent eats the apples while the less sensitive
    agent should leave them to its team mate but clear the way by eating obstructing lemons.

    Reference Paper : Value-Decomposition Networks For Cooperative Multi-Agent Learning (Section 4.2)
    ---
    """
    metadata = {'render.modes': [
        'human', 'rgb_array'], 'video.frames_per_second': 5}

    def __init__(self, full_observable=False, step_cost=-0.01, max_steps=100, clock=False):
        # グリッドの形
        self._grid_shape = (3, 8)
        # エージェント数
        self.n_agents = 2
        # 最大ステップ数
        self._max_steps = max_steps
        # 現在のタイムステップ
        self._step_count = None
        # タイムステップが経過する毎に与えられる罰
        self._step_cost = step_cost
        # 完全観測可能か
        self.full_observable = full_observable
        # 観測に経過時間を含めるか
        self._add_clock = clock

        # 行動空間（[上,下,左,右,何もしない]の5種類が人数分）
        self.action_space = MultiAgentActionSpace(
            [spaces.Discrete(5) for _ in range(self.n_agents)])

        # 観測の最大値
        self._obs_high = np.ones(2 + (3 * 3 * 5) + (1 if clock else 0))
        # 観測の最小値
        self._obs_low = np.zeros(2 + (3 * 3 * 5) + (1 if clock else 0))

        # 全て観測可能な場合の最大・最小値
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)

        # 観測空間
        self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high)
                                                             for _ in range(self.n_agents)])

        # 各エージェントの初期位置
        self.init_agent_pos = {
            0: [0, self._grid_shape[1] - 2], 1: [2, self._grid_shape[1] - 2]}

        # 各エージェントの得られる報酬値
        self.agent_reward = {0: {'lemon': -10, 'apple': 10},
                             1: {'lemon': -1, 'apple': 1}}

        self.agent_prev_pos = None  # 1ステップ前のエージェントの位置
        self._base_grid = None  #
        self._full_obs = None  # グリッドの現在の状態
        self._agent_dones = None  # 終了したかどうか
        self.viewer = None  # 描画用
        self._food_count = None  # 残っている果物の数
        self._total_episode_reward = None  # 報酬の総和
        self.steps_beyond_done = None  # エラー用

        # 乱数のシードを初期化
        self.seed()

    def get_action_meanings(self, agent_i=None):
        """
        各行動の意味を文字列で取得
        """
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def __draw_base_img(self):
        """
        エージェントを除くグリッドや果物の画像を生成
        """
        self._base_img = draw_grid(
            self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):
                if PRE_IDS['wall'] in self._full_obs[row][col]:
                    fill_cell(self._base_img, (row, col),
                              cell_size=CELL_SIZE, fill=WALL_COLOR, margin=0.05)
                elif PRE_IDS['lemon'] in self._full_obs[row][col]:
                    fill_cell(self._base_img, (row, col),
                              cell_size=CELL_SIZE, fill=LEMON_COLOR, margin=0.05)
                elif PRE_IDS['apple'] in self._full_obs[row][col]:
                    fill_cell(self._base_img, (row, col),
                              cell_size=CELL_SIZE, fill=APPLE_COLOR, margin=0.05)

    def __create_grid(self):
        """
        create grid and fill in lemon and apple locations. This grid doesn't fill agents location
        グリッドを生成して果物を配置する
        グリッドを返す
        """
        _grid = []
        for row in range(self._grid_shape[0]):  # 各行ごと
            if row % 2 == 0:
                # 偶数行
                # 偶数列はリンゴ、奇数列はレモンを配置して、右端2マスは空にする
                _grid.append([PRE_IDS['apple'] if (c % 2 == 0) else PRE_IDS['lemon']
                              for c in range(self._grid_shape[1] - 2)] + [PRE_IDS['empty'], PRE_IDS['empty']])
            else:
                # 奇数行
                # 偶数列はレモン、奇数列はリンゴを配置して、右端2マスは空にする
                _grid.append([PRE_IDS['apple'] if (c % 2 != 0) else PRE_IDS['lemon']
                              for c in range(self._grid_shape[1] - 2)] + [PRE_IDS['empty'], PRE_IDS['empty']])

        return _grid

    def __create_random_grid(self):
        """
        ランダムに果物を配置したグリッドを生成
        """
        _grid = []

        # 果物の合計数（右の2列以外）
        fruit_total = self._grid_shape[0] * (self._grid_shape[1] - 2)

        # 半分リンゴ、半分レモンを並べた1行のリスト
        all_fruits = [PRE_IDS['apple'] if (
            i < fruit_total // 2) else PRE_IDS['lemon'] for i in range(fruit_total)]

        # Numpy配列に変換してシャッフル
        shuffled_all_fruits = np.random.permutation(np.array(all_fruits))

        # 1行の配列をグリッドの行列に成形
        _grid = shuffled_all_fruits.reshape(self._grid_shape[0], -1).tolist()

        # 各行の右端2列に空のマスを追加
        for row in _grid:
            row.extend([PRE_IDS['empty'] for _ in range(2)])
            # a = row + emp

        return _grid

    def __init_full_obs(self):
        """
        グリッドや果物などのグローバル状態を初期化
        """
        # エージェントの位置を初期位置に戻す
        self.agent_pos = copy.copy(self.init_agent_pos)
        self.agent_prev_pos = copy.copy(self.init_agent_pos)

        # 果物を配置したグリッドを生成
        # self._full_obs = self.__create_grid()
        self._full_obs = self.__create_random_grid()

        # グリッドに各エージェントを配置
        for agent_i in range(self.n_agents):
            self.__update_agent_view(agent_i)

        # エージェントを除いた画像を生成
        self.__draw_base_img()

    def get_agent_obs(self):
        """
        各エージェントの部分観測を取得
        観測 = [自身の現在位置, 周囲3x3マスの情報(One-hot), 経過時間]
        """
        _obs = []

        # エージェントごと
        for agent_i in range(self.n_agents):
            # 現座地を取得
            pos = self.agent_pos[agent_i]

            # add coordinates
            # 現在地の座標を観測に含める
            # X-Y座標を0〜1（小数点以下第2位まで）で表す
            _agent_i_obs = [round(pos[0] / self._grid_shape[0], 2),
                            round(pos[1] / (self._grid_shape[1] - 1), 2)]

            # add 3 x3 mask around the agent current location and share neighbours
            # ( in practice: this information may not be so critical since the map never changes)
            # エージェントの周囲 3 x 3 マスの情報を観測に含める

            # 3マス x 3マス x 5のOne-hotベクトル(壁, リンゴ, レモン, エージェント1, エージェント2) を定義
            _agent_i_neighbour = np.zeros((3, 3, 5))

            # 行ごと
            for r in range(pos[0] - 1, pos[0] + 2):
                # 列ごと
                for c in range(pos[1] - 1, pos[1] + 2):
                    if self.is_valid((r, c)):  # 位置がグリッド外でないか
                        # One-hotベクトルを生成
                        item = [0, 0, 0, 0, 0]
                        if PRE_IDS['lemon'] in self._full_obs[r][c]:
                            # レモン
                            item[ITEM_ONE_HOT_INDEX['lemon']] = 1
                        elif PRE_IDS['apple'] in self._full_obs[r][c]:
                            # リンゴ
                            item[ITEM_ONE_HOT_INDEX['apple']] = 1
                        elif PRE_IDS['agent'] in self._full_obs[r][c]:
                            # エージェント
                            item[ITEM_ONE_HOT_INDEX[self._full_obs[r][c]]] = 1
                        elif PRE_IDS['wall'] in self._full_obs[r][c]:
                            # 壁
                            item[ITEM_ONE_HOT_INDEX['wall']] = 1

                        # One-hotベクトルを挿入
                        _agent_i_neighbour[r -
                                           (pos[0] - 1)][c - (pos[1] - 1)] = item

            # 3x3x5の行列を1つの長いリストにする
            _agent_i_obs += _agent_i_neighbour.flatten().tolist()

            # adding time
            # 0〜1の範囲で現在経過した時間を追加（タイムステップ）
            if self._add_clock:
                _agent_i_obs += [self._step_count / self._max_steps]

            # リストに自分の観測を追加
            _obs.append(_agent_i_obs)

        # 完全観測可能な場合
        if self.full_observable:
            # 全員分の観測を全員が受け取る
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]

        return _obs

    def reset(self):
        """
        環境を初期化
        エージェントの観測を返す
        """
        # グローバル状態を初期化
        self.__init_full_obs()

        # タイムステップをリセット
        self._step_count = 0
        # 報酬の総和
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        # 残っている果物の数
        self._food_count = {'lemon': ((self._grid_shape[1] - 2) // 2) * self._grid_shape[0],
                            'apple': ((self._grid_shape[1] - 2) // 2) * self._grid_shape[0]}
        # 終了したか
        self._agent_dones = [False for _ in range(self.n_agents)]
        # 最大ステップを超えたか
        self.steps_beyond_done = None

        # 観測
        return self.get_agent_obs()

    def is_valid(self, pos):
        """
        位置がグリッドからはみ出ていないかを確認
        """
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _has_no_agent(self, pos):
        """
        あるマスにエージェントがいないことを確認
        """
        return self.is_valid(pos) and (PRE_IDS['agent'] not in self._full_obs[pos[0]][pos[1]])

    def __update_agent_pos(self, agent_i, move):
        """
        エージェントが行動を出力して移動する
        """
        # 現在位置
        curr_pos = copy.copy(self.agent_pos[agent_i])

        # 次の位置
        if move == 0:  # down
            # 下
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            # 左
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            # 上
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            # 右
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            # 何もしない（No Operation）
            next_pos = None
        else:
            raise Exception('Action Not found!')

        # 1つ前の位置を現在位置で更新
        self.agent_prev_pos[agent_i] = self.agent_pos[agent_i]

        # 次の位置へ移動可能かどうかをチェックして反映
        if next_pos is not None and self._has_no_agent(next_pos):
            self.agent_pos[agent_i] = next_pos

    def __update_agent_view(self, agent_i):
        """
        エージェントの位置をグリッド上に反映
        """
        # 1つ前にエージェントがいた位置を空にする
        self._full_obs[self.agent_prev_pos[agent_i][0]
                       ][self.agent_prev_pos[agent_i][1]] = PRE_IDS['empty']
        # 現在のエージェントの位置にエージェントを配置（"A + エージェント番号"）
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i]
                                                   [1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def step(self, agents_action):
        """
        行動を環境に出力してタイムステップを1つ進める
        [各エージェントの観測, 報酬, 終了フラグ, 追加情報（残り個数）]
        """
        # 人数分の行動が入力されているかチェック
        assert len(agents_action) == self.n_agents

        # タイムステップを進める
        self._step_count += 1
        # 時間経過によるマイナスの報酬
        rewards = [self._step_cost for _ in range(self.n_agents)]

        # エージェントごと
        for agent_i, action in enumerate(agents_action):
            # 行動をとって位置を移動
            self.__update_agent_pos(agent_i, action)

            # 前回と位置が変わっていたら
            if self.agent_pos[agent_i] != self.agent_prev_pos[agent_i]:
                for food in ['lemon', 'apple']:
                    # 移動先に果物があるかを調べる
                    if PRE_IDS[food] in self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]]:
                        # 果物をGETしたことによる報酬を獲得
                        rewards[agent_i] += self.agent_reward[agent_i][food]
                        # 果物の残数を更新
                        self._food_count[food] -= 1
                        break
                # エージェントの移動をグリッドに反映
                self.__update_agent_view(agent_i)

        # 最大ステップ数を超えている or 全てのリンゴをGETしたら
        if self._step_count >= self._max_steps or self._food_count['apple'] == 0:
            # 全エージェントの終了フラグを立てる
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        # 報酬の総和を更新
        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        # Following snippet of code was refereed from:
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L144

        # 終了したにも関わらずstep()を呼ばれた際のエラー用
        if self.steps_beyond_done is None and all(self._agent_dones):
            self.steps_beyond_done = 0
        elif self.steps_beyond_done is not None:
            if self.steps_beyond_done == 0:
                logger.warning(
                    "You are calling 'step()' even though this environment has already returned all(dones) = True for "
                    "all agents. You should always call 'reset()' once you receive 'all(dones) = True' -- any further"
                    " steps are undefined behavior.")
            self.steps_beyond_done += 1
            rewards = [0 for _ in range(self.n_agents)]

        # [各エージェントの観測, 報酬, 終了フラグ, 追加情報（残り個数）]
        return self.get_agent_obs(), rewards, self._agent_dones, {'food_count': self._food_count}

    def render(self, mode='human'):
        """
        画像を描画
        """
        # 既にあるグリッド画像の上にエージェントを乗せる
        for agent_i in range(self.n_agents):
            # 1つ前と現在のマスを白く塗る
            fill_cell(
                self._base_img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill='white', margin=0.05)
            fill_cell(
                self._base_img, self.agent_prev_pos[agent_i], cell_size=CELL_SIZE, fill='white', margin=0.05)
            # 自身の色の円
            draw_circle(
                self._base_img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLORS[agent_i])
            # エージェント番号
            write_cell_text(self._base_img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        # adds a score board on top of the image
        # img = draw_score_board(self._base_img, score=self._total_episode_reward)
        # img = np.asarray(img)

        img = np.asarray(self._base_img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        """
        乱数シードを指定
        """
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        """
        描画のウィンドウを閉じる
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# 1マスの大きさ
CELL_SIZE = 30

# 各行動の数字が持つ意味
ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

# 各観測の数字が持つ意味
OBSERVATION_MEANING = {
    0: 'empty',
    1: 'lemon',
    2: 'apple',
    3: 'agent',
    -1: 'wall'
}

# each pre-id should be unique and single char
# 各物体を表すID
PRE_IDS = {
    'agent': 'A',
    'wall': 'W',
    'empty': '0',
    'lemon': 'Y',  # yellow color
    'apple': 'R',  # red color
}

# 各物体が対応するOneHotベクトルの要素番号
ITEM_ONE_HOT_INDEX = {
    'lemon': 0,
    'apple': 1,
    'A1': 2,
    'A2': 3,
    'wall': 4,
}

# 描画する際の色
AGENT_COLORS = {
    0: 'red',
    1: 'blue'
}
WALL_COLOR = 'black'
LEMON_COLOR = 'yellow'
APPLE_COLOR = 'green'
