import numpy as np

from multiworld.envs.mujoco.classic_mujoco.ant import AntEnv
from collections import OrderedDict
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path


class AntMazeEnv(AntEnv):

    def _sample_uniform_xy(self, batch_size):
        valid_goals = []
        while len(valid_goals) < batch_size:
            goal = np.random.uniform(
                self.goal_space.low[:2],
                self.goal_space.high[:2],
                size=(2),
            )
            valid = not (
                intersect(1.5, -5.5, 4, 2.5, goal[0], goal[1]) or
                intersect(-4, -2.5, -1.5, 5.5, goal[0], goal[1])
            )
            if valid:
                valid_goals.append(goal)
        return np.r_[valid_goals]

    def step(self, action):
        ob, reward, done, info = super(AntMazeEnv, self).step(action)
        if self.diagnostics_goal is not None:
            info['manhattan_dist_to_target'] = self._get_manhattan_distance(
                ob['xy_achieved_goal'], self.diagnostics_goal
            )
        return ob, reward, done, info

    def _get_manhattan_distance(self, s1, s2, invert_y=True):
        # Maze wall positions
        left_wall_x = -2.75
        left_wall_bottom = 2.5
        right_wall_x = 2.75
        right_wall_top = -2.5
        
        s1 = s1.copy()
        s2 = s2.copy()

        if len(s1.shape) == 1:
            s1 = s1[None]
        if len(s2.shape) == 1:
            s2 = np.repeat(s2[None], len(s1), axis=0)

        if invert_y:
            s1[:,1] *= -1
            s2[:,1] *= -1
        
        # Since maze distances are symmetric, redefine s1,s2 for convenience 
        # so that points in s1 are to the left of those in s2
        combined = np.hstack((s1[:,None], s2[:,None]))
        indices = np.argmin((s1[:,0], s2[:,0]), axis=0)
        s1 = np.take_along_axis(combined, indices[:,None,None], axis=1).squeeze(axis=1)
        s2 = np.take_along_axis(combined, 1 - indices[:,None,None], axis=1).squeeze(axis=1)
        
        x1 = s1[:,0]
        x2 = s2[:,0]
        
        # Horizontal movement
        x_dist = np.abs(x2 - x1)
        
        # Vertical movement
        boundary_ys = [left_wall_bottom, right_wall_top, 0]
        boundary_xs = [left_wall_x, right_wall_x, 7, -7.0001]
        y_directions = [1, -1, 0] # +1 means need to get to bottom, -1 means need to get to top
        curr_y, goal_y = s1[:,1], s2[:,1]
        y_dist = np.zeros(len(s1))
        
        for i in range(3):
            # Get all points where s1 and s2 respectively are in the current vertical section of the maze
            curr_in_section = x1 <= boundary_xs[i]
            goal_in_section = np.logical_and(boundary_xs[i-1] < x2, x2 <= boundary_xs[i])
            goal_after_section = x2 > boundary_xs[i]
            
            # Both in current section: move directly to goal
            mask = np.logical_and(curr_in_section, goal_in_section)
            y_dist += mask * np.abs(curr_y - goal_y)
            curr_y[mask] = goal_y[mask]
            
            # s2 is further in maze: move to next corner
            mask = np.logical_and(curr_in_section, np.logical_and(goal_after_section, y_directions[i] * (boundary_ys[i] - curr_y) > 0))
            y_dist += mask * np.clip(y_directions[i] * (boundary_ys[i] - curr_y), 0, None)
            curr_y[mask] = boundary_ys[i]
            
        return x_dist + y_dist

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'xy-distance',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        return statistics


def intersect(x1, y1, x2, y2, candidate_x, candidate_y):
    return x1 < candidate_x < x2 and y1 < candidate_y < y2


if __name__ == '__main__':
    env = AntMazeEnv(
        goal_low=[-4, -4],
        goal_high=[4, 4],
        goal_is_xy=True,
        reward_type='xy_dense',
    )
    import gym
    from multiworld.envs.mujoco import register_custom_envs
    register_custom_envs()
    env = gym.make('AntMaze150RandomInitEnv-v0')
    # env = gym.make('AntCrossMaze150Env-v0')
    # env = gym.make('DebugAntMaze30BottomLeftRandomInitGoalsPreset1Env-v0')
    env = gym.make(
        # 'AntMaze30RandomInitFS20Env-v0',
        # 'AntMaze30RandomInitEnv-v0',
        # 'AntMazeSmall30RandomInitFS10Env-v0',
        # 'AntMazeSmall30RandomInitFs5Dt3Env-v0',
        # 'AntMaze30RandomInitNoVelEnv-v0',
        # 'AntMaze30StateEnv-v0',
        # 'AntMaze30QposRandomInitFS20Env-v0',
        # 'AntMazeSmall30RandomInitFs10Dt3Env-v0',
        # 'AntMazeQposRewSmall30RandomInitFs5Dt3Env-v0',
        # 'AntMazeXyRewSmall30RandomInitFs5Dt3Env-v0',
        # 'AntMazeQposRewSmall30Fs5Dt3Env-v0',
        # 'AntMazeQposRewSmall30Fs5Dt3NoTermEnv-v0',
        # 'AntMazeXyRewSmall30RandomInitFs5Dt3NoTermEnv-v0',
        # 'AntMazeXyRewSmall30Fs5Dt3NoTermEnv-v0',
        'AntMazeQposRewSmall30Fs5Dt3NoTermEnv-v0',
    )
    env.reset()
    i = 0
    while True:
        i += 1
        env.render()
        action = env.action_space.sample()
        # action = np.zeros_like(action)
        obs, reward, done, info = env.step(action)
        # print(reward, np.linalg.norm(env.sim.data.get_body_xpos('torso')[:2]
        #                              - env._xy_goal) )
        # print(env.sim.data.qpos)
        print(info)
        if i % 5 == 0:
            env.reset()
