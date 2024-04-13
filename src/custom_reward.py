import gymnasium

class CustomRewardWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        self.x_position_last = 0
        self.time_last = 0

    def get_x_position(self):
        return self.ram[0x6d] * 0x100 + self.ram[0x86]

    def get_x_reward(self):
        _reward = self.get_x_position() - self.x_position_last
        self.x_position_last = self.get_x_position()
        if _reward < -5 or _reward > 5:
            return 0
        return _reward

    def read_mem_range(self, address, length):
        return int(''.join(map(str, self.ram[address:address + length])))

    def get_time(self):
        return self.read_mem_range(0x07f8, 3)

    def get_time_penalty(self):
        _reward = self.get_time() - self.time_last
        self.time_last = self.get_time()
        return _reward

    def player_state(self):
        return self.ram[0x000e]

    def y_viewport(self):
        return self.ram[0x00b5]

    def is_dying(self):
        return self.player_state() == 0x0b or self.y_viewport() > 1

    def is_dead(self):
        return self.player_state() == 0x06

    def get_death_penalty(self):
        if self.is_dying() or self.is_dead():
            return -25
        return 0

    def custom_reward_1(self):
        return (4/9)*self.get_x_reward() + (1/9)*self.get_time_penalty() + (4/9)*self.get_death_penalty()

    def custom_reward_2(self):
        return self.get_x_reward() * (self.get_time()/100) + 100*self.get_death_penalty()

    def step(self, action):
        observation, _, terminated, truncated, info = self.env.step(action)
        custom_reward = self.custom_reward()
        return observation, custom_reward, terminated, truncated, info
