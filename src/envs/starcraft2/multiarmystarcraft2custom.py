import math
import numpy as np
from pysc2.lib import protocol
from pysc2.lib.units import Neutral, Protoss, Terran, Zerg
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

from .starcraft2custom import StarCraft2CustomEnv, actions, get_unit_name_by_type

class StarCraft2MultiArmyEnv(StarCraft2CustomEnv):
    def __init__(self, *args, heuristic_alloc=False,
                 heuristic_style='attacking-type-unassigned-diff',
                 spread_armies=True,
                 **kwargs):
        self.spread_armies = spread_armies
        super().__init__(*args, **kwargs)
        assert ('max_army_size' in self.scenario_dict
                and 'max_n_armies' in self.scenario_dict), (
                'Multi-army scenario not selected')
        assert self.pos_ally_centered, (
               "Multi-army assumes allies are in center")
        self.multi_army = True

        assert (math.ceil(self.max_n_enemies / self.max_army_size)
                <= self.max_n_armies), (
                "max_army_size and max_n_armies are incompatible")

        # Enemies won't attack all at once at the start (will be sporadic
        # throughout the episode)
        self.aggressive_enemies = False
        self.heuristic_alloc = heuristic_alloc
        self.heuristic_style = heuristic_style.split('-')
        # Agents will have no idea where to find other armies if sight range is
        # too low
        self._sight_range = 2 * self.pos_separation
        self._min_switch_time = 10
        self._max_switch_time = 20
        self._base_size = 2
        self._army_defeat_health_boost_mult = 4.0

    def _do_switches(self):
        if self._controller.status != protocol.Status.in_game:
            # don't send enemy actions if game is over/crashed
            return
        success = True
        for ar_id in range(self.n_armies):
            army_units = [self.enemies[e_id] for e_id in self.army2enemy[ar_id]]
            alive_army_units = [e_unit for e_unit in army_units if e_unit.health > 0]
            # retreat if only Medivacs left alive
            if len(alive_army_units) > 0 and all(get_unit_name_by_type(e_unit.unit_type).startswith('Medivac') for e_unit in alive_army_units):
                if self._army_is_attacking[ar_id]:
                    success = self._retreat_enemy_army(ar_id)
                    self._army_is_attacking[ar_id] = False
                self._time_since_last_switch[ar_id] = -self._max_switch_time
                continue

            if self._time_to_next_switch[ar_id] == 0:
                if self._army_is_attacking[ar_id]:
                    success = self._retreat_enemy_army(ar_id)
                    self._army_is_attacking[ar_id] = False
                else:
                    success = self._attack_enemy_army(ar_id)
                    self._army_is_attacking[ar_id] = True
                self._time_to_next_switch[ar_id] = self.rs.randint(
                    self._min_switch_time + 1, self._max_switch_time + 1)
                self._time_since_last_switch[ar_id] = -1
        self._time_to_next_switch -= 1
        self._time_since_last_switch += 1
        return success
    
    def try_controller_command(self, fn=lambda: None):
        try:
            fn()
            return True
        except (protocol.ProtocolError, protocol.ConnectionError):
            # self.full_restart()
            return False

    def step(self, *args, **kwargs):
        rets = super().step(*args, **kwargs)
        success = self._do_switches()
        # if not success:
        #     # set terminated to be True
        #     return (rets[0], True, *rets[2:])
        return rets

    def reset(self, unit_override=None, test=False, index=None, n_tasks=None):
        rets = super().reset(unit_override=unit_override, test=test, index=index, n_tasks=n_tasks)
        self.heals_allowed = False
        if self.n_enemies > self.n_agents:
            self.heals_allowed = True
            self.heal_fracs = []
        return rets

    def _retreat_enemy_army(self, ar_id):
        tags = [self.enemies[e_id].tag for e_id in self.army2enemy[ar_id]]
        army_spawn_center = sc_common.Point2D(
            x=self.map_center[0] + self.army_positions[ar_id][0],
            y=self.map_center[1] + self.army_positions[ar_id][1])
        cmd = r_pb.ActionRawUnitCommand(
            ability_id=actions["move"],
            target_world_space_pos=army_spawn_center,
            unit_tags=tags,
            queue_command=False)
        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        req_actions = sc_pb.RequestAction(actions=[sc_action])
        return self.try_controller_command(fn=lambda: self._controller.actions(req_actions))

    def _attack_enemy_army(self, ar_id):
        tags = [self.enemies[e_id].tag for e_id in self.army2enemy[ar_id]]
        map_center = sc_common.Point2D(
            x=self.map_center[0],
            y=self.map_center[1])
        cmd = r_pb.ActionRawUnitCommand(
            ability_id=actions["attack"],
            target_world_space_pos=map_center,
            unit_tags=tags,
            queue_command=False)
        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        req_actions = sc_pb.RequestAction(actions=[sc_action])
        return self.try_controller_command(fn=lambda: self._controller.actions(req_actions))

    def get_masks(self):
        """
        Adds masks useful for task allocation
        1) entity to task matrix (allies are left unassigned unless we're using
            a task allocation heuristic)
        2) task mask vector (what tasks are available)
        """
        masks = super().get_masks()

        entity2task = np.ones((self.max_n_agents + self.max_n_enemies,
                               self.max_n_armies), dtype=np.uint8)
        if self.heuristic_alloc:
            type2armylist = {}
            army2diff = {}
            for e_id, e_unit in self.enemies.items():
                if e_unit.health == 0 and 'dynamic' in self.heuristic_style:
                    continue
                e_type = get_unit_name_by_type(e_unit.unit_type)
                ar_id = self.enemy2army[e_id]
                type2armylist[e_type] = type2armylist.get(e_type, []) + [ar_id]
                army2diff[ar_id] = army2diff.get(ar_id, 0) + 1
            army2count = {ar_id: 0 for ar_id in range(self.n_armies)}
            for al_id in self.shuffled_agent_ids:
                al_unit = self.agents[al_id]
                if al_unit.health == 0 and 'dynamic' in self.heuristic_style:
                    continue
                al_type = get_unit_name_by_type(al_unit.unit_type).split('_RL')[0]
                if al_type not in type2armylist:
                    type2armylist[al_type] = []
                style_fns = {
                    'type': lambda ar_id: ar_id in type2armylist[al_type],
                    'unassigned': lambda ar_id: army2count[ar_id] == 0,
                    'diff': lambda ar_id: army2diff[ar_id],
                    'attacking': lambda ar_id: self._army_is_attacking[ar_id],
                }
                sort_keys = lambda ar_id: tuple(style_fns[style](ar_id) for style in self.heuristic_style if style != 'dynamic')
                army_priority = list(sorted(
                    [ar_id for ar_id in range(self.n_armies) if not self.tasks_complete[ar_id]],
                    key=sort_keys,
                    reverse=True))
                if len(army_priority) == 0:
                    continue
                assign_army_id = army_priority[0]
                try:
                    ind = type2armylist[al_type].index(assign_army_id)
                    type2armylist[al_type].pop(ind)
                except ValueError:
                    pass
                army2diff[assign_army_id] -= 1
                army2count[assign_army_id] += 1
                entity2task[al_id, assign_army_id] = 0
        for ar_id, e_list in self.army2enemy.items():
            e_idxs = np.array(e_list, dtype=int) + self.max_n_agents
            entity2task[e_idxs, ar_id] = 0

        masks['entity2task_mask'] = entity2task
        masks['task_mask'] = np.array(self.tasks_complete, dtype=np.uint8)
        return masks

    def get_entities(self):
        entities = super().get_entities()
        assert len(entities) == self.max_n_agents + self.max_n_enemies

        if self.previous_ally_units is not None:
            for ai in range(self.n_agents):
                curr_t_unit = self.agents[ai]
                prev_t_unit = self.previous_ally_units[ai]
                entities[ai][-2] = (curr_t_unit.pos.x - prev_t_unit.pos.x) / self._move_amount
                entities[ai][-1] = (curr_t_unit.pos.y - prev_t_unit.pos.y) / self._move_amount

        for ei in range(self.n_enemies):
            ar_id = self.enemy2army[ei]
            entities[self.max_n_agents + ei][-4] = self._army_is_attacking[ar_id]
            entities[self.max_n_agents + ei][-3] = self._time_since_last_switch[ar_id] / self._max_switch_time
            if self.previous_enemy_units is not None:
                curr_t_unit = self.enemies[ei]
                prev_t_unit = self.previous_enemy_units[ei]
                entities[self.max_n_agents + ei][-2] = (curr_t_unit.pos.x - prev_t_unit.pos.x) / self._move_amount
                entities[self.max_n_agents + ei][-1] = (curr_t_unit.pos.y - prev_t_unit.pos.y) / self._move_amount
        return entities

    def get_entity_size(self):
        # add dimensions for attack/retreat indicator, time since last switch, and x-y velocity
        return super().get_entity_size() + 4

    def _closest_army_center(self, unit):
        army_dists = [self.distance(unit.pos.x, unit.pos.y,
                      a_x + self.map_center[0],
                      a_y + self.map_center[1])
                      for a_x, a_y in self.army_positions]
        return np.argmin(army_dists)

    def _closest_enemies(self, ar_id):
        a_x, a_y = self.army_positions[ar_id]
        enemy_dists = [self.distance(unit.pos.x, unit.pos.y,
                      a_x + self.map_center[0],
                      a_y + self.map_center[1])
                      for unit in self.enemies.values()]
        return np.argsort(enemy_dists)

    def _identify_armies(self):
        """
        Identify which army each enemy unit belongs to
        """
        self.enemy2army = {}
        self.army2enemy = {}

        for e_id, e_unit in self.enemies.items():
            ar_id = self._closest_army_center(e_unit)
            self.enemy2army[e_id] = ar_id
            self.army2enemy[ar_id] = self.army2enemy.get(ar_id, []) + [e_id]
        if len(self.army2enemy) < self.n_armies:
            # armies spawned too close together and can't be identified
            # properly, simply ensure all armies have at least one unit. This
            # should be *very* rare.
            assert not self.spread_armies, "This should only happen when armies aren't spread out"
            for ar_id in range(self.n_armies):
                if ar_id not in self.army2enemy:
                    sorted_e_ids = self._closest_enemies(ar_id)
                    for e_id in sorted_e_ids:
                        curr_ar_id = self.enemy2army[e_id]
                        if len(self.army2enemy[curr_ar_id]) > 1:
                            self.enemy2army[e_id] = ar_id
                            self.army2enemy[curr_ar_id].pop(self.army2enemy[curr_ar_id].index(e_id))
                            self.army2enemy[ar_id] = [e_id]
                            break

    def _assign_pos(self, scenario, n_armies=None):
        # Splits enemy units randomly into separate armies
        ally_army, enemy_army = scenario
        ally_pos = (0, 0)
        ally_army_wpos = [(num, unit,
                           ally_pos + (self.rs.rand(2) - 0.5)
                           * 2 * self.pos_jitter)
                          for (num, unit) in ally_army]

        if self.pos_rotate:
            theta = self.rs.random() * 2 * np.pi
        else:
            theta = np.pi

        n_enemies = sum(num for num, _ in enemy_army)
        min_armies = min(2, self.max_n_armies)
        curr_min_armies = max(min_armies, math.ceil(n_enemies / self.max_army_size))
        curr_max_armies = min(n_enemies, self.max_n_armies)

        if n_armies is None:
            n_armies = self.rs.randint(curr_min_armies, curr_max_armies + 1)
        else:
            assert n_armies <= curr_max_armies and n_armies >= curr_min_armies

        army_sizes = []
        for i in range(n_armies):
            num_remaining = n_enemies - sum(army_sizes)  # num remaining to be assigned
            min_size = (num_remaining
                        - ((n_armies - i - 1) * self.max_army_size))  # max num that can be assigned after this
            min_size = max(1, min_size)

            # don't exhaust all units before getting to last army
            max_size = min(num_remaining - (n_armies - i - 1), self.max_army_size)
            army_sizes.append(self.rs.randint(min_size, max_size + 1))

        assert sum(army_sizes) == n_enemies, "Army splitting broke"

        enemy_list = sum(
            ([unit for _ in range(num)] for num, unit in enemy_army), [])
        self.rs.shuffle(enemy_list)
        curr_idx = 0
        r = self.pos_separation
        enemy_army_wpos = []
        self.army_positions = []
        for i, army_sz in enumerate(army_sizes):
            if self.spread_armies:
                self.spread_mult = 1
            else:
                self.spread_mult = self.rs.rand()
            curr_theta = theta + (i / n_armies) * 2 * np.pi * self.spread_mult
            army_pos = (r * np.cos(curr_theta), r * np.sin(curr_theta))
            self.army_positions.append(army_pos)

            army_enemies = enemy_list[curr_idx:curr_idx + army_sz]
            curr_idx += army_sz
            for unit in set(army_enemies):
                enemy_army_wpos.append((army_enemies.count(unit),
                                        unit,
                                        army_pos))
        self.n_armies = n_armies
        self.tasks_complete = [0 for _ in range(self.n_armies)] + [1 for _ in range(self.max_n_armies - self.n_armies)]

        return (ally_army_wpos, enemy_army_wpos)

    def reward_armies(self):
        reward = np.zeros(self.max_n_armies)
        if self.reward_sparse:
            return reward
        delta_deaths = np.zeros(self.max_n_armies)
        delta_enemy = np.zeros(self.max_n_armies)

        for e_id, e_unit in self.enemies.items():
            ar_id = self.enemy2army[e_id]
            if not self.death_tracker_enemy[e_id]:
                prev_health = (
                    self.previous_enemy_units[e_id].health
                    + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths[ar_id] += self.reward_death_value
                    delta_enemy[ar_id] += prev_health
                else:
                    delta_enemy[ar_id] += prev_health - e_unit.health - e_unit.shield

        reward = (delta_enemy + delta_deaths)

        return reward

    def task_end_codes(self):
        end_codes = [1 for _ in range(self.max_n_armies)]
        for ar_id, army in self.army2enemy.items():
            n_alive = 0
            n_in_base = 0
            for e_id in army:
                e_unit = self.enemies[e_id]
                n_alive += (e_unit.health > 0)
                if get_unit_name_by_type(e_unit.unit_type).startswith('Medivac'):
                    continue
                dist_to_base = self.distance(e_unit.pos.x, e_unit.pos.y,
                                             self.map_center[0],
                                             self.map_center[1])
                n_in_base += (dist_to_base < self._base_size)
            if n_in_base > 0:
                end_codes[ar_id] = -1
            elif n_alive == 0:
                end_codes[ar_id] = 1
            else:
                end_codes[ar_id] = 0
        return end_codes

    def get_scenario_desc(self):
        scens = []
        utypes = sorted(list(get_unit_name_by_type(utype).split('_RL')[0] for utype in self.unit_types))
        cut_i = 1
        while all(utypes[0][:cut_i] == utype[:cut_i] for utype in utypes):
            cut_i += 1
        short_utypes = [utype[:cut_i] for utype in utypes]

        for ar_id, army_ids in self.army2enemy.items():
            counts = dict((k, 0) for k in short_utypes)
            for eid in army_ids:
                utype = get_unit_name_by_type(self.enemies[eid].unit_type).split('_RL')[0][:cut_i]
                counts[utype] += 1
            scen_str = ''.join(f'{k}{v}' for k, v in counts.items())
            scens.append(scen_str)
        return scens

    def process_end_codes(self, end_codes):
        if not self.heals_allowed:
            # no health boost for symmetric scenarios or scenarios where there
            # are more agents than enemies
            return True
        heal_agents = []
        for ar_id, end_code in enumerate(end_codes):
            if end_code == 1 and not self.tasks_complete[ar_id]:
                # This task was just won, so give health/shield boost to all nearby alive agents
                agents_closest_armies = [(al_id, self._closest_army_center(al_unit)) for al_id, al_unit in self.agents.items() if al_unit.health > 0]
                close_to_ar = [al_id for al_id, closest_ar_id in agents_closest_armies if closest_ar_id == ar_id]
                heal_agents += close_to_ar
                if not all(ec != 0 for ec in end_codes):
                    # only want to count for non-terminal subtasks since healing doesn't matter afterwards
                    self.heal_fracs += [len(heal_agents) / self.n_agents]
        cmds = []
        # health boost inversely proportional to the number of agents and armies.
        # proportional to the unit disadvantage
        boost_frac = (1 / (self.n_agents * max(self.n_armies - 1, 1))) * (self.n_enemies - self.n_agents) * self._army_defeat_health_boost_mult
        for al_id in heal_agents:
            al_unit = self.agents[al_id]
            new_health = min(al_unit.health + al_unit.health_max * boost_frac,
                             al_unit.health_max)
            cmd = d_pb.DebugCommand(
                unit_value=d_pb.DebugSetUnitValue(
                    unit_value=d_pb.DebugSetUnitValue.UnitValue.Life,
                    value=new_health,
                    unit_tag=al_unit.tag
                )
            )
            cmds.append(cmd)
            # new value won't kick in until next game step, so fake it here for
            # now s.t. observations show new value
            self.agents[al_id].health = new_health
            if self.shield_bits_ally > 0:
                new_shield = min(al_unit.shield + al_unit.shield_max * boost_frac,
                                 al_unit.shield_max)
                cmd = d_pb.DebugCommand(
                    unit_value=d_pb.DebugSetUnitValue(
                        unit_value=d_pb.DebugSetUnitValue.UnitValue.Shields,
                        value=new_shield,
                        unit_tag=al_unit.tag
                    )
                )
                cmds.append(cmd)
                self.agents[al_id].shield = new_shield
        if self._controller.status == protocol.Status.in_game and len(cmds) > 0:
            # only send debug commands if game is still running, otherwise will
            # cause whole process to crash. Next step() or reset() will do full
            # reset of game
            return self.try_controller_command(fn=lambda: self._controller.debug(cmds))
        return True

    def get_env_info(self, args):
        env_info = super().get_env_info(args)
        env_info['n_tasks'] = self.max_n_armies
        return env_info
