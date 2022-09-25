from numpy.random import RandomState
from functools import partial
from itertools import combinations_with_replacement, product
from copy import deepcopy


def get_all_unique_teams(all_types, min_len, max_len):
    all_uniq = []
    for i in range(min_len, max_len + 1):
        all_uniq += list(combinations_with_replacement(all_types, i))
    all_uniq_counts = []
    for scen in all_uniq:
        curr_uniq = list(set(scen))
        uniq_counts = list(zip([scen.count(u) for u in curr_uniq], curr_uniq))
        all_uniq_counts.append(uniq_counts)
    return all_uniq_counts


def fixed_armies(ally_army, enemy_army, ally_centered=False, rotate=False,
                 separation=10, jitter=0, episode_limit=100,
                 map_name="empty_passive", rs=None):
    scenario_dict = {'scenarios': [(ally_army, enemy_army)],
                     'max_types_and_units_scenario': (ally_army, enemy_army),
                     'ally_centered': ally_centered,
                     'rotate': rotate,
                     'separation': separation,
                     'jitter': jitter,
                     'episode_limit': episode_limit,
                     'map_name': map_name}
    return scenario_dict


def symmetric_armies(army_spec, ally_centered=False,
                     rotate=False, separation=10,
                     jitter=0, episode_limit=100, map_name="empty_passive",
                     rs=None):
    if rs is None:
        rs = RandomState()

    unique_sub_teams = []
    for unit_types, n_unit_range in army_spec:
        unique_sub_teams.append(get_all_unique_teams(unit_types, n_unit_range[0],
                                                     n_unit_range[1]))
    unique_teams = [sum(prod, []) for prod in product(*unique_sub_teams)]

    scenarios = list(zip(unique_teams, unique_teams))
    # sort by number of types and total number of units
    max_types_and_units_team = sorted(unique_teams, key=lambda x: (len(x), sum(num for num, unit in x)), reverse=True)[0]
    max_types_and_units_scenario = (max_types_and_units_team,
                                    max_types_and_units_team)

    scenario_dict = {'scenarios': scenarios,
                     'max_types_and_units_scenario': max_types_and_units_scenario,
                     'ally_centered': ally_centered,
                     'rotate': rotate,
                     'separation': separation,
                     'jitter': jitter,
                     'episode_limit': episode_limit,
                     'map_name': map_name}
    return scenario_dict


def combine_teams(team_a, team_b):
    type2num_dict = {}
    for num, unit_type in team_a:
        type2num_dict[unit_type] = type2num_dict.get(unit_type, 0) + num
    for num, unit_type in team_b:
        type2num_dict[unit_type] = type2num_dict.get(unit_type, 0) + num
    return [(num, unit_type) for unit_type, num in type2num_dict.items()]


def asymm_armies(army_spec, spec_delta, ally_centered=False,
                 rotate=False, separation=10,
                 jitter=0, episode_limit=100, map_name="empty_passive",
                 rs=None):
    if rs is None:
        rs = RandomState()

    unique_sub_teams = []
    for unit_types, n_unit_range in army_spec:
        unique_sub_teams.append(get_all_unique_teams(unit_types, n_unit_range[0],
                                                     n_unit_range[1]))
    enemy_teams = [sum(prod, []) for prod in product(*unique_sub_teams)]
    agent_teams = deepcopy(enemy_teams)
    # add extra units to enemy team
    delta_teams = get_all_unique_teams(spec_delta[0], spec_delta[1][0], spec_delta[1][1])
    enemy_teams = [combine_teams(e_team, delta_teams[rs.choice(len(delta_teams))]) for e_team in enemy_teams]

    scenarios = list(zip(agent_teams, enemy_teams))
    # sort by number of types and total number of units
    max_types_and_units_ag_team = sorted(agent_teams, key=lambda x: (len(x), sum(num for num, unit in x)), reverse=True)[0]
    max_types_and_units_en_team = sorted(enemy_teams, key=lambda x: (len(x), sum(num for num, unit in x)), reverse=True)[0]
    max_types_and_units_scenario = (max_types_and_units_ag_team,
                                    max_types_and_units_en_team)

    scenario_dict = {'scenarios': scenarios,
                     'max_types_and_units_scenario': max_types_and_units_scenario,
                     'ally_centered': ally_centered,
                     'rotate': rotate,
                     'separation': separation,
                     'jitter': jitter,
                     'episode_limit': episode_limit,
                     'map_name': map_name}
    return scenario_dict


def multi_armies(scenario_gen, max_army_size=3, max_n_armies=3, train_n_enemies=None, train_n_armies=None, rs=None):
    """
    Split enemy army into multiple armies (which will be spawned in different locations)
    """
    scenario_dict = scenario_gen(rs=rs)
    scenario_dict['max_army_size'] = max_army_size
    scenario_dict['max_n_armies'] = max_n_armies
    scenario_dict['train_n_enemies'] = train_n_enemies
    scenario_dict['train_n_armies'] = train_n_armies
    return scenario_dict


custom_scenario_registry = {
    "3-8m_symmetric": partial(symmetric_armies,
                            [(('Marine',), (3, 8))],
                            rotate=True,
                            ally_centered=False,
                            separation=14,
                            jitter=1, episode_limit=100, map_name="empty_passive"),
    "3-8sz_symmetric": partial(symmetric_armies,
                             [(('Stalker', 'Zealot'), (3, 8))],
                             rotate=True,
                             ally_centered=False,
                             separation=14,
                             jitter=1, episode_limit=150, map_name="empty_passive"),
    "3-8MMM_symmetric": partial(symmetric_armies,
                              [(('Marine', 'Marauder'), (3, 6)),
                               (('Medivac',), (0, 2))],
                              rotate=True,
                              ally_centered=False,
                              separation=14,
                              jitter=1, episode_limit=150, map_name="empty_passive"),
    "3-8csz_symmetric": partial(symmetric_armies,
                              [(('Stalker', 'Zealot'), (3, 6)),
                               (('Colossus',), (0, 2))],
                              rotate=True,
                              ally_centered=False,
                              separation=14,
                              jitter=1, episode_limit=150, map_name="empty_passive"),
    ########## MULTI-ARMY SCENARIOS ##########
    "6-8sz_maxsize4_maxarmies3_symmetric": partial(
        multi_armies,
        partial(symmetric_armies,
                [(('Stalker', 'Zealot'), (3, 8))],
                rotate=True,
                ally_centered=True,
                separation=14,
                jitter=1, episode_limit=150, map_name="empty_passive"),
        max_army_size=4,
        max_n_armies=3,
    ),
    "6-8sz_maxsize4_maxarmies3_unitdisadvantage": partial(
        multi_armies,
        partial(asymm_armies,
                [(('Stalker', 'Zealot'), (3, 8))],
                (('Stalker', 'Zealot'), (1, 1)),
                rotate=True,
                ally_centered=True,
                separation=14,
                jitter=1, episode_limit=150, map_name="empty_passive"),
        max_army_size=4,
        max_n_armies=3,
    ),
    "6-8MMM_maxsize4_maxarmies3_symmetric": partial(
        multi_armies,
        partial(symmetric_armies,
                [(('Marine', 'Marauder'), (3, 6)),
                 (('Medivac',), (0, 2))],
                rotate=True,
                ally_centered=True,
                separation=14,
                jitter=1, episode_limit=150, map_name="empty_passive"),
        max_army_size=4,
        max_n_armies=3,
    ),
    "6-8MMM_maxsize4_maxarmies3_unitdisadvantage": partial(
        multi_armies,
        partial(asymm_armies,
                [(('Marine', 'Marauder'), (3, 6)),
                 (('Medivac',), (0, 2))],
                (('Marine', 'Marauder'), (1, 1)),
                rotate=True,
                ally_centered=True,
                separation=14,
                jitter=1, episode_limit=150, map_name="empty_passive"),
        max_army_size=4,
        max_n_armies=3,
    ),
}
