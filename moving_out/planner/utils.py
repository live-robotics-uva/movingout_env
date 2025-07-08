def convert_walls(walls):
    converted_walls = []
    for wall in walls:
        p1, p2 = wall
        left_bottom = [min(p1[0], p2[0]), min(p1[1], p2[1])]
        right_top = [max(p1[0], p2[0]), max(p1[1], p2[1])]
        converted_walls.append(
            [left_bottom[0], right_top[0], left_bottom[1], right_top[1]]
        )
    return converted_walls


def convert_goal_region(goal_region):
    converted_region = []
    for wall in goal_region:
        p1, p2 = wall
        left_bottom = [min(p1[0], p2[0]), min(p1[1], p2[1])]
        right_top = [max(p1[0], p2[0]), max(p1[1], p2[1])]
        converted_region.append(
            [left_bottom[0], right_top[0], left_bottom[1], right_top[1]]
        )
    return converted_region
