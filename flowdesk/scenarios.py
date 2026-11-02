"""Built-in geometries. Obstacles are closed solids, including at open outer edges."""
import numpy as np

NOZZLE_SCENARIO = "de_laval"
NOZZLE_LABEL = "Сопло Лаваля"

PROFILE_SCENARIO = "elliptic_profile"
PROFILE_LABEL = "Продувка · несимметричный эллиптический профиль"

SCENARIOS = {
    "channel": "Прямой канал",
    "step": "Обратная ступенька",
    "contraction": "Сужение канала",
    "expansion": "Симметричное расширение",
    "cylinder": "Обтекание цилиндра",
    "baffles": "Канал с перегородками",
}


def geometry(config):
    x = (np.arange(config.nx)+.5)/config.nx
    y = (np.arange(config.ny)+.5)/config.ny
    x, y = np.meshgrid(x, y)
    solid = np.zeros(x.shape, bool)
    if config.scenario == "step":
        solid = (y < .5) & (x < .2)
    elif config.scenario == "contraction":
        solid = (x > .4) & ((y < .25) | (y > .75))
    elif config.scenario == "expansion":
        solid = (x < .25) & ((y < .25) | (y > .75))
    elif config.scenario == "cylinder":
        # Physical circle, radius H/6, rather than a circle in normalized coordinates.
        solid = ((x-.3)*config.length)**2+((y-.5)*config.height)**2 < (config.height/6)**2
    elif config.scenario == "baffles":
        solid = (((x > .28) & (x < .34) & (y < .65)) |
                 ((x > .60) & (x < .66) & (y > .35)))
    elif config.scenario == NOZZLE_SCENARIO:
        from .nozzle import height_at
        half_height = height_at(config, x*config.length)/2
        solid = abs((y-.5)*config.height) > half_height
    return ~solid
