# below are the names and the coordinates of the points
# copied from the original MesoNet code:
#
# (c) 2019 Forys, Xiao, and Murphy lab, CC-BY 4.0
#

LANDMARK_IDS = (
    'L1',
    'L2',
    'L3',
    'M4',
    'M5',
    'M6',
    'R7',
    'R8',
    'R9',
)

LANDMARK_NAMES = {
    'L1': 'left',
    'L2': 'top left',
    'L3': 'bottom left',
    'M4': 'top center',
    'M5': 'bregma',
    'M6': 'lambda',
    'R7': 'right',
    'R8': 'top right',
    'R9': 'bottom right',
}

LANDMARK_COORDS_512 = {
    'L1': (102, 148),  # 'left' in the GUI
    'L2': (166, 88),  # 'top left' in the GUI; the left edge of the root of the left OB
    'L3': (214, 454),  # 'bottom left' in the GUI
    'M4': (256, 88),  # 'top center' in the GUI; the midline at the root of OBs
    'M5': (256, 256),  # presumable bregma
    'M6': (256, 428),  # presumable lambda
    'R7': (410, 148),  # 'right' in the GUI; the right counterpart of L1
    'R8': (346, 88),  # 'top right' in the GUI; the right edge of the root of the right OB
    'R9': (298, 454),  # 'bottom right' in the GUI; the right counterpart of L3
}

LEFT_LANDMARK_IDS = (
    'L1', 'L2', 'L3',
)
LEFT_LANDMARK_NAMES = tuple(
    LANDMARK_NAMES[i] for i in LEFT_LANDMARK_IDS
)

MIDDLE_LANDMARK_IDS = (
    'M4', 'M5', 'M6',
)
MIDDLE_LANDMARK_NAMES = tuple(
    LANDMARK_NAMES[i] for i in MIDDLE_LANDMARK_IDS
)

RIGHT_LANDMARK_IDS = (
    'R7', 'R8', 'R9',
)
RIGHT_LANDMARK_NAMES = tuple(
    LANDMARK_NAMES[i] for i in RIGHT_LANDMARK_IDS
)
