import random

random.seed(4517)

WALL = "cobblestone"
FOOT = "mossy_cobblestone"
TRIM = "stripped_spruce_log"
POST = "spruce_log"
PLANK = "spruce_planks"
ROOFM = "dark_oak"
AIR = "air"
SLAB = "cobblestone_slab"
CSTAIR = "cobblestone_stairs"
CWALL = "cobblestone_wall"
LEAF = "spruce_door"
LANT = "lantern[hanging=true]"


def moss(x0, y0, z0, x1, y1, z1, p=0.15):
    """Speckle the four vertical faces of a box with mossy cobble."""
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            for z in (z0, z1):
                if random.random() < p:
                    place_block(x, y, z, FOOT)
        for z in range(z0 + 1, z1):
            for x in (x0, x1):
                if random.random() < p:
                    place_block(x, y, z, FOOT)


def box(x0, y0, z0, x1, y1, z1, t, block=WALL):
    """Massing's shell(): solid box, then hollowed above its floor course."""
    place_cuboid(x0, y0, z0, x1, y1, z1, block)
    if x1 - x0 > 2 * t and z1 - z0 > 2 * t and y1 > y0:
        place_cuboid(x0 + t, y0 + 1, z0 + t, x1 - t, y1, z1 - t, AIR)


def edge_cells(x0, z0, x1, z1):
    out = []
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            if x in (x0, x1) or z in (z0, z1):
                out.append((x, z))
    return out


def coping(x0, z0, x1, z1, y, block=SLAB):
    for x, z in edge_cells(x0, z0, x1, z1):
        place_block(x, y, z, block)


# ================================================================== ridge_lookout --
# battered tower on the high east shelf
# ==================================================================
reserve(1674, -1458, 1688, -1444, "ridge_lookout")
clear_trees(1672, -1460, 1690, -1442, margin=3)
clear_ground_cover(1672, -1460, 1690, -1442)

LF = floor_from_threshold("ridge_lookout")
Y0 = LF["floor_y"]          # 111 -- the yard, off the lane
B = Y0 + 1                  # 112 -- the massing's `base`

TX0, TX1 = 1676, 1686
TZ0, TZ1 = -1454, -1444

F1 = B + 2                  # 114  stage-one floor course
F2 = B + 8                  # 120  shaft floor course
F3 = B + 16                 # 128  watch-room floor course
RY = B + 21                 # 133  eaves of the hip

# --- plinth: splayed bottom course, then three courses of footing ---
foundation_to_grade(TX0, TZ0, TX1, TZ1, B - 1, FOOT, skirt=1)
place_cuboid(TX0 - 1, B - 1, TZ0 - 1, TX1 + 1, B - 1, TZ1 + 1, FOOT)
place_cuboid(TX0, B - 1, TZ0, TX1, B + 1, TZ1, WALL)
moss(TX0, B - 1, TZ0, TX1, B + 1, TZ1, 0.34)

# --- battered first stage, walls two thick ---
box(1677, F1, -1453, 1685, B + 7, -1445, 2)
moss(1677, F1, -1453, 1685, B + 7, -1445, 0.16)
coping(TX0, TZ0, TX1, TZ1, F1)                       # drip course on the footing

# --- the shaft: narrow, blind, most of the height ---
box(1678, F2, -1452, 1684, B + 15, -1446, 1)
moss(1678, F2, -1452, 1684, B + 15, -1446, 0.14)
coping(1677, -1453, 1685, -1445, F2)                 # weathering at the set-back

# --- jetty corbels under the framed top storey ---
for x, z in edge_cells(1677, -1453, 1685, -1445):
    if x == 1677:
        place_block(x, F3 - 1, z, CSTAIR + "[facing=west,half=top]")
    elif x == 1685:
        place_block(x, F3 - 1, z, CSTAIR + "[facing=east,half=top]")
    elif z == -1453:
        place_block(x, F3 - 1, z, CSTAIR + "[facing=north,half=top]")
    else:
        place_block(x, F3 - 1, z, CSTAIR + "[facing=south,half=top]")

# --- the framed watch room, jettied a block out over the shaft ---
box(1677, F3, -1453, 1685, B + 20, -1445, 1)
moss(1677, F3, -1453, 1685, B + 20, -1445, 0.12)

# open it up between spruce posts: a belvedere, not a blind box
XPOST = (1677, 1679, 1681, 1683, 1685)
ZPOST = (-1453, -1451, -1449, -1447, -1445)
for x, z in edge_cells(1677, -1453, 1685, -1445):
    on_zface = z in (-1453, -1445)
    keep = (x in XPOST) if on_zface else (z in ZPOST)
    for y in (F3 + 2, F3 + 3):
        if keep:
            place_block(x, y, z, POST + "[axis=y]")
        else:
            place_block(x, y, z, AIR)
    place_block(x, F3 + 4, z, TRIM + ("[axis=x]" if on_zface else "[axis=z]"))

# tie beam across the watch room, and the roof over it
place_cuboid(1677, F3 + 4, -1449, 1685, F3 + 4, -1449, POST + "[axis=x]")
ridge_y = roof(1677, -1453, 1685, -1445, RY, ROOFM, "hip", "z", (2, 1), 1)
place_block(1681, ridge_y + 1, -1449, "lantern")

# --- arrow loops: deep, small, and only where they are wanted ---
for y in (B + 5, B + 6):
    place_block(1677, y, -1450, AIR)          # west: clear of the stair
    place_block(1678, y, -1450, AIR)
    for z in (-1450, -1448):
        place_block(1684, y, z, AIR)
        place_block(1685, y, z, AIR)
for x in (1680, 1682):
    for y in (B + 5, B + 6):
        place_block(x, y, -1446, AIR)
        place_block(x, y, -1445, AIR)
for y in (B + 11, B + 12):
    place_block(1684, y, -1449, AIR)
    place_block(1681, y, -1452, AIR)
    place_block(1681, y, -1446, AIR)

# ------------------------------------------------------------------ the outshot: a low
# walled yard hung on the north face
# ------------------------------------------------------------------
YX0, YX1 = 1675, 1686
YZ0, YZ1 = -1458, -1455

foundation_to_grade(YX0 - 1, YZ0, YX1, YZ1, Y0 + 1, FOOT, skirt=0)
place_cuboid(YX0 - 1, Y0, YZ0, YX1, Y0, YZ1, WALL)     # doorstep and yard
moss(YX0 - 1, Y0, YZ0, YX1, Y0, YZ1, 0.3)
place_cuboid(YX0 - 1, Y0 + 1, YZ0, YX1, Y0 + 3, YZ1, AIR)

# perimeter: mossy base course, cobble above, a wall coping
for x, z in edge_cells(YX0, YZ0, YX1, YZ1):
    if z == YZ1:
        continue                                     # open to the tower plinth
    place_block(x, Y0 + 1, z, FOOT)
    place_cuboid(x, Y0 + 2, z, x, Y0 + 3, z, WALL)
    place_block(x, Y0 + 4, z, CWALL)
for x in range(YX0, YX1 + 1, 4):
    place_cuboid(x, Y0 + 1, YZ0, x, Y0 + 4, YZ0, TRIM + "[axis=y]")
place_cuboid(YX0, Y0 + 1, YZ1, YX0, Y0 + 4, YZ1, WALL)
place_cuboid(YX1, Y0 + 1, YZ1, YX1, Y0 + 4, YZ1, WALL)

# the gate off the lane, in the west wall
place_cuboid(YX0, Y0 + 1, -1457, YX0, Y0 + 3, -1457, AIR)
GATE = doorway(YX0, Y0 + 1, -1457, "east", "cobblestone", leaf=LEAF, lintel=True)
place_block(YX0, Y0 + 4, -1457, TRIM + "[axis=z]")

# a small roofed bay in the west of that wall, open to the yard
BAX0, BAX1 = 1677, 1681
place_cuboid(BAX0, Y0 + 1, YZ0, BAX1, Y0 + 5, -1456, WALL)
place_cuboid(BAX0, Y0 + 1, YZ0, BAX1, Y0 + 1, -1456, FOOT)
place_cuboid(BAX0, Y0 + 6, YZ0, BAX1, Y0 + 6, -1457, WALL)
moss(BAX0, Y0 + 1, YZ0, BAX1, Y0 + 5, -1456, 0.18)
place_cuboid(BAX0 + 1, Y0 + 1, -1457, BAX1 - 1, Y0 + 6, YZ1, AIR)
place_cuboid(BAX0, Y0 + 1, YZ1, BAX1, Y0 + 4, YZ1, AIR)   # the front stays open
roof(BAX0, YZ0, BAX1, YZ1, Y0 + 4, ROOFM, "shed", "s", (1, 2), 1)
place_block(1679, Y0 + 4, -1456, LANT)

# --- up the plinth: a flight in the yard, then a terrace at the door ---
place_cuboid(1681, F1, -1454, 1686, F1, -1454, WALL)
moss(1681, F1, -1454, 1686, F1, -1454, 0.3)
place_block(1686, F1 + 1, -1454, CWALL)
place_block(1685, F1 + 1, -1454, CWALL)
steps([(1682, Y0 + 1, -1455), (1683, Y0 + 2, -1455), (1684, Y0 + 3, -1455)],
      "cobblestone", axis="x", prefer="east")

place_cuboid(1683, F1 + 1, -1453, 1683, F1 + 2, -1452, AIR)
TDOOR = doorway(1683, F1 + 1, -1453, "south", "cobblestone", leaf=LEAF,
                lintel=True)
for x in (1682, 1683, 1684):
    place_block(x, F1 + 3, -1454, ROOFM + "_slab[type=bottom]")


# --- two doglegs up the tower: store to shaft, shaft to watch room --- each flight is
# straight, hugs a wall, and turns on a masonry landing
place_block(1679, F1 + 3, -1448, WALL)                      # landing
place_cuboid(1681, F1 + 3, -1448, 1682, F1 + 4, -1448, WALL)  # carriage
place_block(1682, F1 + 5, -1448, WALL)
steps([(1679, F1 + 1, -1451), (1679, F1 + 2, -1450), (1679, F1 + 3, -1449)],
      "cobblestone", axis="z", prefer="south")
steps([(1680, F1 + 4, -1448), (1681, F1 + 5, -1448), (1682, F2, -1448)],
      "cobblestone", axis="x", prefer="east")
place_block(1680, F2, -1448, AIR)
place_block(1681, F2, -1448, AIR)

place_block(1679, F2 + 4, -1447, WALL)                      # landing
steps([(1679, F2 + 1, -1451), (1679, F2 + 2, -1450), (1679, F2 + 3, -1449),
       (1679, F2 + 4, -1448)], "cobblestone", axis="z", prefer="south")
steps([(1680, F2 + 5, -1447), (1681, F2 + 6, -1447), (1682, F2 + 7, -1447),
       (1683, F3, -1447)], "cobblestone", axis="x", prefer="east")
place_block(1681, F3, -1447, AIR)
place_block(1682, F3, -1447, AIR)

# --- what is in the tower ---
fitting("store", 1683, F1 + 1, -1450, "west", mat="spruce", extent=3)
fitting("workbench", 1682, F1 + 1, -1447, "north", mat="spruce")
place_block(1681, B + 7, -1449, LANT)
place_block(1682, F3 - 1, -1449, LANT)

fitting("bed", 1683, F3 + 1, -1451, "west", mat="spruce")
fitting("table", 1679, F3 + 1, -1450, "east", mat="spruce")
fitting("store", 1683, F3 + 1, -1446, "north", mat="spruce", extent=2)
place_block(1681, F3, -1449, WALL)
place_block(1681, F3 + 1, -1449, "campfire[lit=true]")
place_block(1680, F3 + 3, -1449, LANT)
place_block(1682, F3 + 3, -1449, LANT)

# the yard itself: a trough for stock off the drove road, fodder in the bay
fitting("trough", 1684, Y0 + 1, -1456, "west", mat="cobblestone")
fitting("store", 1678, Y0 + 1, -1457, "south", mat="spruce", extent=2)


# ================================================================== threshing_barn --
# one long volume, opposed cart doors, west outshot
# ==================================================================
reserve(1602, -1412, 1626, -1400, "threshing_barn")
clear_trees(1600, -1414, 1628, -1397, margin=3)
clear_ground_cover(1600, -1414, 1628, -1397)

BF = floor_from_threshold("threshing_barn")
FN = BF["floor_y"]          # 102 -- upper (north) threshing floor
FS = FN - 1                 # 101 -- lower floor, one whole step down
EAVES = FN + 7              # 109 -- single eaves line over both

BX0, BX1 = 1607, 1625
BZ0, BZ1 = -1411, -1401
ZMID = -1406

foundation_to_grade(BX0, BZ0, BX1, BZ1, FS, FOOT, skirt=1)
place_cuboid(BX0, FS, BZ0, BX1, EAVES, BZ1, WALL)
place_cuboid(BX0, FS, BZ0, BX1, FS + 1, BZ1, FOOT)
place_cuboid(BX0 + 1, FN + 1, BZ0 + 1, BX1 - 1, EAVES, ZMID, AIR)
place_cuboid(BX0 + 1, FS + 1, ZMID + 1, BX1 - 1, EAVES, BZ1 - 1, AIR)
moss(BX0, FS + 2, BZ0, BX1, EAVES, BZ1, 0.15)

# the fall taken as one step in the floor, not a cut pad
for x in range(BX0 + 1, BX1):
    place_block(x, FN, ZMID + 1, SLAB + "[type=bottom]")

# cart-high opposed doors, mid bay, one in each long wall
place_cuboid(1608, FN + 1, BZ0, 1612, FN + 6, BZ0, AIR)
place_cuboid(1608, FS + 1, BZ1, 1612, FS + 6, BZ1, AIR)
for x in (1607, 1613):
    place_cuboid(x, FN + 1, BZ0, x, FN + 6, BZ0, TRIM + "[axis=y]")
    place_cuboid(x, FS + 1, BZ1, x, FS + 6, BZ1, TRIM + "[axis=y]")
place_cuboid(1607, FN + 7, BZ0, 1613, FN + 7, BZ0, POST + "[axis=x]")
place_cuboid(1607, FS + 7, BZ1, 1613, FS + 7, BZ1, POST + "[axis=x]")

# small high openings elsewhere in the long walls
openings(1615, FN, BZ0, 1619, BZ0, spacing=4, width=1, sill=5, head=6)
openings(1615, FS, BZ1, 1624, BZ1, spacing=4, width=1, sill=4, head=6)
openings(BX1, FS, -1404, BX1, -1402, spacing=3, width=1, sill=4, head=6)

# roof trusses, then the big gable
for x in (1610, 1614, 1618, 1622):
    place_cuboid(x, EAVES, BZ0, x, EAVES, BZ1, POST + "[axis=z]")
roof(BX0, BZ0, BX1, BZ1, EAVES + 1, ROOFM, "gable", "z", (1, 1), 1)
place_block(1614, EAVES - 1, -1408, LANT)
place_block(1618, EAVES - 1, -1404, LANT)

# --- outshot: implement shed down the west end, shallower roof ---
OX0, OX1 = 1603, 1606
OZ0, OZ1 = -1412, -1405
OMID = -1409

foundation_to_grade(OX0, OZ0, OX1, OZ1, FS, FOOT, skirt=0)
place_cuboid(BX0, FS, OZ0, BX0, EAVES, OZ0, WALL)        # return of the gable
place_cuboid(OX0, FS, OZ0, OX1, EAVES - 1, OZ1, WALL)
place_cuboid(OX0, FS, OZ0, OX1, FS + 1, OZ1, FOOT)
place_cuboid(OX0 + 1, FN + 1, OZ0 + 1, OX1, EAVES - 1, OMID, AIR)
place_cuboid(OX0 + 1, FS + 1, OMID + 1, OX1, EAVES - 1, OZ1 - 1, AIR)
for x in range(OX0 + 1, OX1 + 1):
    place_block(x, FN, OMID + 1, SLAB + "[type=bottom]")
for x in range(OX0, OX1 + 1):
    top = 107 + (x - OX0) // 2
    place_cuboid(x, top + 1, OZ0, x, EAVES - 1, OZ1, AIR)
moss(OX0, FS + 2, OZ0, OX1, EAVES - 2, OZ1, 0.18)

place_cuboid(1604, FN + 1, OZ0, 1604, FN + 2, OZ0, AIR)
ODOOR = doorway(1604, FN + 1, OZ0, "south", "cobblestone", leaf=LEAF,
                lintel=True)
place_cuboid(BX0, FN + 1, -1410, BX0, FN + 3, -1409, AIR)   # through to the barn
openings(OX0, FN, -1411, OX0, -1406, spacing=3, width=1, sill=2, head=3)
roof(OX0, OZ0, OX1, OZ1, 106, ROOFM, "shed", "w", (1, 2), 1)

# the reserved doorstep stays open ground, level with its own lane cell
place_cuboid(1602, FN, -1412, 1602, FN, -1409, WALL)
place_cuboid(1602, FN + 1, -1412, 1602, FN + 3, -1409, AIR)

# battered buttresses down the long south wall, where the ground is lowest
for x in (1615, 1620, 1624):
    place_cuboid(x, FS, BZ1 + 1, x, FN + 2, BZ1 + 1, WALL)
    place_cuboid(x, FS, BZ1 + 1, x, FS + 1, BZ1 + 1, FOOT)
    place_block(x, FN + 3, BZ1 + 1, SLAB + "[type=bottom]")

# --- the stackyard apron outside the south cart door ---
place_cuboid(1606, FS, BZ1 + 1, 1618, FS, BZ1 + 1, WALL)
for x in range(1606, 1619):
    g = get_height(x, -1399)
    if g < FS:
        place_block(x, FS, -1399, SLAB + "[type=bottom]")
moss(1606, FS, BZ1 + 1, 1618, FS, BZ1 + 1, 0.3)

# --- what is in the barn ---
fitting("fodder", 1623, FN + 1, -1409, "west", mat="spruce", extent=3)
fitting("fodder", 1623, FS + 1, -1403, "west", mat="spruce", extent=3)
fitting("workbench", 1621, FN + 1, -1410, "south", mat="spruce")
fitting("store", 1609, FS + 1, -1403, "north", mat="spruce", extent=3)
fitting("trough", 1616, FN + 1, -1410, "south", mat="cobblestone")
fitting("store", 1605, FN + 1, -1411, "south", mat="spruce", extent=2)
fitting("workbench", 1605, FS + 1, -1406, "north", mat="spruce")
fitting("light", 1606, FS + 1, -1407, "west", mat="spruce")

# ================================================================== walk it, read it
# back, and finish the ground
# ==================================================================
resolve_steps()

print("lookout gate:", GATE)
print("tower door:", TDOOR)
print("outshot door:", ODOOR)
print("check gate:", check_door(YX0, Y0 + 1, -1457))
print("check tower:", check_door(1683, F1 + 1, -1453))
print("check outshot:", check_door(1604, FN + 1, OZ0))
print("approach lookout:", approach("ridge_lookout", x=YX0, y=Y0 + 1, z=-1457,
                                    mat="cobblestone", width=2))
print("approach barn:", approach("threshing_barn", x=1604, y=FN + 1, z=OZ0,
                                 mat="cobblestone", width=2))
print("walk lookout:", check_walkable("ridge_lookout"))
print("walk barn:", check_walkable("threshing_barn"))
print("sealed lookout:", seal_voids(1674, -1458, 1688, -1444, WALL))
print("sealed barn:", seal_voids(1602, -1412, 1626, -1400, WALL))
print("attached:", check_attached())
dress_ground(1672, -1460, 1690, -1442)
dress_ground(1600, -1414, 1628, -1397)
