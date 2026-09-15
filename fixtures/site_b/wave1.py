import math
import random

random.seed(7)

# ------------------------------------------------------------------ palette
WALL = "blackstone"
POL = "polished_blackstone"
TRIM = "polished_blackstone_bricks"
ROOFM = "deepslate_tile"
FRAME = "stripped_birch_log"
FOOT = "basalt"
GLASS = "glass_pane"

# ------------------------------------------------------------------ helpers


def col(x, z, y0, y1, b):
    place_cuboid(x, y0, z, x, y1, z, b)


def coursed(x, z, y0, y1):
    """One wall column of coursed dark stone, with a little texture drift."""
    for y in range(y0, y1 + 1):
        place_block(x, y, z, POL if random.random() < 0.10 else WALL)


def ring_cells(x0, z0, x1, z1):
    out = []
    for x in range(x0, x1 + 1):
        out.append((x, z0))
        out.append((x, z1))
    for z in range(z0 + 1, z1):
        out.append((x0, z))
        out.append((x1, z))
    return out


# ================================================================== MOOT HALL
HX0, HZ0, HX1, HZ1 = 1634, -688, 1655, -674
FY = 68           # floor block
W0, W1 = 69, 81   # wall courses
EAVES = 82

print("reserve hall", reserve(HX0, HZ0, HX1, HZ1, "moot_hall"))
print("reserve well", reserve(1638, -666, 1645, -659, "court_wellhouse"))

clear_trees(1630, -692, 1660, -670)

# --- cut the east bank out of the footprint, then floor and foot it
fill_region(HX0, FY + 1, HZ0, HX1, 96, HZ1, "air")
foundation_to_grade(HX0, HZ0, HX1, HZ1, FY, FOOT)
place_cuboid(HX0, FY, HZ0, HX1, FY, HZ1, TRIM)
for (x, z) in ring_cells(HX0, HZ0, HX1, HZ1):
    place_block(x, FY, z, FOOT)
# processional strip, west door to dais
place_cuboid(1635, FY, -681, 1650, FY, -681, "smooth_basalt")

# --- west front: solid piers / recessed bays, frontispiece on centre
WEST_SOLID = [-687, -686, -683, -682, -681, -680, -679, -675, -674]
WEST_BAY = [-685, -684, -678, -677, -676]
for z in WEST_SOLID:
    coursed(1634, z, W0, W1)
for z in WEST_BAY:
    coursed(1635, z, W0, W1)          # wall behind the recess
    place_block(1634, 75, z, TRIM)    # corbel band
    coursed(1634, z, 76, W1)          # jettied mass over the recess

# --- north wall: panel set back one, piers on the footprint line
for x in range(1635, 1655):
    coursed(x, -687, W0, W1)
for x in (1636, 1640, 1644, 1648, 1652):
    coursed(x, -688, W0, W1)
# corner porch: open recess at the north-west, mass carried over on a birch lintel
for x in (1634, 1635):
    place_block(x, 73, -688, FRAME + "[axis=x]")
    coursed(x, -688, 74, W1)

# --- east wall
for z in range(-687, -674):
    coursed(1654, z, W0, W1)
for z in (-685, -682, -679, -676):
    coursed(1655, z, W0, W1)
for (x, z) in [(1655, -688), (1654, -688), (1655, -687),
               (1655, -674), (1654, -674), (1655, -675), (1635, -674)]:
    coursed(x, z, W0, W1)

# --- south wall
for x in range(1635, 1655):
    coursed(x, -675, W0, W1)
for x in (1637, 1641, 1645, 1649, 1653):
    coursed(x, -674, W0, W1)

# --- courses: plinth, mid string, eaves corbel (all on the outer line)
OUTER = ring_cells(HX0, HZ0, HX1, HZ1)
for (x, z) in OUTER:
    if (x, z) in ((1634, -688), (1635, -688)):
        continue
    place_block(x, W0, z, TRIM)
    place_block(x, 75, z, TRIM)
    place_block(x, W1, z, TRIM)

# the plinth follows the grade: in each recess it rises to meet the ground outside it,
# so the base steps up the bench instead of running dead level
for (x, z) in OUTER:
    if get_block(x, 74, z) != "air":
        continue
    nb = []
    if z == HZ0:
        nb.append((x, z - 1))
    if z == HZ1:
        nb.append((x, z + 1))
    if x == HX0:
        nb.append((x - 1, z))
    if x == HX1:
        nb.append((x + 1, z))
    g = max(get_height(a, b) for (a, b) in nb)
    if g > W0:
        place_cuboid(x, W0 + 1, z, x, g, z, TRIM)

# --- openings: narrow, high, framed in pale birch
openings(1638, FY, -687, 1653, -687, spacing=4, width=1, sill=5, head=9,
         block=GLASS, sill_block=TRIM, lintel=FRAME)
openings(1636, FY, -675, 1638, -675, spacing=3, width=1, sill=4, head=9,
         block=GLASS, sill_block=TRIM, lintel=FRAME)
openings(1645, FY, -675, 1653, -675, spacing=4, width=1, sill=4, head=9,
         block=GLASS, sill_block=TRIM, lintel=FRAME)
openings(1654, FY, -686, 1654, -683, spacing=3, width=1, sill=6, head=10,
         block=GLASS, sill_block=TRIM, lintel=FRAME)
openings(1654, FY, -679, 1654, -676, spacing=3, width=1, sill=6, head=10,
         block=GLASS, sill_block=TRIM, lintel=FRAME)

# bright cuts deep in the two west recesses
for z in (-684, -677):
    place_block(1635, 70, z, TRIM)
    col(1635, z, 71, 74, GLASS)
    place_block(1635, 75, z, FRAME + "[axis=z]")

# --- the roof: one large hip, ridge east-west, tight to the wall
RIDGE = roof(HX0, HZ0, HX1, HZ1, EAVES, ROOFM,
             style="hip", axis="z", pitch=(1, 2), overhang=0)
print("hall ridge", RIDGE)

# --- west frontispiece: the portal reveal thickened inside, capped at the eaves
for z in (-683, -679):
    coursed(1635, z, W0, W1)
    place_block(1635, W1, z, TRIM)

# great west door
fill_region(1634, W0, -681, 1634, 71, -681, "air")
place_block(1634, W0, -681, "birch_door[facing=west,half=lower,hinge=left]")
place_block(1634, W0 + 1, -681, "birch_door[facing=west,half=upper,hinge=left]")
place_block(1634, 71, -681, GLASS)
for z in range(-683, -678):
    place_block(1634, 72, z, FRAME + "[axis=z]")
for y in range(W0, 71):
    place_block(1634, y, -682, FRAME + "[axis=y]")
    place_block(1634, y, -680, FRAME + "[axis=y]")
place_block(1634, 71, -682, "lantern[hanging=true]")
place_block(1634, 71, -680, "lantern[hanging=true]")
# the slit: one tall cut, stepped, running the wall from the lintel to the eaves
col(1634, -681, 73, 80, GLASS)
col(1634, -682, 75, 80, GLASS)
col(1634, -680, 75, 80, GLASS)
for y in range(73, 81):
    place_block(1634, y, -683, FRAME + "[axis=y]")
    place_block(1634, y, -679, FRAME + "[axis=y]")
place_block(1634, 81, -681, TRIM)
place_block(1634, 81, -682, TRIM)
place_block(1634, 81, -680, TRIM)

# --- hearth and chimney, pushed west of centre on the south wall
place_cuboid(1640, W0, -676, 1642, EAVES, -675, WALL)
fill_region(1640, W0, -676, 1642, 70, -676, "air")
place_block(1641, W0, -676, "campfire[lit=true]")
place_block(1640, 71, -676, "blackstone_stairs[facing=north,half=top]")
place_block(1641, 71, -676, TRIM)
place_block(1642, 71, -676, "blackstone_stairs[facing=north,half=top]")
place_cuboid(1639, FY, -677, 1643, FY, -676, "smooth_basalt")
col(1641, -675, 83, 87, WALL)
col(1641, -676, 83, 86, WALL)
place_block(1641, 87, -676, TRIM)
place_block(1641, 88, -675, TRIM)

# --- dais at the cut-in east end, two courses up
steps([(1650, W0, z) for z in range(-685, -675)], "blackstone", axis="x", prefer="west")
place_cuboid(1651, W0, -685, 1653, W0, -676, WALL)
place_cuboid(1651, 70, -685, 1653, 70, -676, TRIM)
place_block(1652, 71, -681, "blackstone_stairs[facing=west,half=bottom]")
place_block(1652, 71, -682, "blackstone_wall")
place_block(1652, 71, -680, "blackstone_wall")
for z in (-683, -679):
    place_block(1651, 71, z, "blackstone_wall")
    place_block(1651, 72, z, "lantern")

# --- roof frame: birch rafters laid up under the slope, so nothing walkable
def ceiling(x, z, top=90):
    y = top
    while y > W1 and get_block(x, y, z) == "air":
        y -= 1
    return y


for x in (1637, 1641, 1645, 1649, 1653):
    for z in range(-686, -675):
        c = ceiling(x, z)
        if W1 + 1 <= c - 1 <= RIDGE:
            place_block(x, c - 1, z, FRAME + "[axis=z]")

# --- wall lanterns on birch brackets, uneven on the two sides
for x in (1639, 1645, 1651):
    place_block(x, 74, -686, "birch_stairs[facing=south,half=top]")
    place_block(x, 73, -686, "lantern[hanging=true]")
for x in (1637, 1647):
    place_block(x, 74, -676, "birch_stairs[facing=north,half=top]")
    place_block(x, 73, -676, "lantern[hanging=true]")
for z in (-684, -678):
    place_block(1653, 74, z, "birch_stairs[facing=east,half=top]")
    place_block(1653, 73, z, "lantern[hanging=true]")

# --- benches and two trestles, the centre lane left clear
for z, f in ((-686, "north"), (-676, "south")):
    for x in range(1636, 1654):
        if 1639 <= x <= 1643 or x >= 1650:
            continue
        place_block(x, W0, z, "birch_stairs[facing=%s,half=bottom]" % f)
for tz in (-684, -678):
    place_cuboid(1637, W0, tz, 1647, W0, tz, FRAME + "[axis=x]")
    for x in (1639, 1644):
        place_block(x, W0 + 1, tz, "lantern")
    for x in range(1637, 1648):
        place_block(x, W0, tz - 1, "birch_stairs[facing=south,half=bottom]")
        place_block(x, W0, tz + 1, "birch_stairs[facing=north,half=bottom]")

# --- niche lights on the street face
for z in (-685, -678):
    place_block(1634, 70, z, "blackstone_wall")
    place_block(1634, 71, z, "lantern")

# --- court door, set back in the north-west porch
fill_region(1635, W0, -687, 1635, 71, -687, "air")
place_block(1635, W0, -687, "birch_door[facing=south,half=lower,hinge=right]")
place_block(1635, W0 + 1, -687, "birch_door[facing=south,half=upper,hinge=right]")
place_block(1635, 71, -687, FRAME + "[axis=x]")
place_block(1636, W0, -687, FRAME + "[axis=y]")
place_block(1636, W0 + 1, -687, FRAME + "[axis=y]")
place_block(1635, 72, -688, "lantern[hanging=true]")

# --- the east light: one tall cut over the dais, into the bank the hall is cut into
place_block(1654, 70, -681, TRIM)
col(1654, -681, 71, 75, GLASS)
for y in range(70, 77):
    place_block(1654, y, -682, FRAME + "[axis=y]")
    place_block(1654, y, -680, FRAME + "[axis=y]")
place_block(1654, 76, -681, FRAME + "[axis=z]")

for d in ((1634, W0, -681), (1635, W0, -687)):
    r = check_door(*d)
    print("door", d, r)
    if not r["ok"]:
        print("  !! failing door", d)

# ================================================================ WELLHOUSE
WX0, WZ0, WX1, WZ1 = 1638, -666, 1645, -659
PY = 69           # paving block

clear_trees(1634, -670, 1650, -654)
fill_region(WX0, PY + 1, WZ0, WX1, 92, WZ1, "air")
foundation_to_grade(WX0, WZ0, WX1, WZ1, PY, FOOT)
place_cuboid(WX0, PY, WZ0, WX1, PY, WZ1, TRIM)
for (x, z) in ring_cells(WX0, WZ0, WX1, WZ1):
    place_block(x, PY, z, FOOT)

# feather the cut back into the bank instead of leaving a kerb
for x in range(1646, 1650):
    for z in range(-669, -655):
        t = PY + (x - 1645)
        h = get_height(x, z)
        if h > t:
            fill_region(x, t + 1, z, x, h, z, "air")
for z in list(range(-669, -666)) + list(range(-658, -654)):
    for x in range(1638, 1646):
        d = (-666 - z) if z < -666 else (z + 658)
        t = PY + max(1, d)
        h = get_height(x, z)
        if h > t:
            fill_region(x, t + 1, z, x, h, z, "air")

# four heavy piers; the gaps between them are the tall openings
PIERS = [(1639, -665), (1642, -665), (1639, -662), (1642, -662)]
for (px, pz) in PIERS:
    for x in range(px, px + 2):
        for z in range(pz, pz + 2):
            place_block(x, 70, z, TRIM)
            coursed(x, z, 71, 72)
            place_block(x, 73, z, TRIM)
# birch lintel plate carrying the roof
for (x, z) in ring_cells(1639, -665, 1643, -661):
    ax = "x" if (z == -665 or z == -661) else "z"
    place_block(x, 74, z, FRAME + "[axis=%s]" % ax)

WRIDGE = roof(1639, -665, 1643, -661, 75, ROOFM,
              style="hip", axis="z", pitch=(1, 1), overhang=1)
print("wellhouse ridge", WRIDGE, "hall ridge", RIDGE)

# the spring, built up in a raised curb rather than sunk
place_block(1641, 68, -663, FOOT)
for (x, z) in ring_cells(1640, -664, 1642, -662):
    place_block(x, 70, z, TRIM)
place_block(1641, 69, -663, "water")
place_block(1641, 70, -663, "water")
for x in (1640, 1642):
    place_block(x, 71, -663, "birch_fence")
    place_block(x, 72, -663, "birch_fence")
place_cuboid(1640, 73, -663, 1642, 73, -663, FRAME + "[axis=x]")
place_block(1641, 72, -663, "lantern[hanging=true]")
for (x, z) in ((1641, -665), (1641, -661), (1639, -663), (1643, -663)):
    place_block(x, 73, z, "lantern[hanging=true]")

# waist-high parapet on the two uphill sides, with the yard gate east
for z in range(-665, -659):
    place_block(1645, 70, z, "blackstone_wall")
for x in range(1638, 1646):
    place_block(x, 70, -659, "blackstone_wall")
place_block(1645, 70, -662, "birch_fence_gate[facing=east]")

# a seat under the eaves, and a step off the high side
place_block(1644, 70, -663, "blackstone_stairs[facing=east,half=bottom]")
place_block(1644, 70, -662, "blackstone_stairs[facing=east,half=bottom]")
steps([(1646, 70, z) for z in range(-665, -660)], "blackstone", axis="x", prefer="west")

# ==================================================================== finish
print("seal hall", seal_voids(HX0, HZ0, HX1, HZ1, WALL))
print("seal well", seal_voids(WX0, WZ0, WX1, WZ1, WALL))
dress_ground(1656, -690, 1661, -670)
dress_ground(1639, -673, 1657, -669)
dress_ground(1646, -670, 1650, -655)
dress_ground(1638, -658, 1645, -655)
print("attached", check_attached())
