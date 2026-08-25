"""First milestone controller for carrying grab_box with fixed hand visuals."""

from __future__ import annotations

import math
from typing import Callable

import carb
import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade


LEFT_ARM_JOINTS = tuple(f"zarm_l{i}_joint" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"zarm_r{i}_joint" for i in range(1, 8))
WHEEL_JOINTS = (
    "wheel_left_front_joint",
    "wheel_right_front_joint",
    "wheel_left_behind_joint",
    "wheel_right_behind_joint",
)
TORSO_JOINTS = ("knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint")
HEAD_JOINTS = ("zhead_1_joint", "zhead_2_joint")
EXPECTED_DOFS = set(WHEEL_JOINTS + TORSO_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS)

# Solved for the current 90-degree-rotated grab_box and the actual S49 palm
# mesh frames. Both palms are vertical: the left normal points toward -Y and
# the right normal toward +Y. The approach pose keeps a 75 mm side clearance;
# the hold pose closes the palm surfaces onto the two box sides.
APPROACH_ARMS = (
    (-0.4578061911, 0.2427956664, -0.1411977185, -1.5491467590, 0.2461684501, 0.1318752331, 0.4027017849),
    (-0.4574084010, -0.2395977650, 0.1434211923, -1.5509707049, -0.2443057083, -0.1352737020, 0.4037756215),
)
HOLD_ARMS = (
    (-0.4854265195, 0.0564407817, -0.2933028563, -1.5350321631, 0.0677167893, 0.2905581028, 0.4317227208),
    (-0.4855064974, -0.0529761940, 0.2955845200, -1.5345846554, -0.0658768949, -0.2936553662, 0.4317818185),
)

# Seven synchronized waypoints lift the box by 150 mm while preserving the
# vertical, inward-facing palm frames. A higher lift would push both wrist
# pitch/roll joints onto their imported limits.
LIFT_ARMS = (
    ((-0.4842712785, 0.0562467617, -0.2932139601, -1.5330768887, 0.0681880249, 0.2903715674, 0.4285326704),
     (-0.4843507536, -0.0527846441, 0.2954929029, -1.5326297258, -0.0663576830, -0.2934683355, 0.4285887425)),
    ((-0.5195026216, 0.0603222487, -0.2941903669, -1.5838376245, 0.0572246770, 0.2943429123, 0.5153737609),
     (-0.5195972937, -0.0567779237, 0.2965284790, -1.5833859106, -0.0551155579, -0.2974512746, 0.5155167075)),
    ((-0.5609616409, 0.0630471649, -0.2944574987, -1.6256245777, 0.0473681893, 0.2967677705, 0.5996076528),
     (-0.5610684906, -0.0594130875, 0.2968287573, -1.6251675915, -0.0449890772, -0.2998700343, 0.5998339982)),
    ((-0.6082157904, 0.0645026585, -0.2940291711, -1.6586486735, 0.0388203910, 0.2977619115, 0.6809381015),
     (-0.6083315075, -0.0607732346, 0.2964066528, -1.6581863279, -0.0361845907, -0.3008426055, 0.6812425733)),
    ((-0.7888378392, 0.3194403790, -0.6291351601, -1.6884601672, 0.3083980715, 0.6327017134, 0.6981320000),
     (-0.7914972954, -0.3180072396, 0.6354438210, -1.6880331120, -0.3091067694, -0.6396927666, 0.6981320000)),
    ((-0.8942274424, 0.3804733883, -0.7207077272, -1.7072601967, 0.3917832605, 0.6981320000, 0.6981320000),
     (-0.8929191427, -0.3748088963, 0.7194829111, -1.7065575081, -0.3878030077, -0.6981320000, 0.6981320000)),
    ((-0.9666519728, 0.4052389333, -0.7458105359, -1.7157970453, 0.4518092765, 0.6981320000, 0.6981320000),
     (-0.9651593198, -0.3994724114, 0.7443082354, -1.7151206190, -0.4476465549, -0.6981320000, 0.6981320000)),
)

# Before approaching the rack, extend both palms 75 mm while preserving their
# loaded height and orientation.  The insertion target is computed from this
# pose, so the base stops farther from the rack instead of moving during place.
RACK_CLEAR_ARMS = (
    (-1.0987639942, 0.2638404165, -0.7104886844, -1.2668436767,
     0.5556557736, 0.5037097168, 0.4655859579),
    (-1.0974287075, -0.2589413300, 0.7081227175, -1.2663561476,
     -0.5525102438, -0.5040974702, 0.4646993238),
)

# Cartesian-continuous path from the loaded carry pose to RACK_CLEAR_ARMS.
# The palms advance 75 mm without changing height or orientation; using this
# path during torso elevation avoids the large intermediate palm rotation
# produced by direct joint-space interpolation of only the two endpoints.
RACK_ELEVATE_ARM_PATH = (
    ((-0.9752139631, 0.3945473341, -0.7421542203, -1.6829061376, 0.4622699044, 0.6828620341, 0.6791833963),
     (-0.9737248954, -0.3888505210, 0.7405780809, -1.6822498666, -0.4582068694, -0.6828954812, 0.6790978727)),
    ((-0.9841833080, 0.3836745326, -0.7386425454, -1.6493133171, 0.4723363520, 0.6674411455, 0.6602640760),
     (-0.9826991221, -0.3780473664, 0.7369924943, -1.6486763723, -0.4683693980, -0.6675070745, 0.6600958016)),
    ((-0.9935486479, 0.3726380584, -0.7352745466, -1.6150568998, 0.4820093235, 0.6518956615, 0.6413807061),
     (-0.9920707522, -0.3670804882, 0.7335507596, -1.6144384354, -0.4781348863, -0.6519932758, 0.6411324223)),
    ((-1.0033208113, 0.3614280530, -0.7320401843, -1.5800914085, 0.4913135206, 0.6362125746, 0.6224932584),
     (-1.0018507239, -0.3559401961, 0.7302428819, -1.5794906054, -0.4875283100, -0.6363412999, 0.6221674698)),
    ((-1.0135131309, 0.3500326376, -0.7289294701, -1.5443652516, 0.5002719221, 0.6203764036, 0.6035608072),
     (-1.0120524740, -0.3446147739, 0.7270588930, -1.5437813121, -0.4965729099, -0.6205358657, 0.6031597885)),
    ((-1.0241418026, 0.3384377626, -0.7259324941, -1.5078199057, 0.5089059402, 0.6043690689, 0.5845410462),
     (-1.0226923005, -0.3330903316, 0.7239888812, -1.5072520479, -0.5052903499, -0.6045590760, 0.5840668490)),
    ((-1.0352263793, 0.3266269157, -0.7230394132, -1.4703886554, 0.5172356274, 0.5881695787, 0.5653896045),
     (-1.0337898610, -0.3213505174, 0.7210229818, -1.4698361063, -0.5137009255, -0.5883901079, 0.5648440615)),
    ((-1.0467903820, 0.3145807342, -0.7202404207, -1.4319950051, 0.5252798774, 0.5717535934, 0.5460592387),
     (-1.0453687867, -0.3093761329, 0.7181513471, -1.4314569929, -0.5218237674, -0.5720047816, 0.5454439666)),
    ((-1.0588620677, 0.3022764906, -0.7175256980, -1.3925506568, 0.5330566258, 0.5550928390, 0.5264988553),
     (-1.0574574558, -0.2971446223, 0.7153640990, -1.3920264018, -0.5296770453, -0.5553749773, 0.5258152537)),
    ((-1.0714754148, 0.2896874131, -0.7148853462, -1.3519529012, 0.5405830585, 0.5381543209, 0.5066522969),
     (-1.0700899831, -0.2846293971, 0.7126512587, -1.3514416040, -0.5372781790, -0.5384678524, 0.5059015447)),
    ((-1.0846714084, 0.2767817800, -0.7123092919, -1.3100812027, 0.5478758353, 0.5208992696, 0.4864567988),
     (-1.0833075115, -0.2717989355, 0.7100026521, -1.3095820301, -0.5446440663, -0.5212447915, 0.4856398458)),
    RACK_CLEAR_ARMS,
)

# Continuous IK solution from RACK_CLEAR_ARMS: both palm frames move down by
# 210 mm in fourteen equal steps.  The torso rises by 80 mm at the same time,
# so the box descends 130 mm while the base remains fixed at its insertion pose.
PLACE_ARM_PATH = (
    ((-1.0676827415, 0.2629185201, -0.7111656351, -1.2624414499, 0.5555643774, 0.5020053682, 0.4311381333),
     (-1.0663593583, -0.2580305236, 0.7088096069, -1.2619197569, -0.5524528993, -0.5023122306, 0.4302145323)),
    ((-1.0377938896, 0.2609180447, -0.7116467774, -1.2545287349, 0.5560026445, 0.4989429357, 0.3951019599),
     (-1.0364857973, -0.2560472612, 0.7092963325, -1.2539722672, -0.5529286265, -0.4991720053, 0.3941372075)),
    ((-1.0092143844, 0.2578236345, -0.7119277164, -1.2430654875, 0.5569598798, 0.4945158879, 0.3575666850),
     (-1.0079251663, -0.2529760499, 0.7095786178, -1.2424733771, -0.5539262239, -0.4946705277, 0.3565568416)),
    ((-0.9820381547, 0.2536140749, -0.7120056959, -1.2279853776, 0.5584209992, 0.4887111243, 0.3185780442),
     (-0.9807716430, -0.2487955383, 0.7096538261, -1.2273564003, -0.5554301493, -0.4887948796, 0.3175193275)),
    ((-0.9563684719, 0.2482618114, -0.7118787683, -1.2091949584, 0.5603673675, 0.4815090447, 0.2781698131),
     (-0.9551287987, -0.2434780258, 0.7095201134, -1.2085274488, -0.5574213368, -0.4815256394, 0.2770585271)),
    ((-0.9323194864, 0.2417312747, -0.7115454809, -1.1865674342, 0.5627776125, 0.4728815384, 0.2363600676),
     (-0.9311111516, -0.2369877881, 0.7091761072, -1.1858591689, -0.5598780242, -0.4728348658, 0.2351925066)),
    ((-0.9100188273, 0.2339764152, -0.7110044295, -1.1599336543, 0.5656286102, 0.4627890662, 0.1931462840),
     (-0.9088467903, -0.2292786101, 0.7086204493, -1.1591816886, -0.5627767519, -0.4626831722, 0.1919186033)),
    ((-0.8896117830, 0.2249370982, -0.7102536534, -1.1290690898, 0.5688966578, 0.4511764458, 0.1484988131),
     (-0.8884815936, -0.2202901777, 0.7078511789, -1.1282695239, -0.5660935529, -0.4510155021, 0.1472068555)),
    ((-0.8712678650, 0.2145337675, -0.7092898258, -1.0936746846, 0.5725588924, 0.4379666775, 0.1023518530),
     (-0.8701858530, -0.2099427331, 0.7068649039, -1.0928223167, -0.5698053912, -0.4377549376, 0.1009909098)),
    ((-0.8551911372, 0.2026593444, -0.7081071461, -1.0533478913, 0.5765950755, 0.4230516432, 0.0545903628),
     (-0.8541646993, -0.1981289565, 0.7056556671, -1.0524356850, -0.5738919738, -0.4227933685, 0.0531548317)),
    ((-0.8416368335, 0.1891664638, -0.7066957560, -1.0075371257, 0.5809899998, 0.4062775273, 0.0050300331),
     (-0.8406748844, -0.1847011624, 0.7042133196, -1.0065553522, -0.5783381999, -0.4059768566, 0.0035129034)),
    ((-0.8309391776, 0.1738463262, -0.7050393135, -0.9554664101, 0.5857370425, 0.3874207404, -0.0466153384),
     (-0.8300528994, -0.1694500667, 0.7025210262, -0.9544011840, -0.5831377853, -0.3870814430, -0.0482233152)),
    ((-0.8235608007, 0.1563912704, -0.7031109419, -0.8960021808, 0.5908440163, 0.3661453723, -0.1007910455),
     (-0.8227649752, -0.1520671621, 0.7005510850, -0.8948327682, -0.5882992473, -0.3657703456, -0.1025028066)),
    ((-0.8201882076, 0.1363224427, -0.7008656901, -0.8273962892, 0.5963440643, 0.3419210022, -0.1582114445),
     (-0.8195038262, -0.1320718885, 0.6982571287, -0.8260896379, -0.5938570778, -0.3415111415, -0.1600464407)),
)

# Stop 35 mm farther back than the original pickup pose.  The fixed fingers
# extend toward the box's front edge, so centring the whole hand pair farther
# rearward gives the palm and four pads a broader side-wall contact patch.
PICKUP_BASE_TO_BOX_X = 0.63675
BASE_MAX_SPEED = 0.25
BASE_ACCELERATION = 0.20
WHEEL_RADIUS = 0.13
APPROACH_HOLD_SECONDS = 0.75
ARM_APPROACH_SECONDS = 3.0
HAND_HOLD_SECONDS = 2.0
HAND_SETTLE_SECONDS = 0.5
HAND_SETTLE_TIMEOUT_SECONDS = 4.0
ARM_TARGET_TOLERANCE = math.radians(4.0)
GRASP_ATTACH_SECONDS = 1.0
LIFT_SECONDS = 4.0
LIFT_SETTLE_SECONDS = 0.75
MINIMUM_LIFT_HEIGHT = 0.12
# Clear the pickup table and its remaining box stack before rotating the load.
# This motion is along the exact reverse of BASE_APPROACH, so the held box does
# not sweep sideways through neighbouring boxes during the subsequent turn.
PICKUP_RETREAT_DISTANCE = 0.45
PICKUP_RETREAT_MAX_SPEED = 0.12
PICKUP_RETREAT_ACCELERATION = 0.05
PICKUP_RETREAT_SETTLE_SECONDS = 0.75
TURN_ANGLE = math.pi
TURN_MAX_SPEED = math.radians(10.0)
TURN_ACCELERATION = math.radians(6.0)
TURN_SETTLE_SECONDS = 1.0
TURN_WHEEL_TRACK_RADIUS = 0.32
MAXIMUM_TURN_BOX_DROP = 0.03
CARRY_TARGET_RACK_PATH = "/World/rack2"
# The box-only clearance could be smaller, but the S63 base shell reaches
# nearly as far forward as the held box and touches rack2's posts first.
CARRY_RACK_CLEARANCE = 0.42
CARRY_MAX_SPEED = 0.15
CARRY_ACCELERATION = 0.03
CARRY_SETTLE_SECONDS = 1.0
MAXIMUM_CARRY_BOX_DROP = 0.04
MAXIMUM_TRANSPORT_TILT = math.radians(7.5)
BOX_ANGULAR_DAMPING = 80.0
PLACE_BOX_ANGULAR_DAMPING = 160.0
RACK_HIGH_TORSO = (0.31145330, -0.68001516, 0.36856186)
PLACE_HIGH_TORSO = (0.41382152, -0.89126469, 0.47744317)
RACK_ELEVATE_SECONDS = 12.0
RACK_ELEVATE_SETTLE_SECONDS = 0.75
RACK_INSERT_BOX_X = -1.85
RACK_INSERT_MAX_SPEED = 0.10
RACK_INSERT_ACCELERATION = 0.04
RACK_INSERT_SETTLE_SECONDS = 0.75
# Release before the visible box reaches the rack.  The box prim origin and the
# visible shelf differ by about 25 mm in this imported STEP scene, so a 32 mm
# trigger preserves a small visual gap instead of waiting for a contact stall.
PLACE_GRAVITY_DROP_TRIGGER_CLEARANCE = 0.032
PLACE_LOWER_SECONDS = 12.0
# The interactive viewport can make the articulation track the final arm pose
# a little later than the commanded twelve-second trajectory. Keep commanding
# that final pose until physical shelf contact, with a separate real timeout.
PLACE_LOWER_TIMEOUT_SECONDS = 18.0
PLACE_TORSO_TRACKING_TOLERANCE = math.radians(2.0)
PLACE_GRAVITY_DROP_LINEAR_DAMPING = 80.0
# rack2's original level-3 geometry consists of several rollers/rail parts.
# CCD and the rack-side contact buffer can report the first stable multi-part
# contact above the marker-derived visual reference.  This is only a state
# transition tolerance; the zero rest offset still lets gravity settle the box
# onto the original rack geometry during PLACE_BOX_SETTLE.
PLACE_GRAVITY_DROP_CONTACT_CLEARANCE = 0.030
PLACE_GRAVITY_DROP_TIMEOUT_SECONDS = 6.0
RELEASE_ROLLER_ENTRY_SPEED = 0.20
MINIMUM_RELEASE_SLIDE_DISTANCE = 0.55
RACK_ROLL_ASSIST_FORCE = 0.15
RACK_ROLL_ASSIST_TARGET_X = -2.55
LEVEL3_CONNECTOR_X_RANGE = (-2.38, -2.28)
LEVEL3_CONNECTOR_Y_RANGE = (3.70, 4.25)
LEVEL3_CONNECTOR_Z_RANGE = (1.15, 1.35)
LEVEL3_CONNECTOR_NAMES = {"__143", "__143_1", "__382", "__382_1"}
PLACE_SUPPORT_SETTLE_SECONDS = 0.75
PLACE_RELEASE_SECONDS = 2.0
PLACE_BOX_SETTLE_SECONDS = 2.0
POST_PLACE_EXTRA_RETREAT = 0.20
TORSO_LOWER_SECONDS = 4.0
ARM_RETURN_SECONDS = 3.0
RETREAT_POSE_SETTLE_SECONDS = 1.5
RETREAT_SECONDS = 12.0
# The imported lane marker spans 40 mm vertically.  Its lower face plus 25 mm
# matches the measured resting height on rack2's original level-3 collision.
LEVEL3_VISUAL_SURFACE_OFFSET = 0.025
# Fixed-hand release pose: both palms move 80 mm away from the box while
# preserving their loaded-lift height and inward-facing orientation.
RELEASE_ARMS = (
    (-0.8926405788, 0.5874833962, -0.5971304278, -1.7375876310, 0.6231733405, 0.5428696686, 0.6078233348),
    (-0.8912457051, -0.5819733744, 0.5955711434, -1.7386422418, -0.6188433752, -0.5436920635, 0.6089543610),
)
RESET_SETTLE_SECONDS = 0.5
MAX_WHEEL_ACCELERATION = 4.0

# The imported fingers have no joints.  Convex collisions are generated from
# the visible palm and finger meshes, making both complete fixed hands act as
# the two jaws of a gripper without a hidden shape spanning the air gap.
GRIP_MATERIAL_PATH = "/DexhandPalmGripMaterial"
RELEASE_MATERIAL_PATH = "/DexhandBoxReleaseMaterial"
LEVEL3_SUPPORT_PATH = "/DexhandRack2Level3Support"
FIXED_THUMB_TRANSLATIONS = {
    # Retract both non-colliding thumb roots 3 mm toward the palms so their
    # fixed tips remain visibly clear of the box's upper rim during release.
    "l_thumb_prox": (0.004537, -0.013242, -0.1076135),
    "r_thumb_prox": (0.004500, 0.0116473, -0.1091828),
}
FIXED_PALM_TRANSLATIONS = {
    # Add 6 mm visible-palm preload per side.  This keeps the broad palm
    # surfaces carrying the load when rack-placement joint tracking opens the
    # wrist spacing slightly; no hidden contact geometry is introduced.
    "l_palm": (0.0, -0.006, -0.07),
    "r_palm": (0.0, 0.006, -0.07),
}
FIXED_FINGER_ROOTS = (
    "l_index_prox", "l_middle_prox", "l_ring_prox", "l_little_prox",
    "r_index_prox", "r_middle_prox", "r_ring_prox", "r_little_prox",
)
# The palm meshes were preloaded 6 mm toward the box, but the imported fingers
# are separate sibling transforms rather than children of those meshes.  Apply
# a 23 mm inward offset to each finger root.  Side/front viewport inspection
# shows that this fixed cupped-hand mesh has about 26 mm of built-in depth
# between its protruding palm face and the white finger-pad faces.  Keep 3 mm
# of compliance instead of fully cancelling that depth: a full 26 mm preload
# over-constrained the box and increased loaded-carry tilt. Curl angles remain
# unchanged.
FIXED_FINGER_TRANSLATIONS = {
    "l_index_prox": (0.0290066, -0.02523961, -0.158289),
    "l_middle_prox": (0.00799729, -0.02129005, -0.162285),
    "l_ring_prox": (-0.0131057, -0.02130827, -0.160292),
    "l_little_prox": (-0.0337273, -0.02533593, -0.153775),
    "r_index_prox": (0.029003, 0.01912146, -0.158234),
    "r_middle_prox": (0.008, 0.01486941, -0.161942),
    "r_ring_prox": (-0.013103, 0.01499303, -0.159953),
    "r_little_prox": (-0.033731, 0.01943278, -0.153730),
}
FIXED_FINGER_BASE_ORIENTATIONS = {
    "l_index_prox": (-0.00514945, 0.0303624, 0.717140, -0.696248),
    "l_middle_prox": (0.000563312, -0.000562864, 0.706825, -0.707388),
    "l_ring_prox": (0.00657584, -0.0314397, 0.710030, -0.703438),
    "l_little_prox": (-0.00616957, -0.0436547, 0.715503, -0.697217),
    "r_index_prox": (0.670793, -0.740977, -0.0307155, 0.00677999),
    "r_middle_prox": (0.682290, -0.731082, 0.0, 0.0),
    "r_ring_prox": (0.678234, -0.734173, 0.0306456, -0.00708951),
    "r_little_prox": (0.671815, -0.739434, 0.0432936, 0.00521716),
}
# The imported left/right hands are not exact geometric mirrors.  Per-finger
# absolute curls close their different visual gaps without excessive bending.
FIXED_FINGER_CURL_DEGREES = {
    "l_index_prox": -2.5, "l_middle_prox": -0.5,
    "l_ring_prox": 2.5, "l_little_prox": 0.0,
    "r_index_prox": 5.0, "r_middle_prox": 7.0,
    "r_ring_prox": 10.0, "r_little_prox": 8.0,
}


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float).copy()


def unique_named_prim(stage: Usd.Stage, name: str) -> Usd.Prim:
    matches = [prim for prim in stage.Traverse() if prim.GetName() == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one prim named {name!r}, found {len(matches)}")
    return matches[0]


def articulation_root(stage: Usd.Stage) -> Usd.Prim:
    roots = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one articulation root, found {len(roots)}")
    return roots[0]


def world_pose(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    return (
        np.asarray(matrix.ExtractTranslation(), dtype=float),
        np.asarray(
            [quaternion.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
            dtype=float,
        ),
    )


def yaw_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def set_anchor_pose(anchor: UsdPhysics.FixedJoint, position: np.ndarray, yaw: float) -> None:
    # Playback keeps the stage edit target on the anonymous Session Layer, so
    # both the anchor and PhysX dynamic-transform writeback use one layer.
    anchor.CreateLocalPos0Attr().Set(Gf.Vec3f(*np.asarray(position, dtype=float)))
    anchor.CreateLocalRot0Attr().Set(
        Gf.Quatf(
            math.cos(0.5 * yaw),
            Gf.Vec3f(0.0, 0.0, math.sin(0.5 * yaw)),
        )
    )


def enable_hand_mesh_collider(
    mesh_prim: Usd.Prim, material: UsdShade.Material
) -> UsdPhysics.CollisionAPI:
    """Use one rendered fixed-hand mesh as a stable convex collider."""
    collision = UsdPhysics.CollisionAPI.Apply(mesh_prim)
    collision.CreateCollisionEnabledAttr().Set(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).CreateApproximationAttr().Set(
        UsdPhysics.Tokens.convexHull
    )
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
    physx_collision.CreateContactOffsetAttr().Set(0.003)
    physx_collision.CreateRestOffsetAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )
    return collision


def set_fixed_thumb_translation(stage: Usd.Stage, name: str, translation) -> None:
    matches = [
        prim
        for prim in stage.Traverse()
        if prim.GetName().lower() == name.lower()
        and prim.GetParent().GetName() in {"zarm_l7_link", "zarm_r7_link"}
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one fixed {name} root, found {len(matches)}")
    translate_ops = [
        op
        for op in UsdGeom.Xformable(matches[0]).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    if len(translate_ops) != 1:
        raise RuntimeError(f"Expected one {name} translate op, found {len(translate_ops)}")
    translate_ops[0].Set(Gf.Vec3d(*translation))


def set_fixed_finger_curl(
    stage: Usd.Stage,
    name: str,
    base_orientation,
    angle_degrees: float,
) -> None:
    """Author an absolute fixed curl; repeated replays must not accumulate it."""
    matches = [
        prim
        for prim in stage.Traverse()
        if prim.GetName().lower() == name
        and prim.GetParent().GetName() in {"zarm_l7_link", "zarm_r7_link"}
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one fixed {name} root, found {len(matches)}")
    orient_ops = [
        op
        for op in UsdGeom.Xformable(matches[0]).GetOrderedXformOps()
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient
    ]
    if len(orient_ops) != 1:
        raise RuntimeError(f"Expected one {name} orient op, found {len(orient_ops)}")
    w, x, y, z = base_orientation
    current_d = Gf.Quatd(
        float(w), Gf.Vec3d(float(x), float(y), float(z))
    )
    half_angle = math.radians(angle_degrees) * 0.5
    local_curl = Gf.Quatd(
        math.cos(half_angle), Gf.Vec3d(math.sin(half_angle), 0.0, 0.0)
    )
    curled = current_d * local_curl
    if orient_ops[0].GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        curled_imaginary = curled.GetImaginary()
        orient_ops[0].Set(
            Gf.Quatf(
                float(curled.GetReal()),
                Gf.Vec3f(
                    float(curled_imaginary[0]),
                    float(curled_imaginary[1]),
                    float(curled_imaginary[2]),
                ),
            )
        )
    else:
        orient_ops[0].Set(curled)


def quintic(elapsed: float, duration: float) -> tuple[float, float]:
    u = float(np.clip(elapsed / duration, 0.0, 1.0))
    position = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    velocity = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
    return position, velocity


class TrapezoidalProfile:
    def __init__(self, distance: float, max_speed: float, acceleration: float):
        distance = abs(float(distance))
        acceleration_time = max_speed / acceleration
        acceleration_distance = 0.5 * acceleration * acceleration_time**2
        if 2.0 * acceleration_distance >= distance:
            self.acceleration_time = math.sqrt(distance / acceleration)
            self.cruise_time = 0.0
        else:
            self.acceleration_time = acceleration_time
            self.cruise_time = (distance - 2.0 * acceleration_distance) / max_speed
        self.total_time = 2.0 * self.acceleration_time + self.cruise_time
        self.acceleration = acceleration

    def evaluate(self, elapsed: float) -> tuple[float, float]:
        elapsed = float(np.clip(elapsed, 0.0, self.total_time))
        ramp = self.acceleration_time
        if elapsed <= ramp:
            velocity = self.acceleration * elapsed
            position = 0.5 * self.acceleration * elapsed**2
        elif elapsed <= ramp + self.cruise_time:
            velocity = self.acceleration * ramp
            position = 0.5 * self.acceleration * ramp**2 + velocity * (elapsed - ramp)
        else:
            deceleration_time = elapsed - ramp - self.cruise_time
            velocity = self.acceleration * (ramp - deceleration_time)
            position = (
                0.5 * self.acceleration * ramp**2
                + self.acceleration * ramp * self.cruise_time
                + self.acceleration * ramp * deceleration_time
                - 0.5 * self.acceleration * deceleration_time**2
            )
        return position, velocity


class DexhandBoxCarryController:
    """Attach to lab.usd and run the pickup-pose milestone."""

    def __init__(self, status_callback: Callable[[str], None] | None = None):
        self._status_callback = status_callback
        self._timeline = omni.timeline.get_timeline_interface()
        self._update_subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="KuavoDexhandBoxCarryDemo")
        )
        self._stage = None
        self._phase = "DETACHED"
        self._resume_phase = None
        self._elapsed = 0.0
        self._reset_wait_frames = 0
        self._reset_fabric_disabled = False
        self._reset_initialized = False
        self._auto_play_after_reset = False
        self._initial_root_position = None
        self._initial_root_orientation = None
        self._initial_box_position = None
        self._initial_box_orientation = None
        self._initial_box_local_transform = None
        self._initial_dof_positions = None
        self._robot = None
        self._box = None
        self._anchor = None
        self._palm_collisions = []
        self._box_colliders = []
        self._target_rack_colliders = []
        self._persistent_edit_target = None
        self._turn_profile = TrapezoidalProfile(
            TURN_ANGLE, TURN_MAX_SPEED, TURN_ACCELERATION
        )
        self._set_status("DETACHED - open lab.usd, then click Play From Start")

    @property
    def phase(self) -> str:
        return self._phase

    def shutdown(self) -> None:
        # Stop PhysX while the Session Layer is still targeted, then expose the
        # persistent layer for user editing.  Reversing this order lets PhysX
        # write its final articulation transforms into lab.usd.
        if self._timeline.is_playing():
            self._timeline.pause()
        if self._stage is not None:
            self._timeline.stop()
            self._end_session_edit()
        self._update_subscription = None
        self._robot = None
        self._box = None

    def attach(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No USD stage is open")
        if stage != self._stage:
            self._initial_root_position = None
            self._initial_root_orientation = None
            self._initial_box_position = None
            self._initial_box_orientation = None
            self._initial_box_local_transform = None
            self._initial_dof_positions = None
        self._stage = stage
        self._phase = "ATTACHED"
        self._set_status("ATTACHED - click Reset to Start or Play From Start")

    def reset(self, auto_play: bool = False) -> None:
        if self._stage is None or self._stage != omni.usd.get_context().get_stage():
            self.attach()
        self._timeline.stop()
        self._end_session_edit()
        self._timeline.set_start_time(0.0)
        self._timeline.set_end_time(3600.0)
        self._timeline.set_looping(False)
        self._timeline.set_current_time(0.0)
        self._robot = None
        self._box = None
        self._reset_wait_frames = 2
        self._reset_fabric_disabled = False
        self._reset_initialized = False
        self._auto_play_after_reset = bool(auto_play)
        self._phase = "RESET_PENDING"
        self._elapsed = 0.0
        self._set_status("RESETTING - stopping the previous physics session")

    def replay(self) -> None:
        self.reset(auto_play=True)

    def play(self) -> None:
        if self._phase in {
            "DETACHED", "ATTACHED", "HOLD_READY", "LIFT_READY", "TURN_READY",
            "CARRY_READY", "DEMO_COMPLETE",
        }:
            self.reset(auto_play=True)
            return
        if self._phase == "PAUSED" and self._resume_phase is not None:
            self._phase = self._resume_phase
            self._resume_phase = None
            self._timeline.play()
            self._set_status(self._phase)
            return
        if self._phase == "READY":
            self._start_demo()

    def pause(self) -> None:
        if self._phase not in {
            "DETACHED", "ATTACHED", "READY", "HOLD_READY", "LIFT_READY",
            "TURN_READY", "CARRY_READY", "DEMO_COMPLETE", "PAUSED",
        }:
            self._resume_phase = self._phase
            self._phase = "PAUSED"
            self._timeline.pause()
            self._set_status(f"PAUSED - {self._resume_phase}")

    def _set_status(self, text: str) -> None:
        carb.log_info(f"Kuavo Dexhand Box Carry Demo: {text}")
        if self._status_callback is not None:
            self._status_callback(text)

    def _transition(self, phase: str, message: str | None = None) -> None:
        self._phase = phase
        self._elapsed = 0.0
        self._set_status(message or phase)

    def _begin_session_edit(self) -> Usd.EditTarget:
        """Temporarily target the anonymous layer used by runtime overrides."""
        previous = self._stage.GetEditTarget()
        if previous.GetLayer() == self._stage.GetSessionLayer():
            previous = self._persistent_edit_target or Usd.EditTarget(
                self._stage.GetRootLayer()
            )
        self._persistent_edit_target = previous
        self._stage.SetEditTarget(Usd.EditTarget(self._stage.GetSessionLayer()))
        return previous

    def _end_session_edit(self, previous: Usd.EditTarget | None = None) -> None:
        """Return GUI edits to a persistent layer after runtime authoring."""
        target = previous or self._persistent_edit_target
        if target is None or target.GetLayer() == self._stage.GetSessionLayer():
            target = Usd.EditTarget(self._stage.GetRootLayer())
        self._stage.SetEditTarget(target)

    def _configure_session(self) -> None:
        stage = self._stage
        previous_edit_target = self._begin_session_edit()
        root_prim = articulation_root(stage)
        box_prim = unique_named_prim(stage, "grab_box")
        if self._initial_root_position is None:
            self._initial_root_position, self._initial_root_orientation = world_pose(root_prim)
            self._initial_box_position, self._initial_box_orientation = world_pose(box_prim)
            self._initial_box_local_transform = Gf.Matrix4d(
                UsdGeom.Xformable(box_prim).GetLocalTransformation()
            )

        # Restore the authored box transform before the timeline starts and
        # before PhysX creates its rigid-body handle.  Authoring an Xform op
        # after simulation start would override the dynamic rigid pose and
        # make the box appear frozen even with kinematicEnabled=false.
        UsdGeom.Xformable(box_prim).MakeMatrixXform().Set(
            self._initial_box_local_transform
        )

        for path in (
            "/DexhandBoxGraspJoint",
            "/DexhandBoxBaseAnchor",
            GRIP_MATERIAL_PATH,
            RELEASE_MATERIAL_PATH,
            LEVEL3_SUPPORT_PATH,
        ):
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(path)
        for prim in list(stage.Traverse()):
            if prim.GetName() == "DexhandPalmPadCollider":
                stage.RemovePrim(prim.GetPath())

        self._root_prim = root_prim
        self._box_prim = box_prim
        target_rack = stage.GetPrimAtPath(CARRY_TARGET_RACK_PATH)
        if not target_rack.IsValid():
            raise RuntimeError(f"Target rack not found: {CARRY_TARGET_RACK_PATH}")
        physical_rack = unique_named_prim(stage, "rack3_flow_rack")
        # Match rack1 exactly: the STEP rack uses enabled Xform colliders with
        # MeshCollision approximation="none".  rack2 also contains four tag
        # Cube colliders that rack1 does not have; keep those visual tags out
        # of physics.  Explicitly author this on every reset so stale session
        # opinions from an earlier extension version cannot leave rack2 off.
        target_step_colliders = [
            prim
            for prim in Usd.PrimRange(target_rack, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsInstanceProxy()
        ]
        physical_rack_colliders = [
            prim
            for prim in Usd.PrimRange(physical_rack, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsInstanceProxy()
        ]
        if not target_step_colliders or not physical_rack_colliders:
            raise RuntimeError("Target rack collision geometry is incomplete")
        for collider in target_step_colliders:
            is_rack1_style_collider = collider.GetTypeName() == "Xform"
            UsdPhysics.CollisionAPI(collider).CreateCollisionEnabledAttr().Set(
                is_rack1_style_collider
            )
            if is_rack1_style_collider:
                UsdPhysics.MeshCollisionAPI.Apply(
                    collider
                ).CreateApproximationAttr().Set(UsdPhysics.Tokens.none)
        for collider in physical_rack_colliders:
            UsdPhysics.CollisionAPI(collider).CreateCollisionEnabledAttr().Set(False)
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        rack_range = bbox_cache.ComputeWorldBound(target_rack).ComputeAlignedRange()
        box_range = bbox_cache.ComputeWorldBound(box_prim).ComputeAlignedRange()
        self._rack_front_x = float(rack_range.GetMax()[0])
        level3_tag = unique_named_prim(stage, "rack2_tag_03_lane_level3_xpos_end")
        level3_tag_range = bbox_cache.ComputeWorldBound(level3_tag).ComputeAlignedRange()
        self._level3_surface_z = (
            float(level3_tag_range.GetMin()[2]) + LEVEL3_VISUAL_SURFACE_OFFSET
        )
        # The imported STEP model has non-roller connector solids between the
        # front and rear halves of level 3.  Their collision boxes form a hard
        # vertical stop even though the visible flow lane is continuous.  Keep
        # the connector visuals, both roller sections, side guides, and the end
        # stop; disable only those internal connector collision pieces.
        self._disabled_level3_connectors = []
        for collider in target_step_colliders:
            if collider.GetName() not in LEVEL3_CONNECTOR_NAMES:
                continue
            collider_range = bbox_cache.ComputeWorldBound(collider).ComputeAlignedRange()
            center = 0.5 * (collider_range.GetMin() + collider_range.GetMax())
            if (
                LEVEL3_CONNECTOR_X_RANGE[0] <= float(center[0]) <= LEVEL3_CONNECTOR_X_RANGE[1]
                and LEVEL3_CONNECTOR_Y_RANGE[0] <= float(center[1]) <= LEVEL3_CONNECTOR_Y_RANGE[1]
                and LEVEL3_CONNECTOR_Z_RANGE[0] <= float(center[2]) <= LEVEL3_CONNECTOR_Z_RANGE[1]
            ):
                UsdPhysics.CollisionAPI(collider).CreateCollisionEnabledAttr().Set(False)
                self._disabled_level3_connectors.append(str(collider.GetPath()))
        if not self._disabled_level3_connectors:
            raise RuntimeError("No level-3 internal connector colliders were found")
        self._box_root_to_leading_face = float(
            self._initial_box_position[0] - box_range.GetMin()[0]
        )
        self._carry_target_box_x = (
            self._rack_front_x + CARRY_RACK_CLEARANCE + self._box_root_to_leading_face
        )
        articulation_physx = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
        articulation_physx.CreateSolverPositionIterationCountAttr().Set(32)
        articulation_physx.CreateSolverVelocityIterationCountAttr().Set(4)

        physics_scenes = [
            prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)
        ]
        if not physics_scenes:
            raise RuntimeError("No physics scene found for CCD configuration")
        for physics_scene in physics_scenes:
            PhysxSchema.PhysxSceneAPI.Apply(
                physics_scene
            ).CreateEnableCCDAttr().Set(True)

        box_body = UsdPhysics.RigidBodyAPI.Apply(box_prim)
        box_body.CreateKinematicEnabledAttr().Set(True)
        box_body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        box_body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        UsdPhysics.MassAPI.Apply(box_prim).CreateMassAttr().Set(1.0)
        box_physx = PhysxSchema.PhysxRigidBodyAPI.Apply(box_prim)
        box_physx.CreateLinearDampingAttr().Set(0.2)
        box_physx.CreateAngularDampingAttr().Set(BOX_ANGULAR_DAMPING)
        box_physx.CreateEnableCCDAttr().Set(False)
        box_physx.CreateSolverPositionIterationCountAttr().Set(32)
        box_physx.CreateSolverVelocityIterationCountAttr().Set(8)
        box_colliders = [
            prim
            for prim in Usd.PrimRange(box_prim, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsInstanceProxy()
        ]
        if not box_colliders:
            raise RuntimeError("No editable grab_box colliders were found")
        self._box_colliders = box_colliders
        for collider in box_colliders:
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(collider)
            # A previous replay may have authored the drop-only values in the
            # session layer.  Remove those opinions so pickup and transport use
            # the original collider settings from lab.usd.
            physx_collision.GetContactOffsetAttr().Clear()
            physx_collision.GetRestOffsetAttr().Clear()

        self._target_rack_colliders = target_step_colliders
        for collider in target_step_colliders:
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(collider)
            physx_collision.GetContactOffsetAttr().Clear()
            physx_collision.GetRestOffsetAttr().Clear()

        robot_path = str(root_prim.GetPath()).split("/Geometry/")[0]
        background_bodies = [
            prim
            for prim in stage.Traverse()
            if prim != box_prim
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and not str(prim.GetPath()).startswith(robot_path)
        ]
        for prim in background_bodies:
            body = UsdPhysics.RigidBodyAPI(prim)
            # The nine open boxes on the other rack are presentation props
            # with free joints in the source MJCF.  Let gravity act on those
            # boxes; only the remaining environment rigid bodies stay pinned.
            free_fall_box = prim.GetName().lower().startswith("rack2_box_")
            body.CreateKinematicEnabledAttr().Set(not free_fall_box)
            body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

        wheel_links = {name.removesuffix("_joint") for name in WHEEL_JOINTS}
        wheel_colliders = [
            prim
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.CollisionAPI)
            and any(
                f"/{link}/" in str(prim.GetPath())
                or str(prim.GetPath()).endswith(f"/{link}")
                for link in wheel_links
            )
        ]
        if not wheel_colliders:
            raise RuntimeError("No wheel colliders were found")
        for collider in wheel_colliders:
            UsdPhysics.CollisionAPI(collider).CreateCollisionEnabledAttr().Set(False)

        arm_bodies = [
            prim
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and (prim.GetName().startswith("zarm_l") or prim.GetName().startswith("zarm_r"))
        ]
        palm_links = {
            unique_named_prim(stage, "zarm_l7_link").GetPath(),
            unique_named_prim(stage, "zarm_r7_link").GetPath(),
        }
        # The rendered fixed hands use conservative convex hulls.  Those hulls
        # protrude below the visible fingers and catch rack2's inclined lane
        # during insertion.  Filter only hand-vs-rack contact; grab_box keeps
        # its real collision with rack2's rack1-matched STEP geometry.
        for rack_prim in (target_rack, physical_rack):
            rack_filter = UsdPhysics.FilteredPairsAPI.Apply(rack_prim)
            rack_filter.CreateFilteredPairsRel().SetTargets(
                sorted(palm_links, key=lambda path: str(path))
            )
        box_filter = UsdPhysics.FilteredPairsAPI.Apply(box_prim)
        box_filter.CreateFilteredPairsRel().SetTargets(
            [prim.GetPath() for prim in arm_bodies if prim.GetPath() not in palm_links]
        )

        # The replacement fixed dexterous-hand model has a different swept
        # volume from the clamp model. During the inherited front-of-table
        # approach, its arm collision shapes catch the table apron before the
        # palms reach the box. Keep those imported arm shapes disabled; the two
        # rendered palm meshes are enabled separately below for grasping.
        arm_colliders = [
            prim
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.CollisionAPI)
            and any(
                f"/{body.GetName()}/" in str(prim.GetPath())
                or str(prim.GetPath()).endswith(f"/{body.GetName()}")
                for body in arm_bodies
            )
        ]
        if not arm_colliders:
            raise RuntimeError("No arm colliders were found")
        for collider in arm_colliders:
            UsdPhysics.CollisionAPI(collider).CreateCollisionEnabledAttr().Set(False)

        for name, translation in FIXED_THUMB_TRANSLATIONS.items():
            set_fixed_thumb_translation(stage, name, translation)
        for name, translation in FIXED_PALM_TRANSLATIONS.items():
            set_fixed_thumb_translation(stage, name, translation)
        for name in FIXED_FINGER_ROOTS:
            set_fixed_thumb_translation(stage, name, FIXED_FINGER_TRANSLATIONS[name])
            set_fixed_finger_curl(
                stage,
                name,
                FIXED_FINGER_BASE_ORIENTATIONS[name],
                FIXED_FINGER_CURL_DEGREES[name],
            )

        grip_material = UsdShade.Material.Define(stage, GRIP_MATERIAL_PATH)
        material_api = UsdPhysics.MaterialAPI.Apply(grip_material.GetPrim())
        material_api.CreateStaticFrictionAttr().Set(5.0)
        material_api.CreateDynamicFrictionAttr().Set(5.0)
        material_api.CreateRestitutionAttr().Set(0.0)
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(grip_material.GetPrim())
        physx_material.CreateFrictionCombineModeAttr().Set("max")
        physx_material.CreateRestitutionCombineModeAttr().Set("min")

        # The imported rack rollers are static collision meshes rather than
        # freely rotating rigid rollers.  Use a low-friction release material
        # after opening the hands so gravity can move the box down the measured
        # six-degree lane just like the free boxes on the opposite rack.
        release_material = UsdShade.Material.Define(stage, RELEASE_MATERIAL_PATH)
        release_material_api = UsdPhysics.MaterialAPI.Apply(release_material.GetPrim())
        release_material_api.CreateStaticFrictionAttr().Set(0.0)
        release_material_api.CreateDynamicFrictionAttr().Set(0.0)
        release_material_api.CreateRestitutionAttr().Set(0.0)
        release_physx_material = PhysxSchema.PhysxMaterialAPI.Apply(
            release_material.GetPrim()
        )
        release_physx_material.CreateFrictionCombineModeAttr().Set("min")
        release_physx_material.CreateRestitutionCombineModeAttr().Set("min")

        # No synthetic shelf/support is created. grab_box contacts rack2's
        # rack1-matched STEP collision parts directly.

        hand_meshes = []
        component_tokens = ("_palm", "_thumb", "_index", "_middle", "_ring", "_little")
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path = str(prim.GetPath())
            lower_name = prim.GetName().lower()
            is_hand_component = (
                ("/zarm_l7_link/" in path and lower_name.startswith("l_"))
                or ("/zarm_r7_link/" in path and lower_name.startswith("r_"))
            ) and any(token in lower_name for token in component_tokens)
            is_duplicate = prim.GetParent().GetName().endswith("_1")
            if is_hand_component and not is_duplicate:
                hand_meshes.append(prim)
        if len(hand_meshes) != 22:
            raise RuntimeError(
                f"Expected 22 rendered fixed-hand component meshes, found {len(hand_meshes)}"
            )
        grip_hand_meshes = [
            mesh for mesh in hand_meshes if "_thumb" not in mesh.GetName().lower()
        ]
        if len(grip_hand_meshes) != 18:
            raise RuntimeError(
                f"Expected 18 palm/finger grip meshes, found {len(grip_hand_meshes)}"
            )
        self._palm_collisions = [
            enable_hand_mesh_collider(mesh, grip_material) for mesh in grip_hand_meshes
        ]

        box_colliders = [
            prim
            for prim in Usd.PrimRange(box_prim)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not box_colliders:
            raise RuntimeError("No grab_box colliders were found")
        for collider in box_colliders:
            UsdShade.MaterialBindingAPI.Apply(collider).Bind(
                release_material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )

        self._initial_yaw = yaw_from_quaternion(self._initial_root_orientation)
        self._pickup_root_position = self._initial_root_position.copy()
        self._pickup_root_position[0] = self._initial_box_position[0] - PICKUP_BASE_TO_BOX_X
        self._pickup_root_position[1] = self._initial_box_position[1]
        displacement = self._pickup_root_position - self._initial_root_position
        displacement[2] = 0.0
        self._base_direction = displacement / np.linalg.norm(displacement)
        self._base_profile = TrapezoidalProfile(
            np.linalg.norm(displacement), BASE_MAX_SPEED, BASE_ACCELERATION
        )

        self._anchor = UsdPhysics.FixedJoint.Define(stage, "/DexhandBoxBaseAnchor")
        self._anchor.CreateBody1Rel().SetTargets([root_prim.GetPath()])
        self._anchor.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        self._anchor.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        set_anchor_pose(self._anchor, self._initial_root_position, self._initial_yaw)
        self._anchor.CreateJointEnabledAttr().Set(True)

    def _initialize_physics_handles(self) -> None:
        self._robot = Articulation(str(self._root_prim.GetPath()))
        self._box = RigidPrim(str(self._box_prim.GetPath()))
        self._used_gravity_drop = False
        self._names = list(self._robot.dof_names)
        if len(self._names) != 24 or set(self._names) != EXPECTED_DOFS:
            raise RuntimeError(f"Unexpected articulation DOFs: {self._names}")
        index = {name: self._names.index(name) for name in self._names}
        self._arm_indices = [index[name] for name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS]
        self._wheel_indices = [index[name] for name in WHEEL_JOINTS]
        self._controlled_indices = [
            index[name] for name in TORSO_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS
        ]
        self._torso_indices = [index[name] for name in TORSO_JOINTS]

        current_q = as_numpy(self._robot.get_dof_positions()).reshape(-1, 24)[0]
        if self._initial_dof_positions is None:
            # The demo's authored Home configuration is zero for all 24 DOFs.
            # Never adopt the live articulation pose seen when an extension is
            # enabled after a completed/paused run.
            self._initial_dof_positions = np.zeros_like(current_q)
        self._q_home = self._initial_dof_positions.copy()
        self._q_approach = self._q_home.copy()
        self._q_hold = self._q_home.copy()
        self._q_approach[self._arm_indices] = np.asarray(APPROACH_ARMS).reshape(-1)
        self._q_hold[self._arm_indices] = np.asarray(HOLD_ARMS).reshape(-1)
        self._lift_path = []
        for arms in LIFT_ARMS:
            pose = self._q_home.copy()
            pose[self._arm_indices] = np.asarray(arms).reshape(-1)
            self._lift_path.append(pose)
        self._lift_path = np.asarray(self._lift_path)
        self._q_rack_high = self._lift_path[-1].copy()
        self._q_rack_high[self._arm_indices] = np.asarray(RACK_CLEAR_ARMS).reshape(-1)
        self._q_rack_high[self._torso_indices[:3]] = np.asarray(RACK_HIGH_TORSO)
        self._rack_elevate_path = [self._lift_path[-1].copy()]
        rack_torso = np.asarray(RACK_HIGH_TORSO)
        for waypoint, arms in enumerate(RACK_ELEVATE_ARM_PATH, start=1):
            fraction = waypoint / len(RACK_ELEVATE_ARM_PATH)
            pose = self._lift_path[-1].copy()
            pose[self._arm_indices] = np.asarray(arms).reshape(-1)
            pose[self._torso_indices[:3]] = (
                self._q_home[self._torso_indices[:3]]
                + (rack_torso - self._q_home[self._torso_indices[:3]]) * fraction
            )
            self._rack_elevate_path.append(pose)
        self._rack_elevate_path = np.asarray(self._rack_elevate_path)
        # Raise the upright torso by 80 mm while the arms descend 210 mm in its
        # frame. The palms therefore descend 130 mm in world space, while the
        # elbow links retain clearance above rack2's front rollers.
        self._place_path = [self._q_rack_high.copy()]
        rack_torso = np.asarray(RACK_HIGH_TORSO)
        place_torso = np.asarray(PLACE_HIGH_TORSO)
        for waypoint, arms in enumerate(PLACE_ARM_PATH, start=1):
            fraction = waypoint / len(PLACE_ARM_PATH)
            pose = self._q_rack_high.copy()
            pose[self._arm_indices] = np.asarray(arms).reshape(-1)
            pose[self._torso_indices[:3]] = (
                rack_torso + (place_torso - rack_torso) * fraction
            )
            self._place_path.append(pose)
        self._place_path = np.asarray(self._place_path)

        stiffness, damping = (as_numpy(v).reshape(-1, 24)[0] for v in self._robot.get_dof_gains())
        max_effort = as_numpy(self._robot.get_dof_max_efforts()).reshape(-1, 24)[0]
        armature = as_numpy(self._robot.get_dof_armatures()).reshape(-1, 24)[0]
        for dof in self._arm_indices:
            stiffness[dof] = 2000.0
            damping[dof] = 400.0
            max_effort[dof] = 300.0
            armature[dof] = 10.0
        for name in TORSO_JOINTS:
            dof = index[name]
            stiffness[dof] = 10000.0
            damping[dof] = 1000.0
            max_effort[dof] = 2000.0
            armature[dof] = 20.0
        for name in HEAD_JOINTS:
            dof = index[name]
            stiffness[dof] = 1000.0
            damping[dof] = 100.0
            max_effort[dof] = 200.0
            armature[dof] = 2.0
        for dof in self._wheel_indices:
            stiffness[dof] = 0.0
            damping[dof] = 400.0
            armature[dof] = 5.0
        self._robot.set_dof_gains(stiffness, damping)
        self._robot.set_dof_max_efforts(max_effort)
        self._robot.set_dof_armatures(armature)
        self._robot.set_dof_positions(self._q_home)
        self._robot.set_dof_velocities(np.zeros(24))
        self._robot.set_dof_position_targets(self._q_home)
        self._box.set_world_poses(self._initial_box_position, self._initial_box_orientation)
        self._pitch_velocity_command = np.zeros(4, dtype=float)
        self._start_q = self._q_home.copy()

    def _activate_physical_grip(self) -> None:
        box_positions, box_orientations = self._box.get_world_poses()
        box_position = as_numpy(box_positions).reshape(-1, 3)[0]
        self._grasp_start_box_orientation = as_numpy(box_orientations).reshape(-1, 4)[0]
        self._max_transport_tilt = 0.0
        self._max_transport_tilt_phase = "GRASP_ATTACH"
        self._max_transport_tilt_elapsed = 0.0

        body = UsdPhysics.RigidBodyAPI(self._box_prim)
        body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        body.CreateKinematicEnabledAttr().Set(False)
        box_physx = PhysxSchema.PhysxRigidBodyAPI.Apply(self._box_prim)
        # Preserve the original grasp/transport contact response. CCD is only
        # needed for the short free fall onto the imported thin rack geometry.
        box_physx.CreateEnableCCDAttr().Set(False)
        box_physx.CreateAngularDampingAttr().Set(BOX_ANGULAR_DAMPING)
        self._grasp_start_box_position = box_position.copy()

    def _enable_drop_collision_protection(self) -> None:
        """Enable CCD only for the released-box descent."""
        PhysxSchema.PhysxRigidBodyAPI.Apply(
            self._box_prim
        ).CreateEnableCCDAttr().Set(True)

    def _clear_box_drop_damping(self) -> None:
        """Expose the authored/default damping so the released box can slide."""
        box_physx = PhysxSchema.PhysxRigidBodyAPI.Apply(self._box_prim)
        box_physx.GetLinearDampingAttr().Clear()
        box_physx.GetAngularDampingAttr().Clear()

    def _apply_rack_roller_assist(self) -> None:
        """Approximate free roller rotation until the box crosses both sections."""
        box_position = as_numpy(self._box.get_world_poses()[0]).reshape(-1, 3)[0]
        if float(box_position[0]) > RACK_ROLL_ASSIST_TARGET_X:
            self._box.apply_forces(
                np.asarray([[-RACK_ROLL_ASSIST_FORCE, 0.0, 0.0]], dtype=float)
            )

    def _start_demo(self) -> None:
        self._begin_session_edit()
        self._start_q = as_numpy(self._robot.get_dof_positions()).reshape(-1, 24)[0]
        self._transition("BASE_APPROACH")
        self._timeline.play()

    def _on_update(self, event) -> None:
        if self._phase == "RESET_PENDING":
            try:
                self._step_reset_pending()
            except Exception as exc:
                self._fail(exc)
            return
        if not self._timeline.is_playing() or self._phase in {
            "DETACHED", "ATTACHED", "READY", "HOLD_READY", "LIFT_READY",
            "TURN_READY", "CARRY_READY", "DEMO_COMPLETE", "PAUSED", "ERROR",
        }:
            return
        try:
            dt = max(0.0, min(float(event["dt"]), 0.1))
        except (KeyError, TypeError, ValueError):
            dt = 1.0 / 60.0
        try:
            if self._phase == "RESETTING":
                self._step_reset(dt)
            else:
                self._elapsed += dt
                self._step_demo(max(dt, 1.0 / 120.0))
        except Exception as exc:
            self._fail(exc)

    def _fail(self, exc: Exception) -> None:
        carb.log_error(f"Kuavo Dexhand Box Carry Demo: {exc}")
        self._timeline.pause()
        self._phase = "ERROR"
        self._set_status(f"ERROR - {exc}")

    def _step_reset_pending(self) -> None:
        if self._timeline.is_playing():
            self._timeline.stop()
            return
        if self._reset_wait_frames > 0:
            self._reset_wait_frames -= 1
            return
        if not self._reset_fabric_disabled:
            SimulationManager.set_physics_sim_device("cpu")
            SimulationManager.set_physics_dt(1.0 / 120.0)
            SimulationManager.enable_fabric(False)
            carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
            self._reset_fabric_disabled = True
            self._reset_wait_frames = 2
            self._set_status("RESETTING - configuring 120 Hz CPU PhysX")
            return
        self._configure_session()
        self._phase = "RESETTING"
        self._elapsed = 0.0
        self._set_status("RESETTING - initializing PhysX handles")
        self._timeline.play()

    def _step_reset(self, dt: float) -> None:
        if not self._reset_initialized:
            self._initialize_physics_handles()
            self._reset_initialized = True
            self._elapsed = 0.0
        self._elapsed += dt
        set_anchor_pose(self._anchor, self._initial_root_position, self._initial_yaw)
        self._robot.set_dof_position_targets(
            self._q_home[self._controlled_indices], dof_indices=self._controlled_indices
        )
        self._robot.set_dof_velocity_targets(0.0, dof_indices=self._arm_indices)
        self._robot.set_dof_velocity_targets(0.0, dof_indices=self._wheel_indices)
        if self._elapsed >= RESET_SETTLE_SECONDS:
            if self._auto_play_after_reset:
                self._auto_play_after_reset = False
                self._start_demo()
            else:
                self._timeline.pause()
                self._phase = "READY"
                self._set_status("READY - click Play From Start")

    def _step_demo(self, command_dt: float) -> None:
        q_command = self._q_home.copy()
        qd_command = np.zeros(24, dtype=float)
        wheel_velocity_target = np.zeros(4, dtype=float)
        elapsed = self._elapsed

        if self._phase in {
            "PLACE_BOX_SETTLE",
            "RETREAT_POSE_SETTLE",
            "BASE_RETREAT_HIGH",
            "ARM_RETURN",
            "TORSO_LOWER",
        }:
            self._apply_rack_roller_assist()

        if self._phase in {
            "GRASP_ATTACH", "LIFT", "LIFT_SETTLE", "PICKUP_RETREAT",
            "PICKUP_RETREAT_SETTLE", "TURN", "TURN_SETTLE",
            "CARRY", "CARRY_SETTLE", "RACK_ELEVATE", "RACK_ELEVATE_SETTLE",
            "RACK_INSERT", "RACK_INSERT_SETTLE", "PLACE_LOWER",
            "PLACE_SUPPORT_SETTLE",
        } and hasattr(self, "_grasp_start_box_orientation"):
            current_orientation = as_numpy(self._box.get_world_poses()[1]).reshape(-1, 4)[0]
            reference_up = quaternion_matrix(self._grasp_start_box_orientation)[:, 2]
            current_up = quaternion_matrix(current_orientation)[:, 2]
            tilt = math.acos(float(np.clip(np.dot(reference_up, current_up), -1.0, 1.0)))
            if tilt > self._max_transport_tilt:
                self._max_transport_tilt = tilt
                self._max_transport_tilt_phase = self._phase
                self._max_transport_tilt_elapsed = elapsed

        if self._phase == "BASE_APPROACH":
            distance, velocity = self._base_profile.evaluate(elapsed)
            position = self._initial_root_position + self._base_direction * distance
            set_anchor_pose(self._anchor, position, self._initial_yaw)
            wheel_velocity_target.fill(velocity / WHEEL_RADIUS)
            if elapsed >= self._base_profile.total_time:
                set_anchor_pose(self._anchor, self._pickup_root_position, self._initial_yaw)
                self._transition("BASE_APPROACH_HOLD")
        elif self._phase == "BASE_APPROACH_HOLD":
            set_anchor_pose(self._anchor, self._pickup_root_position, self._initial_yaw)
            if elapsed >= APPROACH_HOLD_SECONDS:
                self._start_q = as_numpy(self._robot.get_dof_positions()).reshape(-1, 24)[0]
                self._transition("ARM_APPROACH")
        elif self._phase == "ARM_APPROACH":
            s, sd = quintic(elapsed, ARM_APPROACH_SECONDS)
            q_command = self._start_q + (self._q_approach - self._start_q) * s
            qd_command = (self._q_approach - self._start_q) * sd
            if elapsed >= ARM_APPROACH_SECONDS:
                self._start_q = as_numpy(self._robot.get_dof_positions()).reshape(-1, 24)[0]
                self._transition("HAND_HOLD")
        elif self._phase == "HAND_HOLD":
            s, sd = quintic(elapsed, HAND_HOLD_SECONDS)
            q_command = self._start_q + (self._q_hold - self._start_q) * s
            qd_command = (self._q_hold - self._start_q) * sd
            if elapsed >= HAND_HOLD_SECONDS:
                self._transition("HAND_SETTLE")
        elif self._phase == "HAND_SETTLE":
            q_command = self._q_hold.copy()
            actual = as_numpy(self._robot.get_dof_positions()).reshape(-1, 24)[0]
            arm_error = float(
                np.max(np.abs(actual[self._arm_indices] - self._q_hold[self._arm_indices]))
            )
            if elapsed >= HAND_SETTLE_SECONDS and arm_error <= ARM_TARGET_TOLERANCE:
                self._activate_physical_grip()
                self._transition("GRASP_ATTACH", "GRASP_CONTACT - grab_box is now a dynamic rigid body")
            if elapsed >= HAND_SETTLE_TIMEOUT_SECONDS:
                raise RuntimeError(
                    f"Arms did not reach the hold pose (max error {arm_error:.3f} rad)"
                )
        elif self._phase == "GRASP_ATTACH":
            q_command = self._q_hold.copy()
            if elapsed >= GRASP_ATTACH_SECONDS:
                self._transition("LIFT")
        elif self._phase == "LIFT":
            s, sd = quintic(elapsed, LIFT_SECONDS)
            path_position = s * (len(self._lift_path) - 1)
            segment = min(int(path_position), len(self._lift_path) - 2)
            blend = path_position - segment
            delta = self._lift_path[segment + 1] - self._lift_path[segment]
            q_command = self._lift_path[segment] + delta * blend
            qd_command = delta * (len(self._lift_path) - 1) * sd
            if elapsed >= LIFT_SECONDS:
                self._transition("LIFT_SETTLE")
        elif self._phase == "LIFT_SETTLE":
            q_command = self._lift_path[-1].copy()
            if elapsed >= LIFT_SETTLE_SECONDS:
                box_position = as_numpy(self._box.get_world_poses()[0]).reshape(-1, 3)[0]
                lift_height = float(box_position[2] - self._grasp_start_box_position[2])
                if lift_height < MINIMUM_LIFT_HEIGHT:
                    raise RuntimeError(
                        f"Box lift was too small ({lift_height:.3f} m)"
                    )
                root_positions, root_orientations = self._robot.get_world_poses()
                retreat_start = as_numpy(root_positions).reshape(-1, 3)[0]
                self._pickup_retreat_start_position = retreat_start.copy()
                self._pickup_retreat_target_position = (
                    retreat_start - self._base_direction * PICKUP_RETREAT_DISTANCE
                )
                self._pickup_retreat_profile = TrapezoidalProfile(
                    PICKUP_RETREAT_DISTANCE,
                    PICKUP_RETREAT_MAX_SPEED,
                    PICKUP_RETREAT_ACCELERATION,
                )
                self._pickup_retreat_yaw = yaw_from_quaternion(
                    as_numpy(root_orientations).reshape(-1, 4)[0]
                )
                self._turn_start_box_position = box_position.copy()
                self._transition(
                    "PICKUP_RETREAT",
                    f"PICKUP_RETREAT - backing {PICKUP_RETREAT_DISTANCE:.2f} m clear "
                    f"of the box stack before turning",
                )
        elif self._phase == "PICKUP_RETREAT":
            q_command = self._lift_path[-1].copy()
            distance, velocity = self._pickup_retreat_profile.evaluate(elapsed)
            position = self._pickup_retreat_start_position - self._base_direction * distance
            set_anchor_pose(self._anchor, position, self._pickup_retreat_yaw)
            wheel_velocity_target.fill(-velocity / WHEEL_RADIUS)
            if elapsed >= self._pickup_retreat_profile.total_time:
                set_anchor_pose(
                    self._anchor,
                    self._pickup_retreat_target_position,
                    self._pickup_retreat_yaw,
                )
                self._transition("PICKUP_RETREAT_SETTLE")
        elif self._phase == "PICKUP_RETREAT_SETTLE":
            q_command = self._lift_path[-1].copy()
            set_anchor_pose(
                self._anchor,
                self._pickup_retreat_target_position,
                self._pickup_retreat_yaw,
            )
            if elapsed >= PICKUP_RETREAT_SETTLE_SECONDS:
                box_position = as_numpy(self._box.get_world_poses()[0]).reshape(-1, 3)[0]
                self._turn_start_position = self._pickup_retreat_target_position.copy()
                self._turn_start_yaw = self._pickup_retreat_yaw
                self._turn_start_box_position = box_position.copy()
                self._transition(
                    "TURN",
                    "TURN - pickup table and neighbouring box stack are clear",
                )
        elif self._phase == "TURN":
            q_command = self._lift_path[-1].copy()
            angle, angular_velocity = self._turn_profile.evaluate(elapsed)
            yaw = self._turn_start_yaw + angle
            set_anchor_pose(self._anchor, self._turn_start_position, yaw)
            wheel_speed = angular_velocity * TURN_WHEEL_TRACK_RADIUS / WHEEL_RADIUS
            wheel_velocity_target[:] = (-wheel_speed, wheel_speed, -wheel_speed, wheel_speed)
            if elapsed >= self._turn_profile.total_time:
                set_anchor_pose(
                    self._anchor,
                    self._turn_start_position,
                    self._turn_start_yaw + TURN_ANGLE,
                )
                self._transition("TURN_SETTLE")
        elif self._phase == "TURN_SETTLE":
            q_command = self._lift_path[-1].copy()
            set_anchor_pose(
                self._anchor,
                self._turn_start_position,
                self._turn_start_yaw + TURN_ANGLE,
            )
            if elapsed >= TURN_SETTLE_SECONDS:
                box_position = as_numpy(self._box.get_world_poses()[0]).reshape(-1, 3)[0]
                box_drop = float(self._turn_start_box_position[2] - box_position[2])
                lift_height = float(box_position[2] - self._grasp_start_box_position[2])
                root_orientation = as_numpy(self._robot.get_world_poses()[1]).reshape(-1, 4)[0]
                yaw_error = abs(
                    wrap_angle(
                        self._turn_start_yaw
                        + TURN_ANGLE
                        - yaw_from_quaternion(root_orientation)
                    )
                )
                if box_drop > MAXIMUM_TURN_BOX_DROP:
                    raise RuntimeError(
                        f"Box slipped {box_drop:.3f} m during the turn"
                    )
                if lift_height < MINIMUM_LIFT_HEIGHT:
                    raise RuntimeError(
                        f"Box lift after turn was too small ({lift_height:.3f} m)"
                    )
                root_position = as_numpy(self._robot.get_world_poses()[0]).reshape(-1, 3)[0]
                box_offset = box_position - root_position
                self._carry_start_position = root_position.copy()
                self._carry_start_box_position = box_position.copy()
                self._carry_target_position = root_position.copy()
                self._carry_target_position[0] = self._carry_target_box_x - box_offset[0]
                displacement = self._carry_target_position - self._carry_start_position
                displacement[2] = 0.0
                carry_distance = float(np.linalg.norm(displacement))
                if carry_distance < 0.05 or displacement[0] >= 0.0:
                    raise RuntimeError(
                        f"Invalid rack approach displacement {displacement.tolist()}"
                    )
                if abs(float(displacement[1])) > 0.01:
                    raise RuntimeError("Rack transport must remain a straight X-axis motion")
                self._carry_direction = displacement / carry_distance
                self._carry_profile = TrapezoidalProfile(
                    carry_distance, CARRY_MAX_SPEED, CARRY_ACCELERATION
                )
                self._transition(
                    "CARRY",
                    f"CARRY - straight transport {carry_distance:.3f} m to rack2 pre-place pose",
                )
        elif self._phase == "CARRY":
            q_command = self._lift_path[-1].copy()
            distance, velocity = self._carry_profile.evaluate(elapsed)
            position = self._carry_start_position + self._carry_direction * distance
            set_anchor_pose(self._anchor, position, self._turn_start_yaw + TURN_ANGLE)
            wheel_velocity_target.fill(velocity / WHEEL_RADIUS)
            if elapsed >= self._carry_profile.total_time:
                set_anchor_pose(
                    self._anchor,
                    self._carry_target_position,
                    self._turn_start_yaw + TURN_ANGLE,
                )
                self._transition("CARRY_SETTLE")
        elif self._phase == "CARRY_SETTLE":
            q_command = self._lift_path[-1].copy()
            set_anchor_pose(
                self._anchor,
                self._carry_target_position,
                self._turn_start_yaw + TURN_ANGLE,
            )
            if elapsed >= CARRY_SETTLE_SECONDS:
                box_position = as_numpy(self._box.get_world_poses()[0]).reshape(-1, 3)[0]
                box_drop = float(self._carry_start_box_position[2] - box_position[2])
                root_position = as_numpy(self._robot.get_world_poses()[0]).reshape(-1, 3)[0]
                target_error = float(np.linalg.norm(root_position[:2] - self._carry_target_position[:2]))
                if box_drop > MAXIMUM_CARRY_BOX_DROP:
                    raise RuntimeError(
                        f"Box slipped {box_drop:.3f} m during rack transport"
                    )
                if target_error > 0.02:
                    raise RuntimeError(
                        f"Rack pre-place position error was {target_error:.3f} m"
                    )
                self._rack_elevate_start_q = as_numpy(
                    self._robot.get_dof_positions()
                ).reshape(-1, 24)[0]
                self._rack_elevate_start_box_z = float(box_position[2])
                self._transition(
                    "RACK_ELEVATE",
                    f"RACK_ELEVATE - coordinated torso lift for level 3, box drop {box_drop:.3f} m",
                )
        elif self._phase == "RACK_ELEVATE":
            s, sd = quintic(elapsed, RACK_ELEVATE_SECONDS)
            path_position = s * (len(self._rack_elevate_path) - 1)
            segment = min(int(path_position), len(self._rack_elevate_path) - 2)
            blend = path_position - segment
            delta = self._rack_elevate_path[segment + 1] - self._rack_elevate_path[segment]
            q_command = self._rack_elevate_path[segment] + delta * blend
            qd_command = delta * (len(self._rack_elevate_path) - 1) * sd
            if elapsed >= RACK_ELEVATE_SECONDS:
                self._transition("RACK_ELEVATE_SETTLE")
        elif self._phase == "RACK_ELEVATE_SETTLE":
            q_command = self._q_rack_high.copy()
            if elapsed >= RACK_ELEVATE_SETTLE_SECONDS:
                box_position = as_numpy(self._box.get_world_poses()[0]).reshape(-1, 3)[0]
                elevation = float(box_position[2] - self._rack_elevate_start_box_z)
                shelf_clearance = float(box_position[2] - self._level3_surface_z)
                if elevation < 0.21 or shelf_clearance < 0.04:
                    raise RuntimeError(
                        f"Level-3 lift clearance was too small ({shelf_clearance:.3f} m)"
                    )
                root_position = as_numpy(self._robot.get_world_poses()[0]).reshape(-1, 3)[0]
                box_offset = box_position - root_position
                self._rack_insert_start_position = root_position.copy()
                self._rack_insert_target_position = root_position.copy()
                self._rack_insert_target_position[0] = RACK_INSERT_BOX_X - box_offset[0]
                displacement = self._rack_insert_target_position - root_position
                displacement[2] = 0.0
                distance = float(np.linalg.norm(displacement))
                self._rack_insert_direction = displacement / distance
                self._rack_insert_profile = TrapezoidalProfile(
                    distance, RACK_INSERT_MAX_SPEED, RACK_INSERT_ACCELERATION
                )
                self._transition(
                    "RACK_INSERT",
                    f"RACK_INSERT - box entering level 3 by {distance:.3f} m",
                )
        elif self._phase == "RACK_INSERT":
            q_command = self._q_rack_high.copy()
            distance, velocity = self._rack_insert_profile.evaluate(elapsed)
            position = self._rack_insert_start_position + self._rack_insert_direction * distance
            set_anchor_pose(self._anchor, position, self._turn_start_yaw + TURN_ANGLE)
            wheel_velocity_target.fill(velocity / WHEEL_RADIUS)
            if elapsed >= self._rack_insert_profile.total_time:
                set_anchor_pose(
                    self._anchor,
                    self._rack_insert_target_position,
                    self._turn_start_yaw + TURN_ANGLE,
                )
                self._transition("RACK_INSERT_SETTLE")
        elif self._phase == "RACK_INSERT_SETTLE":
            q_command = self._q_rack_high.copy()
            set_anchor_pose(
                self._anchor,
                self._rack_insert_target_position,
                self._turn_start_yaw + TURN_ANGLE,
            )
            if elapsed >= RACK_INSERT_SETTLE_SECONDS:
                box_position = as_numpy(self._box.get_world_poses()[0]).reshape(-1, 3)[0]
                if abs(float(box_position[0] - RACK_INSERT_BOX_X)) > 0.03:
                    raise RuntimeError("Box did not reach the level-3 insertion depth")
                PhysxSchema.PhysxRigidBodyAPI.Apply(
                    self._box_prim
                ).CreateAngularDampingAttr().Set(PLACE_BOX_ANGULAR_DAMPING)
                self._transition("PLACE_LOWER")
        elif self._phase == "PLACE_LOWER":
            s, sd = quintic(elapsed, PLACE_LOWER_SECONDS)
            path_position = s * (len(self._place_path) - 1)
            segment = min(int(path_position), len(self._place_path) - 2)
            blend = path_position - segment
            delta = self._place_path[segment + 1] - self._place_path[segment]
            q_command = self._place_path[segment] + delta * blend
            qd_command = delta * (len(self._place_path) - 1) * sd
            set_anchor_pose(
                self._anchor,
                self._rack_insert_target_position,
                self._turn_start_yaw + TURN_ANGLE,
            )
            box_position = as_numpy(self._box.get_world_poses()[0]).reshape(-1, 3)[0]
            release_before_contact = (
                elapsed >= 0.25
                and box_position[2]
                <= self._level3_surface_z + PLACE_GRAVITY_DROP_TRIGGER_CLEARANCE
            )
            if release_before_contact or elapsed >= PLACE_LOWER_TIMEOUT_SECONDS:
                actual_q = as_numpy(self._robot.get_dof_positions()).reshape(-1, 24)[0]
                arm_error = float(
                    np.max(
                        np.abs(
                            actual_q[self._arm_indices]
                            - self._place_path[-1][self._arm_indices]
                        )
                    )
                )
                # Stop before visual contact instead of letting the box press
                # against the imported rack until the old 18-second timeout.
                # The timeout remains only as a safety fallback for unusually
                # slow interactive articulation tracking.
                self._q_place = actual_q.copy()
                self._release_start_q = actual_q.copy()
                self._q_release = actual_q.copy()
                self._place_base_position = self._rack_insert_target_position.copy()
                self._place_torso_error = float(
                    np.max(
                        np.abs(
                            actual_q[self._torso_indices]
                            - q_command[self._torso_indices]
                        )
                    )
                )
                self._enable_drop_collision_protection()
                for collision in self._palm_collisions:
                    collision.CreateCollisionEnabledAttr().Set(False)
                PhysxSchema.PhysxRigidBodyAPI.Apply(
                    self._box_prim
                ).CreateLinearDampingAttr().Set(PLACE_GRAVITY_DROP_LINEAR_DAMPING)
                self._used_gravity_drop = True
                q_command = self._q_place.copy()
                qd_command.fill(0.0)
                self._transition(
                    "PLACE_GRAVITY_DROP",
                    f"PLACE_GRAVITY_DROP - pre-contact release; box descending "
                    f"{box_position[2] - self._level3_surface_z:.3f} m to level 3 "
                    f"(arm lag {arm_error:.3f} rad)",
                )
        elif self._phase == "PLACE_GRAVITY_DROP":
            q_command = self._q_place.copy()
            set_anchor_pose(
                self._anchor,
                self._place_base_position,
                self._turn_start_yaw + TURN_ANGLE,
            )
            box_position = as_numpy(self._box.get_world_poses()[0]).reshape(-1, 3)[0]
            if (
                box_position[2]
                <= self._level3_surface_z + PLACE_GRAVITY_DROP_CONTACT_CLEARANCE
            ):
                self._clear_box_drop_damping()
                self._box_first_rack_contact_position = box_position.copy()
                # The imported flow rack is split into two roller sections.
                # Its non-rotating STEP meshes contain a small connector gap
                # around x=-2.32 m that otherwise catches the leading box edge.
                # A single modest downslope entry velocity emulates the missing
                # roller rotation; after this impulse the box remains governed
                # only by gravity and the original rack collisions.
                self._box.set_velocities(
                    linear_velocities=np.asarray(
                        [[-RELEASE_ROLLER_ENTRY_SPEED, 0.0, 0.0]], dtype=float
                    ),
                    angular_velocities=np.zeros((1, 3), dtype=float),
                )
                self._transition(
                    "PLACE_BOX_SETTLE",
                    "PLACE_BOX_SETTLE - box reached level 3 and is free to slide",
                )
            elif elapsed >= PLACE_GRAVITY_DROP_TIMEOUT_SECONDS:
                raise RuntimeError(
                    f"Released box did not reach level 3 (bottom z={box_position[2]:.3f} m)"
                )
        elif self._phase == "PLACE_SUPPORT_SETTLE":
            q_command = self._q_place.copy()
            set_anchor_pose(
                self._anchor,
                self._place_base_position,
                self._turn_start_yaw + TURN_ANGLE,
            )
            if elapsed >= PLACE_SUPPORT_SETTLE_SECONDS:
                self._release_start_q = as_numpy(
                    self._robot.get_dof_positions()
                ).reshape(-1, 24)[0]
                self._q_release = self._release_start_q.copy()
                release_delta = np.asarray(RELEASE_ARMS).reshape(-1) - np.asarray(
                    LIFT_ARMS[-1]
                ).reshape(-1)
                self._q_release[self._arm_indices] += release_delta
                self._transition(
                    "PLACE_RELEASE",
                    "PLACE_RELEASE - box is supported, opening both fixed-hand jaws",
                )
        elif self._phase == "PLACE_RELEASE":
            s, sd = quintic(elapsed, PLACE_RELEASE_SECONDS)
            q_command = self._release_start_q + (self._q_release - self._release_start_q) * s
            qd_command = (self._q_release - self._release_start_q) * sd
            set_anchor_pose(
                self._anchor,
                self._place_base_position,
                self._turn_start_yaw + TURN_ANGLE,
            )
            if elapsed >= PLACE_RELEASE_SECONDS:
                self._transition("PLACE_BOX_SETTLE")
        elif self._phase == "PLACE_BOX_SETTLE":
            q_command = self._q_release.copy()
            set_anchor_pose(
                self._anchor,
                self._place_base_position,
                self._turn_start_yaw + TURN_ANGLE,
            )
            if elapsed >= PLACE_BOX_SETTLE_SECONDS:
                self._placed_box_position = as_numpy(
                    self._box.get_world_poses()[0]
                ).reshape(-1, 3)[0]
                linear_velocity = as_numpy(self._box.get_velocities()[0]).reshape(-1, 3)[0]
                self._release_slide_speed = float(np.linalg.norm(linear_velocity))
                if abs(float(self._placed_box_position[2] - self._level3_surface_z)) > 0.05:
                    raise RuntimeError(
                        f"Placed box bottom was not on level 3 ({self._placed_box_position[2]:.3f} m)"
                    )
                self._post_release_slide_distance = float(
                    np.linalg.norm(
                        self._placed_box_position[:2]
                        - self._box_first_rack_contact_position[:2]
                    )
                )
                # The hand mesh colliders are needed only while carrying the
                # box. Once it is supported and sliding freely, keep the
                # complete arm pose fixed and back farther away before folding it.
                for collision in self._palm_collisions:
                    collision.CreateCollisionEnabledAttr().Set(False)
                self._q_retreat_hold = as_numpy(
                    self._robot.get_dof_positions()
                ).reshape(-1, 24)[0]
                retreat_to_carry = self._carry_target_position - self._place_base_position
                retreat_to_carry[2] = 0.0
                retreat_direction = retreat_to_carry / np.linalg.norm(retreat_to_carry)
                self._retreat_target_position = (
                    self._carry_target_position
                    + retreat_direction * POST_PLACE_EXTRA_RETREAT
                )
                self._retreat_delta = (
                    self._retreat_target_position - self._place_base_position
                )
                self._retreat_delta[2] = 0.0
                self._transition(
                    "RETREAT_POSE_SETTLE",
                    "RETREAT_POSE_SETTLE - damping the extended-arm pose before moving",
                )
        elif self._phase == "RETREAT_POSE_SETTLE":
            q_command = self._q_retreat_hold.copy()
            set_anchor_pose(
                self._anchor,
                self._place_base_position,
                self._turn_start_yaw + TURN_ANGLE,
            )
            if elapsed >= RETREAT_POSE_SETTLE_SECONDS:
                # Capture the fully damped pose once more so the retreat does
                # not fight a residual articulation oscillation.
                self._q_retreat_hold = as_numpy(
                    self._robot.get_dof_positions()
                ).reshape(-1, 24)[0]
                self._retreat_wheel_start_positions = self._q_retreat_hold[
                    self._wheel_indices
                ].copy()
                self._transition(
                    "BASE_RETREAT_HIGH",
                    "BASE_RETREAT_HIGH - smooth zero-jerk retreat with arms held still",
                )
        elif self._phase == "BASE_RETREAT_HIGH":
            q_command = self._q_retreat_hold.copy()
            s, sd = quintic(elapsed, RETREAT_SECONDS)
            position = self._place_base_position + self._retreat_delta * s
            set_anchor_pose(self._anchor, position, self._turn_start_yaw + TURN_ANGLE)
            retreat_speed = float(np.linalg.norm(self._retreat_delta[:2])) * sd
            wheel_velocity_target.fill(-retreat_speed / WHEEL_RADIUS)
            retreat_distance = float(np.linalg.norm(self._retreat_delta[:2])) * s
            self._post_place_wheel_rotation = retreat_distance / WHEEL_RADIUS
            wheel_positions = (
                self._retreat_wheel_start_positions
                - self._post_place_wheel_rotation
                + math.pi
            ) % (2.0 * math.pi) - math.pi
            self._robot.set_dof_positions(
                wheel_positions,
                dof_indices=self._wheel_indices,
            )
            self._robot.set_dof_velocities(
                np.full(4, -retreat_speed / WHEEL_RADIUS, dtype=float),
                dof_indices=self._wheel_indices,
            )
            if elapsed >= RETREAT_SECONDS:
                set_anchor_pose(
                    self._anchor,
                    self._retreat_target_position,
                    self._turn_start_yaw + TURN_ANGLE,
                )
                self._post_place_retreat_distance = float(
                    np.linalg.norm(
                        self._retreat_target_position[:2]
                        - self._place_base_position[:2]
                    )
                )
                self._arm_return_start_q = as_numpy(
                    self._robot.get_dof_positions()
                ).reshape(-1, 24)[0]
                self._q_arms_lowered = self._arm_return_start_q.copy()
                self._q_arms_lowered[self._arm_indices] = self._q_home[
                    self._arm_indices
                ]
                self._transition(
                    "ARM_RETURN",
                    "ARM_RETURN - rack is clear, lowering both arms while torso stays high",
                )
        elif self._phase == "ARM_RETURN":
            s, sd = quintic(elapsed, ARM_RETURN_SECONDS)
            q_command = self._arm_return_start_q + (
                self._q_arms_lowered - self._arm_return_start_q
            ) * s
            qd_command = (self._q_arms_lowered - self._arm_return_start_q) * sd
            set_anchor_pose(
                self._anchor,
                self._retreat_target_position,
                self._turn_start_yaw + TURN_ANGLE,
            )
            if elapsed >= ARM_RETURN_SECONDS:
                self._torso_lower_start_q = as_numpy(
                    self._robot.get_dof_positions()
                ).reshape(-1, 24)[0]
                self._transition(
                    "TORSO_LOWER",
                    "TORSO_LOWER - arms are down and rack is clear, restoring the torso",
                )
        elif self._phase == "TORSO_LOWER":
            s, sd = quintic(elapsed, TORSO_LOWER_SECONDS)
            q_command = self._torso_lower_start_q + (
                self._q_home - self._torso_lower_start_q
            ) * s
            qd_command = (self._q_home - self._torso_lower_start_q) * sd
            set_anchor_pose(
                self._anchor,
                self._retreat_target_position,
                self._turn_start_yaw + TURN_ANGLE,
            )
            if elapsed >= TORSO_LOWER_SECONDS:
                set_anchor_pose(
                    self._anchor,
                    self._retreat_target_position,
                    self._turn_start_yaw + TURN_ANGLE,
                )
                self._timeline.pause()
                if self._max_transport_tilt > MAXIMUM_TRANSPORT_TILT:
                    raise RuntimeError(
                        f"Box tilt reached {math.degrees(self._max_transport_tilt):.2f} deg "
                        f"during {self._max_transport_tilt_phase} at "
                        f"{self._max_transport_tilt_elapsed:.2f} s"
                    )
                actual_q = as_numpy(self._robot.get_dof_positions()).reshape(-1, 24)[0]
                controlled_error = np.abs(
                    actual_q[self._controlled_indices]
                    - self._q_home[self._controlled_indices]
                )
                final_pose_error = float(np.max(controlled_error))
                worst_controlled = int(np.argmax(controlled_error))
                worst_dof = self._controlled_indices[worst_controlled]
                if final_pose_error > math.radians(4.0):
                    raise RuntimeError(
                        f"Final restored-pose error was {final_pose_error:.3f} rad at "
                        f"{self._names[worst_dof]} (actual {actual_q[worst_dof]:.3f}, "
                        f"target {self._q_home[worst_dof]:.3f})"
                    )
                self._final_pose_error = final_pose_error
                final_box_position = as_numpy(
                    self._box.get_world_poses()[0]
                ).reshape(-1, 3)[0]
                final_box_velocity = as_numpy(
                    self._box.get_velocities()[0]
                ).reshape(-1, 3)[0]
                self._placed_box_speed = float(np.linalg.norm(final_box_velocity))
                self._post_release_slide_distance = float(
                    np.linalg.norm(
                        final_box_position[:2]
                        - self._box_first_rack_contact_position[:2]
                    )
                )
                self._placed_box_position = final_box_position
                if self._post_release_slide_distance < MINIMUM_RELEASE_SLIDE_DISTANCE:
                    raise RuntimeError(
                        f"Released box stopped in the middle of rack2 "
                        f"({self._post_release_slide_distance:.3f} m slide)"
                    )
                if self._placed_box_speed > 0.10:
                    raise RuntimeError(
                        f"Released box was still moving at demo completion "
                        f"({self._placed_box_speed:.3f} m/s)"
                    )
                self._phase = "DEMO_COMPLETE"
                self._set_status(
                    f"DEMO_COMPLETE - grab_box placed on rack2 level 3 at "
                    f"({self._placed_box_position[0]:.3f}, {self._placed_box_position[1]:.3f}, "
                    f"{self._placed_box_position[2]:.3f}), slid "
                    f"{self._post_release_slide_distance:.3f} m, max tilt "
                    f"{math.degrees(self._max_transport_tilt):.2f} deg"
                )
                return
        else:
            raise RuntimeError(f"Unhandled phase {self._phase}")

        self._pitch_velocity_command += np.clip(
            wheel_velocity_target - self._pitch_velocity_command,
            -MAX_WHEEL_ACCELERATION * command_dt,
            MAX_WHEEL_ACCELERATION * command_dt,
        )
        self._robot.set_dof_position_targets(
            q_command[self._controlled_indices], dof_indices=self._controlled_indices
        )
        self._robot.set_dof_velocity_targets(
            qd_command[self._arm_indices], dof_indices=self._arm_indices
        )
        self._robot.set_dof_velocity_targets(
            self._pitch_velocity_command, dof_indices=self._wheel_indices
        )
        # Once the box is released, presentation quality is more important
        # than allowing the imported compliant drives to ring freely beside
        # the rack.  Pin every non-wheel DOF to the commanded pose until the
        # base is completely clear.  Wheels and the root anchor still execute
        # the visible smooth retreat; only the forward torso/arm sway is
        # removed. ARM_RETURN deliberately falls outside this set and resumes
        # ordinary drive-based motion after clearance is established.
        if self._phase in {
            "PLACE_GRAVITY_DROP",
            "PLACE_BOX_SETTLE",
            "RETREAT_POSE_SETTLE",
            "BASE_RETREAT_HIGH",
        }:
            if self._phase in {"RETREAT_POSE_SETTLE", "BASE_RETREAT_HIGH"}:
                lock_pose = self._q_retreat_hold
            else:
                lock_pose = q_command
            self._robot.set_dof_positions(
                lock_pose[self._controlled_indices],
                dof_indices=self._controlled_indices,
            )
            self._robot.set_dof_velocities(
                np.zeros(len(self._controlled_indices), dtype=float),
                dof_indices=self._controlled_indices,
            )
