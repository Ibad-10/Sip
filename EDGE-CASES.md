# Edge Cases: Navigation, Sensing and Failure, per Idea

For each idea: what goes wrong in the real world, how the robot handles it, and how we show it in MuJoCo. Papers are from IEEE Xplore and arXiv abstracts (Chrome was not connected, so these come from public abstracts, not full texts).

## 1. Smoke Search (small tracked rover, fire scout)

### Navigation
| Edge case | Why it is real | Robot response | In simulation |
|---|---|---|---|
| Position drifts with no camera or LiDAR | Smoke kills vision; radar odometry degrades in long repetitive corridors (FIRE-LIVWO, arXiv 2609.05325) | Fuse track odometry, IMU and radar; show a position-uncertainty circle on the map; re-anchor on doorways it has seen before | Noisy odometry; the uncertainty ellipse is drawn on the overlay |
| Closed door | The rover cannot open doors | Marks "closed door, not searched" on the crew map. It never reports the building as clear | Some doors are closed boxes |
| Hole in a burnt floor, or a stair drop | Floors fail in fires | A downward-facing rangefinder detects the drop; it stops and marks the hazard | A gap in the floor; a down-facing ray |
| Stuck in debris, or debris falls on it | Collapse, falling ceiling | Tracks turning but the IMU says it is not moving: stuck. A sudden spike plus tilt: hit. It backs out and re-plans; if trapped for good, it becomes a beacon | A teammate drops a box during the take |
| Narrow gap it might fit through | Furniture, partial collapse | Checks its footprint against the gap on the occupancy grid; does not try if the margin is under X cm | Boxes placed to make a gap |
| Water from hoses, slippery floor | Crews spray | Wheel slip seen as odometry vs IMU mismatch; slows down | Low-friction floor patch |
| Battery running out | Finite energy | Keeps enough reserve to get home along the known-safe path; the mission timer is on screen | Battery counter |
| Radio drops | Buildings block radio (NIST breadcrumb relay work) | Keeps searching; uploads the map when back in range | Toggle a "link" flag mid-run (the wifi-off moment) |
| Robot blocks the escape route | A rover in a corridor obstructs people and crews | Parks against a wall when it detects people moving, and gives way | Not shown; listed in the summary |

### Perception
| Edge case | Why it is real | Robot response | In simulation |
|---|---|---|---|
| Thermal camera loses the person when the room is near body temperature | A person at about 37 °C barely shows up against a hot background | Falls back to radar micro-Doppler (breathing and motion; IEEE 10149735 reports about 98% validation accuracy in heavy smoke) and to audio | Thermal contrast reduced as the heat rises |
| Hot objects look like people (radiators, embers, reflections from glass or metal) | Thermal false positives | Needs a person-shaped blob plus radar motion or a voice before "confirmed"; otherwise "unconfirmed, second look" | Hot box decoys |
| Unconscious person makes no sound | Audio alone cannot locate unconscious victims (rubble sensor pilot study, PMC5877370) | Audio is a bonus, never required; thermal and radar carry it | One silent person and one calling |
| The robot's own motor noise drowns out voices | Ego-noise (IEEE hose-shaped rescue robot papers) | Stops the tracks to listen for 2 seconds at each doorway | Shown as a "listening" pause |
| Person hidden under a bed or blanket | Occlusion | Radar sees through light materials (UWB victim detection, IEEE 11083953); otherwise mark "room partly searched" | Person behind furniture |
| Firefighters in the building are people too | It will "find" the crew | Crew positions come from the command post; demo scope has no crew inside. Listed under Limits | Not in demo |

### Decision making
| Edge case | Robot response |
|---|---|
| Gemma is slow or gives a bad answer | Gemma only picks from options the planner has already checked as safe. The hard safety layer (heat limit, drops, battery) always overrides it. If Gemma takes more than about 2 seconds, it falls back to the nearest-frontier rule |
| Two people found: who first? | Report both at once. Rule: the person nearest rising heat is flagged first. A human decides rescue order |
| Fire spreads onto its planned path | The heat cost map is recomputed each tick; the path re-plans (the 25% adaptation moment) |
| It cannot be sure | Labels: "confirmed person", "possible person", "searched, no one found (confidence %)", "not searched". Never "clear" |

## 2. Sniffer (ground robot, gas leak hunter)
| Edge case | Why it is real | Robot response |
|---|---|---|
| A gas patch looks like the source | Plume patchiness traps robots (SniffySquad, arXiv 2411.06121) | Does not declare until upwind samples drop off and a second sensor agrees ("gas source declaration" problem, IEEE 1308025) |
| Gas pools in corners and behind obstacles | Obstacles create eddies, so a corner reads high (IEEE 8823147) | Uses the map: a reading in a dead-end corner with no pipe nearby gets low prior |
| Wind changes direction | Outdoor and ventilated sites | Re-estimates the source probability map (information-driven search, IEEE 9347683) and re-plans |
| Two leaks | Real plants | The probability map keeps more than one peak; keeps searching after the first find |
| Sensors are slow to react and recover | Metal-oxide gas sensors lag by seconds | Moves slowly near high readings; waits for readings to settle |
| The robot stirs the gas | Its own motion disturbs the plume | Stops before sampling |
| Gas heavier or lighter than air | LPG pools low, methane rises | Sensors at two heights |
| **The robot itself could ignite the gas** | A spark in an explosive atmosphere | A real build must be intrinsically safe (ATEX). This is the critical failure mode; state it, and the robot retreats above a set concentration |
| Confirming the source by sight | Invisible gas | Camera looks for a hissing valve plus acoustic leak detection (an industrial method in real use) |

Verdict: rich edge cases, but the ignition risk makes the real product hard, and judges may ask about it.

## 3. Lost & Found (home robot, object memory)
| Edge case | Why it is real | Robot response |
|---|---|---|
| Item moved since last seen | Stale memory (DynaMem; DREAM, arXiv 2606.00576) | Confidence decays with time; on arrival, compares the live view with memory; re-searches likely places |
| Item hidden in a drawer or under a cushion | Occlusion | "Not visible; may be inside the drawer", never "not here" |
| Two similar items (two pairs of glasses) | Ambiguity | Asks "the black ones or the reading glasses?" |
| The person is wearing it | A classic | Checks the person first: "they're on your head" |
| A confident wrong answer upsets someone with dementia | Distress | Always gives the time it last saw the item, and never invents a location |
| Re-checking costs time | Re-perception only helps when observations are reliable (arXiv 2607.23797) | Searches in order of likelihood and stops after N places |
| Home clutter: rugs, cables, pets, dark at night | Real homes | Low-speed mode; infrared light at night |
| Privacy: cameras in a home | Real concern | Everything stays on the device; it stores labels, not images |

Verdict: safe, and the edge cases are good stories, but low sci-fi factor.

## 4. Rubble Dog (quadruped, acoustic search in a collapse)
| Edge case | Robot response |
|---|---|
| Rubble shifts under its weight; the robot could cause a secondary collapse | Light robot; probe footholds; still a real risk |
| Slipping and falling | Needs a trained walking policy (MuJoCo Playground), which takes hours |
| Echoes in voids give the wrong direction | Take readings from several positions and triangulate |
| Motor noise drowns voices | Stop and listen (hose-shaped robot ego-noise work) |
| Unconscious victims are silent | Add radar breathing detection |
| Dust and no radio through rubble | Autonomous; return to transmit |

Verdict: strong story, too many hard edge cases for 22 hours.

## 5. Pill Helper (fixed arm, sorting medication)
| Edge case | Robot response |
|---|---|
| Pills that look alike | Read the packaging, not the pill; check deterministically against the prescription |
| A dropped pill rolls away | Detect it on camera and re-grasp, which is the hard part in simulation |
| A wrong pill is harmful | The builder guide requires clinical actions to be human-approved; a pharmacist checks before sealing |
| The user's hand is in the workspace | Stop the arm when a hand is seen |

Verdict: navigation is not the issue; grasping and clinical approval are.

## 6. Night Watch (home rover, fall finder)
| Edge case | Robot response |
|---|---|
| Person lying on a bed vs on the floor | Uses floor height from depth |
| A pet or a pile of clothes lying down | Needs a pose check plus a voice check |
| Person says "I'm fine" but is not | Asks follow-up questions; tells the family anyway |
| Robot cannot lift anyone | It only alerts and talks; it brings the phone |
| False-alarm fatigue | Fixed sensors already struggle with this, and the market is crowded |

## IEEE Xplore check (Chrome, 7 Nov 2026)
- Searching "firefighting robot autonomous navigation smoke" (2022–2026) gave 8 conference papers, nearly all low-cost IoT robots with flame sensors and water pumps (for example "IoT-Based Firefighting Robot with Gas Detection", 2026; "Edge-Optimized Firefighting Robot: Autonomous Navigation via PID Control and Fire Detection with FOMO Vision", 2025). None does autonomous search under smoke-degraded sensing. Meaning: judges have seen many "firefighting robot" student projects. Our pitch must say "search scout", not "firefighting robot", and lead with sensor degradation and uncertainty.
- Kulhandjian et al., "AI-based Human Detection and Localization in Heavy Smoke using Radar and IR Camera", IEEE RadarConf 2023 (DOI 10.1109/RadarConf2351548.2023.10149735): micro-Doppler radar plus an infrared camera with a DCNN; detects a human in heavy smoke with about 98% average validation accuracy. This is the citation for our radar-plus-thermal fusion.
- The "thermal fails near body temperature" edge case: an IEEE search found no paper on it. It rests on physics reasoning (low thermal contrast), so present it as our own design reasoning, not as a cited result.
- Full texts need the user's Brunel institutional sign-in, which the user must do themselves.

## Takeaway
Smoke Search still wins, but the edge cases changed the design:
1. The hardest real problem is thermal failing when the room is near body temperature. This is why radar plus audio fusion is needed, not optional. Put it in the pitch.
2. Gemma must never be the safety layer. Deterministic rules (heat, drops, battery) always override it.
3. The map never says "clear", only labelled confidence.
