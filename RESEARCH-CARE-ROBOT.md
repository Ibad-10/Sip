# Home Care Robot: Research Pack

Compiled 7 November 2026 for AgentVerse 2026 (robotics track). Idea: a mobile robot with an arm that hears "can I have a glass of water?", fetches it and brings the straw safely to the person's mouth, helps with eating (feeding, kept in at the user's request), reminds about medicines and fetches the pre-filled organiser, and talks like a helpful carer. Users: older people and disabled people who live at home or in care homes.

Vendor figures are self-reported. Scopus was not searched (it needs an institutional login); IEEE Xplore was searched through Chrome (abstracts only).

## 1. Need (evidence)
| Figure | Source |
|---|---|
| 2 million people aged 65+ in England have some unmet need for social care; 1.1 million struggle to dress, 640,000 to bathe (2024) | Age UK |
| About 4.5 million people aged 50+ need help with essential everyday tasks (2025) | Age UK State of Health and Care 2025 |
| 1 in 4 people in the UK are disabled (FRS 2024–25); 16.7 million per Activity Alliance | DWP Family Resources Survey; Activity Alliance |
| Medication-related harm affects about 1 in 3 older adults after hospital discharge, costing the NHS about £400m a year; non-adherence was the sole cause in 11% of cases | Pharmaceutical Journal (UK prospective study) |
| About 10% of hospital admissions of older people are attributed to non-adherence (estimate) | ILC-UK |
| Unintentional non-adherence (forgetting) is more common in older people | Pharmacy Magazine |
| Poor adherence and dehydration are well-documented problems for older adults living alone | Georgia Tech medication and water delivery paper |
| Elder care assistive robot market about $3.4bn (2025), projected $9.8bn by 2033 (estimates vary) | Grand View Research |
| YC's Fall 2026 Request for Startups includes "AI for the aging population" (voice, robotics, eldercare monitoring) | YC RFS summary |

## 2. Competitors
| Product | What it does | Gap for us |
|---|---|---|
| ElliQ (Intuition Robotics) | Tabletop companion: conversation, reminders | No arm, does not move |
| Hyodol (South Korea) | Talking doll: conversation, medication reminders, reports to carers (12,000+ subsidised units, secondary source) | No arm, does not move |
| Rhem Labs (YC F26, $500k pre-seed) | Home care device: calls, groceries, reminders; reportedly mmWave radar for falls | No manipulation reported |
| Kinova JACO | Wheelchair-mounted arm with a drinking mode; about $35,000, often not covered by insurance | Fixed to a wheelchair; does not fetch |
| Obi | Feeding device (Obi3 real-world study, JMIR preprint 2025) | Fixed at the table; feeding only |
| Labrador Systems Retriever | Carries up to 25 lb around the home, voice control; $5,000 "Caddie" version. As of early 2026: about $5.45M funding, 9 staff, no public deployment metrics | No arm, cannot hand things over |
| Hello Robot Stretch | Mobile base plus telescoping arm. Nearly $3M NIH funding (Oct 2026) for older adults with cognitive impairment, with the University of Illinois. A UNH lab Stretch at a family home prompts a man with dementia to eat, drink and take medicines | Research platform, not a product. Also our simulation robot |
| Andromeda Robotics (Abi) | Senior-care robot; reported $17M raise, March 2026 (secondary source) | Companion focus |
| Twolabs (YC Spring 2026) | Humanoids for caregiving (thin sources) | Early |
| Weave Robotics (YC) Isaac 0/1 | Home robot for laundry and tidying; Isaac 0 still uses remote operators | Not care-focused |
| 1X NEO, Figure, LG CLOiD | General home humanoids; no confirmed release or slow demos | Not care-specific |
| Diligent Moxi, Aethon TUG | Hospital delivery robots | Hospital logistics, not personal care |
| Evondos (Denmark) | Medication dispensing robot. Case study: takes up space, may suit younger older adults, can confuse people with dementia | Dispenser only |

**Gap:** nothing commercial combines moving around the home, picking things up, handing them safely to the mouth, medicine routines and conversation, fully on the device.

## 3. Research to build on
| Paper | Why it matters |
|---|---|
| Assistive Gym (Erickson et al., ICRA 2020, 84 IEEE citations) | Simulation framework for assistive tasks: feeding, drinking, dressing, bathing, itch scratching, arm manipulation. Uses PyBullet; we borrow task definitions and human modelling |
| Assistive VR Gym (RO-MAN 2020) | Real people in VR to improve virtual assistive robots |
| Robotic Arm System for Assisting Persons with Physical Disabilities to Drink Water (IEEE ICIIBMS 2024) | Direct prior art for drinking |
| JACO drinking research (RESNA 2016) | Most JACO users drink through a straw; an automatic drinking motion was designed for drinking from a glass |
| Active robot-assisted feeding with a general-purpose mobile manipulator (PR2, arXiv 1904.03568) | Mobile manipulator feeding; monitors and prevents anomalous executions |
| Lessons learned from a robot-assisted feeding system for out-of-lab use (Nanavati et al., HRI 2025) | Users overcome many off-nominal events when given control and customisation |
| Breakaway utensil attachment (arXiv, Feb 2025) | Mechanical fail-safe that breaks at 65 N, following ISO/TS 15066 |
| Intelligent feeding robot RCT (2026, 40 older adults) | Robot vs human feeding: intake, success rate, duration, satisfaction, safety |
| Hands-free interface for robot-assisted self-feeding (IEEE ICORR 2022) | Interface ideas for users who cannot use their hands |
| Cornell "Feel the Bite"; FEAST (arXiv 2506.14968, 2025) | Inside-mouth bite transfer; mouth openings under 2 cm and spasms are real constraints; FEAST five-day in-home study |
| Medication adherence with a companion robot and wearable (Smart Health 2023) | Reminders, dose confirmation, logging; apps and pill boxes are not age-friendly |
| Groningen medication robot study (2016) | Older adults found advanced interface settings difficult; keep it simple |
| Georgia Tech medication and water delivery robot | Pills plus water delivery to improve adherence and hydration |
| Clinical trial NCT02807506 | Robot carries an 8 oz water bottle on a clinician-set schedule |
| Care-O-bot 3 (IEEE 2009, 85 citations) | Classic home assistant reference |

## 4. Open-source code and tools
| Tool | Use | Notes |
|---|---|---|
| **stretch_mujoco** (Hello Robot) | Stretch 3 in MuJoCo: position control of joints, velocity control of the base, colour and depth cameras, 2D lidar, Python and ROS 2 | Pins mujoco 3.2.6; runs on Windows; the experimental 3.3.0 model is unstable. Has `robocasa_gen` to generate kitchens |
| **MuJoCo Menagerie** | Stretch 2 and 3 models | Backup source of the model |
| **RoboCasa / RoboCasa365** | MuJoCo kitchens: 2,500 environments, 3,200+ objects, 365 tasks | About 5 GB of assets; built on robosuite |
| **personalrobotics/ada_feeding**, feeding_web_interface | Real feeding stack (ROS 2, Kinova) and a React interface for sip-and-puff or chin-joystick users | Reference for flow and interface, not run directly |
| **Assistive Gym** (GitHub) | Human models and task definitions | PyBullet, so reference only |
| **Gemma 4 E2B / E4B** | Local language model with audio input (E2B and E4B only, clips up to 30 s) via llama.cpp `llama-server` | One benchmark found E2B garbled English speech while E4B was accurate, so test both on the A2000 |
| **MediaPipe Face Landmarker** | Face mesh to find the mouth and whether it is open | No built-in "mouth open" output; compute it from inner-lip landmarks |
| **Piper TTS** | Offline speech output, MIT licence, runs on CPU | Voices are ONNX models |
| **NVIDIA SimReady assets** (Hugging Face) | Props for the scene | See section 5 |

## 5. NVIDIA SimReady assets (Hugging Face, CC-BY-4.0, 2,677 assets, 490 GB, USD format)
Relevant folders found in the first 1,000 directory names:
- **Care and medical:** `bed_hospital_standard_a01/a02`, `bedding_a01`, `chair_patient_mobile_a01`, `chair_hospital_a01/b01-b03`, `cabinet_for_drugs_a01-a06`, `cart_medical_utility_a01-a03`, `bandage_medical_*`, `facemask_medical_a01`
- **Drinking and eating:** `dishware_glass_8oz_a01`, `dishware_glass_tumbler_a01`, `dishware_plastic_cup_a01`, `dishware_mug_a01/a02/b01/c01`, `dishware_cup_tea_a01`, `dishware_plastic_spoon_a01`, `dishware_plastic_fork_a01`, `dishware_bowl_a01`, `dishware_plastic_smallPlate_a01`, `dishware_plastic_tray_a01`, `consumable_cup_paper_a01`, `container_bottle_glass_a01`
- **Home:** `facilitySystem_kitchenette_module_a01`, `accessory_eyeglasses_a01`, `fauset_bathroom_a01`
- **Robot support:** `equipment_robot_charger_a01`, `equipment_robot_trolley_a01-c01`
- The full list is longer (the API returned the first 1,000 of 2,677); saved at the scratchpad file `simready_dirs.txt`.

**Format catch:** SimReady is USD, built for Isaac Sim. MuJoCo's USD loading is experimental, and the converter `mujoco-usd-converter` only goes the other way (MJCF to USD). Plan: download only the few props we need, extract meshes with Python `pxr` (usd-core) to OBJ, and write simple MJCF with collision primitives (cylinders for cups, boxes for furniture). Assets are about 180 MB each on average, so do not bulk-download. RoboCasa already provides native MuJoCo kitchens and objects, so use SimReady only for the care-specific hero props (hospital bed, patient chair, drug cabinet, 8 oz glass).

## 5b. SimReady Central (Hugging Face Space, seen by the user on 7 Nov 2026)
- 3,732 assets: NVIDIA SimReady 2,537, ISVs 1,102, OEMs 49, DSX 45. Search plus filters for rigid bodies, mass, licence and dimensions.
- Kitchen: base and wall cabinets, kitchenette module, counters, kitchen island, range hood, gas and electric ranges, stove, microwave, French-door fridge, dish rack, faucet, cutting board, knives, spatula, pans, mixer.
- House: interior door, staircase, washing machine, office chair, cutlery set, storage box, robot trolley, pitcher, mugs.
- Sizes vary hugely: the kitchenette is 1.33 GB, a counter 378 MB, a kitchen island 617 MB, while the faucet is 2.7 MB and a cutting board 7 MB. Pick small assets or decimate meshes.
- Licences vary by maker; use the licence filter and keep a credits list (CC-BY-4.0 requires attribution).
- Still USD, so the same conversion plan applies. Use simple primitives for collision, and keep the detailed mesh visual only so MuJoCo stays fast.

## 5c. Off-the-shelf parts to reduce risk
| Risk | Off-the-shelf fix | Fallback |
|---|---|---|
| Finding the cup, pill organiser and spoon in camera images | Open-vocabulary detectors run locally: OWLv2, Grounding DINO, YOLO-World (type the object name, no training). Gemma 4 vision for "which compartment is today?" | MuJoCo's segmentation render gives exact object masks; add noise and say so |
| Object position for grasping | Depth camera on Stretch plus the detector's box gives a 3D point | Known object pose from the simulator plus noise |
| Grasping | Stretch's arm is nearly Cartesian (vertical lift, telescoping arm, wrist yaw), so reaching a point is simple geometry, not a hard inverse-kinematics problem. Use scripted grasp primitives (approach, close, lift) driven by the perceived pose. Grasp networks (Contact-GraspNet and similar) are not needed | Weld constraint on contact (stated in Limits) |
| Navigation | stretch_mujoco has a 2D lidar; build a simple occupancy grid and A* in Python | Fixed waypoints, with obstacles still handled by stopping and re-routing |
| Finding the mouth | MediaPipe Face Landmarker on the robot's camera image | Head pose from the simulated human plus noise, if MediaPipe fails on a rendered face |
| A believable person in the scene | MuJoCo humanoid models; Assistive Gym's human models as a reference. SMPL-X body models are realistic but licensed for non-commercial use only | A simple seated humanoid with a textured head |
| Speech in | Gemma 4 E4B audio input locally | faster-whisper (local speech-to-text) feeding text to Gemma |
| Speech out | Piper TTS (offline, MIT) | Text on screen |
| Training data | Not needed for the core: everything above is pre-trained or scripted. RoboCasa provides kitchen tasks and demonstration datasets if we want a learned policy later (check the current release) | None needed |

## 6. Scope agreed so far
- Core: water fetch and straw hand-over with head tracking; medicine reminder plus fetching the pharmacist-filled organiser and checking the compartment; local voice.
- Feeding: kept at the user's request. Risk: spoon and food physics, and the mouth area is safety-critical. Plan it as the last module, using rigid food pieces and a spoon pose target rather than real food physics.
- Not a nurse: no medical advice, no dosing decisions. Logs taken or declined and tells a carer.
- Robot: Stretch 3 (mobile base plus telescoping arm), not a humanoid.

## 7. Risks and open questions
- Stretch's telescoping arm reaching a seated person's mouth: check the reach in simulation in hour one.
- MediaPipe on a simulated human face: needs a realistic face texture or mesh, or the mouth target falls back to a known head pose plus noise. Test early.
- Gemma 4 E2B audio quality: may need E4B, which uses more VRAM.
- Stretch MuJoCo pins mujoco 3.2.6; check that it installs alongside other tools.
- Privacy: everything on the device; store logs, not video.
