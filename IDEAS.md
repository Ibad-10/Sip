# Robotics Track Ideas: Judge Roast and Ranking

Scores are my estimate of what a judge would give a well-executed version, using the track weights: Perception 25, Adaptation 25, Autonomy 15, Relevance 20, Ambition 15. Feasibility is for two people, about 22 hours, an Iris Xe laptop and an RTX A2000 8 GB laptop. Need statistics marked "verify" still need a source before they go in the pitch.

## What the brief actually rewards (re-read)
- "Every decision traces to what it perceived, and it corrects mid-run." The agent must make a visible mistake or meet surprise and fix it. A perfect run with no correction scores lower.
- "The scene changes mid-sequence and the agent visibly re-plans." The re-plan must be visible: an old path, a new path and a stated reason, on screen.
- "Runs within a stated edge-device budget and behaves sensibly when unsure." State the budget as a number (VRAM, watts, Hz, latency), and show an "unsure" behaviour on screen.
- "A well-evidenced need, with the most critical failure mode handled on screen." Cite real statistics. Name the worst failure, then show it being handled in the video.
- "If it feels like science fiction, you're heading the right way." Ambition counts for only 15%, but the tagline says the judges want to be impressed.
- Video: 1 minute or less, one continuous take, captured from the stream, perception shown next to the actions, one unscripted change. This is the single most important asset, so design the idea around it.
- Technical summary: where each model runs, on what hardware, at what latency. Measure this; don't guess.
- Pitch: "what it can and cannot do". Saying the limits up front earns trust.

## Decision log
- 7 Nov 2026: Line Guardian (Counter + brain) rejected by the user: too many assumptions, not realistic, not scalable.

## Round 10: roast of a "Counter for any line, warehouse or shop"

### Finding: a general-purpose counter is already crowded
| Company | Form | What it does |
|---|---|---|
| Gather AI | Drones, plus **MHE Vision**: cameras on forklifts, pallet jacks and tuggers that capture images passively during normal work | Counts pallets and cases, reads labels; pairs with most vehicle and RTLS systems. Claims 99.9% accuracy (vendor) |
| doks. innovation inventAIRy XL | Drone tethered to a ground AGV | Up to 5 hours flight; claims 2,000+ rack locations per mission (vendor); customers include Fiege, Knowles, Rigterink, Kite Packaging |
| Brain Corp Sense Suite | Scanning towers on floor-scrubber robots (first at Sam's Club) | Shelf images and RFID become action lists for staff |
| Simbe Tally | Shelf-scanning AMR | Retail; 5,400 items an hour |
| Corvus, Verity | Warehouse drones | Inventory scans, usually after hours |
| Würth iBin, iSCALE, Bossard SmartBin | Fixed sensors per bin | Small C-parts |

So "camera on vehicles you already have" (our earlier v2) is Gather AI's MHE Vision. "Tall racks" is solved by drones (inventAIRy XL, Gather AI). Do not pitch a general counter.

### Where we are still strong
- **At the line, during production:** low line-side racks and flow racks, people and tuggers moving, mixed containers (KLTs, Euro pallets, rack slots). The products above target warehouse racking or retail shelves.
- **Closing the loop:** count, then time to empty, then dispatch (VDA 5050 or a store request), then check the label at delivery. From what we found, the products above focus on visibility and accuracy; we did not see them dispatching replenishment robots. Phrase it that way, not as a certainty.
- **Position:** hardware-agnostic brain. Our Counter is the eyes where there are none; the brain can also take counts from Gather AI-style vehicle cameras or smart bins. For the hackathon, build the Counter, because the track requires an embodied agent.
- **One-liner:** "Inventory robots count the warehouse. Nobody is watching the line."

### Form factor: what the Counter is
| Option | Verdict | Why |
|---|---|---|
| **Small ground AMR with a fixed camera mast (about 2 m, cameras low, mid and high, plus one depth camera)** | **Yes** | Line-side racks are low (assumed about 2 m; verify for the target plant); same pattern as Simbe and Brain Corp towers; stable, cheap, safe near people |
| Telescopic mast | No | Moving parts, tips over, slow; not needed for low racks |
| Drone | No (for the line) | People and tuggers around during production; battery life; tall racks are already served by inventAIRy XL and Gather AI |
| Humanoid | No | Slow, expensive, no benefit for looking |
| Tall pallet racking in the warehouse | Out of scope | Already served; take their counts through the brain if a customer has them |

### Judge roast of the Counter, with answers
| Question | Answer |
|---|---|
| "Gather AI already does this" | They count the warehouse; we watch the line and trigger delivery |
| "Why not fixed cameras at the line?" | Per-bin or per-rack hardware, blind spots, re-fitting at every model change. The brain accepts fixed cameras too |
| "A patrol is not continuous" | Route by time to empty: fast-draining bins get visited more often |
| "You can't count jumbled parts" | Estimate fill level by depth, not piece count (IFAC 2023: about 7% average deviation with a monocular camera); report confidence |
| "It's in the way of tuggers and forklifts" | Standard AMR safety scanners; patrols on the pedestrian walkway; low speed |
| "Who acts on its trigger?" | The existing fleet through VDA 5050, or the tugger driver's tablet |
| "Label is turned away or dirty" | Second look from another angle; if still unread, flag a person |
| "New model, new bins" | Configuration (rack map, container types, thresholds), not new hardware |
| "Wi-Fi drops" | Keeps patrolling, queues triggers locally |

### Hospitals and patient data
- Hospital version is a later market, not the hackathon build.
- If built: privacy by design. Process images on the device, store counts not images, blur faces, and take demand as "drug code, ward, due time" with no patient name or bed. UK GDPR Article 28 requires a processor agreement; video of patients may be special-category data; NHS suppliers that process NHS data fall under the Data Security and Protection Toolkit.

## Round 9: Line Guardian deep research and use-case check

### Prior art
| Source | What it shows | Meaning for us |
|---|---|---|
| Int. J. Production Research (2018), material supply control for assembly lines | Detects supply needs, schedules and dispatches mobile units; notes that the lack of real-time feedback on shop-floor inventory levels limits material flow | The gap, stated by researchers |
| TU Hamburg, data-driven task assignment for robots feeding a mixed-model assembly line | Uses real-time robot and station state; measures station idle time from shortages | Prior art for dispatch; we add the eyes |
| Patents US 10,885,495 / 11,587,033 / 12,073,357 | Vehicle acts on a replenishment signal, including reading a Kanban card optically | Card reading exists; estimating fill level from a mobile robot does not appear |
| IFAC 2023, monocular fill-level detection for eKanban | Low-cost RGB camera plus CNN, about 7% average deviation | A fill-level-by-camera accuracy figure we can cite |
| KIT IFL (Logistics Journal 2021) | Smart camera on Raspberry Pi or Jetson Nano class hardware watches full and empty KLTs in a buffer zone | Fixed camera, edge hardware: supports our edge-budget claim |
| Chery + ForwardX (vendor) | 100+ vision AMRs integrated with the plant's logistics execution system | Large AMR fleets are real in car plants |
| YC: Horizontal (F26) | AI agents that coordinate the people and systems around factory disruptions | Software only, no robots |
| YC: Allus AI (F25), Corvus (S18), Truffle (S26) | Factory vision models; warehouse inventory drones; vision and voice inventory counting | Adjacent, none at the line |
| Hackathons: Droneventory, BoxBeacon (Devpost), Lucen Robotics and an Isaac Sim supermarket restocking project (lablab) | Shelf scanning and restocking, almost all retail | Nobody has done factory line-side |

### Edge questions that shape the use case
| Question | Answer | Evidence |
|---|---|---|
| Pallets up to about 1,000 kg: can our robot lift that? | We do not lift them. Heavy moves stay with machines built for it: iw.hub carries up to 1,000 kg on a dolly or pallet (some pallet-dock setups up to 1,300 kg); MiR1000 and Hikrobot MR-F4-1000 carry 1,000 kg; ABB F712 autonomous forklift up to 2,000 kg; human forklifts and VNA trucks stay | idealworks, MiR, Hikrobot, ABB |
| KLT weight | Small load carriers are light. One maker rates its R-KLT series at 20 kg per container; guides quote up to about 50 kg. VDA 4500 defines dimensions, not a single payload | Schoeller Allibert, KLT guides |
| Picking from big racks and automatic stores | We do not pick. The automatic store (AutoStore, or BOFP and SCW at Hams Hall) does its own picking. A WMS requests bins through its API (AutoStore's Task or Bin interface); external robots connect at the port, never inside the grid | AutoStore interface docs |
| How does it talk to existing robots? | VDA 5050, the standard interface between AGVs/AMRs and a master control. BMW chairs the working group and makes VDA 5050 mandatory in its plants. Version 2.1 (May 2024); 3.0 reported for March 2026 (Bluebotics, unverified against the VDA document) | VDA, idealworks, Bluebotics |
| Why a moving Counter and not fixed smart bins? | Fixed sensors need hardware in every bin, are built for small C-parts and miss pallets, racks and floor stock, and must be re-fitted at every layout change. One patrol covers everything. No per-bin price is published, so make no cost claim | Würth, Bossard |
| A patrol cannot watch every bin at once | Correct. The Counter plans its route by time to empty: fast-draining bins get visited more often (active perception). v2: every AMR that already drives past reports what its camera sees, so no extra robot is needed | Our design |

### Refined use case: Line Guardian is the eyes and the dispatcher, not a new forklift
1. **Counter robot:** patrols the line, estimates fill level of KLTs, Euro-pallet stacks and rack slots, with confidence. It flags phantom stock where the system disagrees with the shelf.
2. **Brain (on-premises):** predicts time to empty, ranks by line-stoppage risk, applies the asymmetric-cost rule, and checks part numbers.
3. **Dispatch through existing interfaces:** a store request (AutoStore-style API) to release the bin or pallet; a VDA 5050 order to the existing AMR or autonomous forklift to carry it. Our demo Runner is one generic VDA 5050-style AMR that docks under a dolly.
4. **Verify at the line:** the Counter checks the label after delivery; a wrong part is flagged before use.

### In the MuJoCo demo
- Line with racks of KLTs and two pallet places; a mock automatic store port where a dolly appears a set time after it is requested.
- Counter: small wheeled robot with a camera mast. Runner: AMR that docks under a dolly (weld constraint, stated).
- Orders are sent as VDA 5050-shaped JSON (from the public schema on GitHub) to make the integration story concrete. Real MQTT is optional.

## Round 8: all logistics ideas in one table
Scores use the track weights (P25, A25, O15, R20, Q15). Evidence marked "vendor" is self-reported and unverified.

| Rank | Idea | What it does | Who it helps | Evidence | Already exists | Our gap | Score | Build |
|---|---|---|---|---|---|---|---|---|
| 1 | Fleet Watch | In a fire, the plant's own AMRs clear exits and search aisles for people | Workers, fire crews | Atherstone 2007: 4 firefighters died; UK Li-ion fires 1,760 in 2025, up 147% (QBE) | AMR fleets (idealworks); fire robots are separate machines | Re-using a fleet already inside, with maps | 94 | Medium |
| 2 | Line Guardian (Counter + Runner) | The Counter watches line-side stock by camera; the Runner fetches and delivers before the line starves | Line leads, logistics planners | Automotive downtime about $2.3M an hour (Siemens 2024) | Smart bins (Würth iBin, iSCALE, Bossard); inventory drones (Gather AI, Corvus, Verity); iw.hub delivery | Closes the see-decide-deliver loop on mixed containers during production | 92 | Medium |
| 3 | Battery Sentinel | A patrol robot with a thermal camera finds a warming battery pack in an EV-parts store, alerts, and routes the fleet away | Plant safety, insurers | Li-ion fires up 147% since 2022; academic early warning only about 10–22 s before runaway while charging (Qingdao) | Fixed thermal cameras (MultiSensor AI); predictive detectors (EPTTAS, vendor); patrol robots only in marketing (iFactory) | Mobile thermal coverage of stock that moves, plus quarantine routing | 89 | Easy-medium |
| 4 | Dock Doctor | At goods-in, checks each pallet's label, damage, wrap and position before a forklift picks it | Goods-in teams | Forklifts struggle with pallets that are off position, damaged or blocked by stretch wrap (Big Joe); manual check of 5 s per pallet in the user's Hams Hall model | Vision inspection tools | Robot that inspects and re-routes bad pallets to a quarantine lane | 85 | Medium |
| 5 | Smart taxi | Tugger train that sees pedestrians and forklifts and re-routes instead of stopping dead | Tugger drivers, pedestrians | 20 of 135 GB workplace deaths (2022/23) were people struck by moving vehicles | Many AGVs and AMRs | Perception-driven re-routing in mixed traffic; crowded space | 84 | Medium |
| 6 | Rack Doctor | Patrol robot spots bent uprights, overhanging pallets and overloaded beams, and isolates the bay | Warehouse managers | Vendor claims: a major racking collapse every week in the UK; £1.5bn a year (unsourced). Inspections are typically annual | Impact sensors on uprights, inspection apps | Continuous visual inspection by the fleet already driving past | 83 | Easy |
| 7 | Blind-corner guardian | Predicts people at blind intersections and warns or slows forklifts | Pedestrians | Struck-by-vehicle deaths above | Forklift proximity systems | Prediction from camera instead of tags | 83 | Medium |
| 8 | Humanoid line loader | Humanoid loads parts and adapts when one is misplaced | Line operators | Figure 02 at BMW: 90,000+ parts in 11 months | Figure, Agility | Little gap; Figure has done it for real | 82 | Hard |
| 9 | Yard hostler | Autonomous yard truck moves trailers and connects air lines | Yard operations | Vendor: drivers get in and out about 6 billion times a year to connect air brakes | Outrider | Little gap; heavy vehicle simulation | 80 | Hard |
| 10 | Returns sorter | Inspects returned items (unoriented, damaged) and sorts them | E-commerce warehouses | Returns are a known hard case for robots | Vision inspection | Arm picking of random items is hard | 80 | Hard |
| 11 | Trailer unloader (mixed or collapsed loads) | Unloads boxes from trailers | Dock teams | Loading dock called the most overlooked bottleneck | Boston Dynamics Stretch, Pickle, Ambi, Mujin, Berkshire Grey Scoop | Crowded | 78 | Hard |
| 12 | Empties manager | Collects empty KLTs and bogies and finds lost containers | Logistics planners | No source yet | Container tracking tags | Unclear need | 76 | Easy |

## Round 7: two robots, a Counter and a Runner (Line Guardian v2)

### What already exists
| Type | Examples | What it does | What it does not do |
|---|---|---|---|
| Fixed smart bins | Würth iBin (camera inside a small-parts bin; 6 pilot customers, 2,000 bins), Würth iSCALE (scale under the KLT), Bossard SmartBin, eTurns SensorBins, Clever Box | Measures fill level per bin and reorders through the ERP | Needs hardware in every bin; built for small C-parts; does not cover Euro pallets, large parts or floor stock; no price published; does not deliver |
| Inventory drones | Gather AI (claims 99.9% accuracy, $40M Series B Feb 2026), Corvus One, Verity (DSV case: audit cost down 50%) | Scans warehouse racking, usually after hours | Not at the line, not during production, does not trigger delivery. Claims are self-reported |
| Mixed robot teams for counting | ICUAS 2023: micro-drone plus ground robot for counting in long, high-shelved warehouses; HetSwarm (arXiv 2502.06722) | Counting faster as a team | Monitoring only, no replenishment |
| Line-side delivery AMRs | idealworks iw.hub at BMW | Moves dollies and trolleys on order | Follows orders from the system; does not see the shelf |
| Orchestration of different robots | Robotics 24/7: a multi-agent orchestration layer is "just now being developed" | | Open gap |
| Inventory accuracy | System records drift from the floor (missed postings, returns, label mix-ups). The DeHoratius & Raman (2004) study is the academic source to cite (figures not yet pulled); a 1–2% phantom-stock figure is a vendor claim | | |

**Gap no one closes:** see the shelf → decide → fetch → deliver, during production, across mixed containers (KLTs, Euro pallets, racks), with two kinds of robot coordinating.

### Should there be two robots? Yes, but no arm at first
- **Counter (scout):** small, fast, camera on a mast. Patrols the long lines continuously, counts or estimates fill level in KLTs, pallets and rack slots, and works out time to empty from consumption between passes.
- **Runner (delivery):** an AMR with a lifting deck that docks under a pre-staged trolley or dolly at the supermarket (as iw.hub does). It drives to the line and takes the empty back. No arm means no grasping risk.
- **Picker arm at the supermarket:** stretch goal only. It fills the trolley that the Runner collects.
- Why two: lines are long and parts are spread out, so one heavy robot cannot watch everything and deliver at once. A cheap watcher plus a strong carrier also matches "Coordinated fleets" in the brief.
- Why not two: twice the integration work. Mitigation: share one map, one planner and one message bus.

### What we add on top of existing work
1. Vision counting on mixed containers with a confidence score, cross-checked against the system of record. Mismatches are flagged as phantom stock.
2. Time to empty predicted from what it sees between passes, not from the bill of materials.
3. Priority when two lines run short at once, weighted by downtime cost (about $2.3M an hour in automotive, Siemens 2024).
4. Asymmetric cost rule: a false trigger costs one trip, a missed shortage stops the line. If unsure and time to empty is short, trigger anyway; if unsure and time is long, the Counter takes a second look first.
5. Label check before delivery to prevent the wrong part (the critical failure).
6. The Runner re-plans live: the Counter finds a more urgent shortage, so the Runner swaps jobs mid-route; an aisle is blocked by a forklift; people step in.
7. Gemma explains each decision on screen ("line 3, bin 12: about 4 min left, aisle B blocked, going via C").

### Judging fit
| Criterion | How it is shown |
|---|---|
| Perception-driven action (25) | Every delivery traces to a count the Counter saw; the count corrects the system's number on screen |
| Adaptation (25) | A teammate empties a bin or blocks an aisle; the Runner swaps jobs mid-route |
| Autonomy (15) | Both robots run on-device; they keep working if plant Wi-Fi drops; stated budget |
| Relevance (20) | Line stoppage cost; wrong-part failure shown and handled |
| Ambition (15) | A two-robot team closing the see-decide-deliver loop |
Estimated score 92 (P24 A24 O12 R18 Q14). Build: medium.

### Edge cases for the two-robot version
- Parts hidden under other parts: estimate fill by depth (height of contents), not by counting pieces.
- Stacked KLTs where only the top one is visible: mark lower ones "unknown", never "full".
- An operator takes parts while the Runner is on its way: the count updates and the job continues.
- The line-side slot is still full when the Runner arrives: it waits or parks the trolley at a buffer.
- Counter and Runner meet in a narrow aisle: right-of-way rule (loaded Runner first) to avoid deadlock.
- The robots lose contact with each other: each carries on with its last plan; the Counter queues triggers.
- A label is unreadable or the part looks wrong: no delivery; flagged to the line lead.
- Battery: charging is scheduled so that one robot is never down during peak consumption.

## Round 6: logistics picking and delivery

### Evidence
- Unplanned downtime in automotive costs about $2.3 million an hour, double 2019 (Siemens, True Cost of Downtime 2024). Per-minute figures quoted range from $22,000 to $36,000; use the hourly figure.
- Automotive is just-in-time, so a missing or wrong part at the line can stop it. No source found yet that quantifies stoppages caused by missing parts specifically.
- Plants are moving to kits delivered to the workstation. Fixed-path AGVs cannot re-route; AMRs can (MMH).
- Kitting needs dexterity. Shiny, black and transparent parts, occlusion by packaging and moving obstacles are open problems (EU i4ms project).
- Vendor piece-picking claims are 98–99.7% success (Brightpick, self-reported), so pure picking is crowded.
- Truck unloading is already sold (Boston Dynamics Stretch, Berkshire Grey Scoop, Mujin TruckBot, Pickle, Anyware Pixmo).

### Ideas
| Rank | Idea | What it does | Gap | P | A | O | R | Q | Total | Build |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Line Guardian: a mobile robot that watches the line and refills it** | A mobile manipulator (base plus arm, like Hello Robot Stretch in MuJoCo Menagerie) patrols line-side racks, counts what is left in each KLT with its camera, works out which bin runs out first, fetches the right KLT from the supermarket, checks the label, and delivers it before the line starves | The system of record often disagrees with the shelf. Existing line-side AMRs follow orders from the system; this robot trusts what it sees and re-prioritises live | 24 | 23 | 12 | 18 | 13 | 90 | Medium |
| 2 | Fleet Watch (fire mode for the logistics fleet) | See Round 5 | | 23 | 24 | 14 | 19 | 14 | 94 | Medium |
| 3 | Smart taxi | See Round 5 | | | | | | | 84 | Medium |
| 4 | Humanoid line-side loader | See Round 5 | | | | | | | 82 | Hard |
| 5 | Goods-in exception robot | Handles damaged pallets, wrong labels and shifted loads | Truck unloading is crowded | | | | | | 78 | Medium |

### Line Guardian in brief
- **Perceive:** camera on the arm reads KLT labels and estimates bin fill level; LiDAR and depth for navigation; it sees people and forklifts.
- **Decide:** "time until empty" per bin from what it sees, not from the system of record. Gemma picks the next job; a deterministic check confirms the part number before delivery.
- **Act:** drive, pick the KLT (a fixed-shape box, easier than small parts; in simulation a weld or suction constraint is acceptable if stated), place it on the rack, take empties back.
- **Unscripted change:** a teammate empties a bin suddenly (rush order), puts a wrong part in a supermarket slot, or a forklift blocks the aisle. The robot re-prioritises or re-routes on screen.
- **Unsure behaviour:** label unreadable or part looks wrong, so it does not deliver, flags it to the line lead, and goes to the next job.
- **Critical failure:** delivering the wrong part, which is worse than a late part because it can go into an engine. Label check plus never delivering on low confidence.
- **Autonomy argument:** weaker than fire. Factory Wi-Fi exists, but plants want on-premises control for safety, IP and latency. State an edge budget.
- **Your edge:** you know KLTs, supermarkets, taxis and buses from the Hams Hall work, and can use realistic numbers.
- **Weaknesses:** judges may see "another warehouse robot"; less emotional; Stretch picking in MuJoCo needs testing early.
- **Edge cases:** label hidden or dirty; two parts that look alike; bin half hidden; line-side slot full; two lines short at once (priority); narrow aisles and deadlock with other robots; battery charging; people stepping in front.

## Round 5: ideas from the user's BMW and logistics background
The user built `D:\Projects\bmw-hams-hall-sim`, a calibrated simulation of the Hams Hall engine plant's intralogistics (goods-in, VNA racking, automatic stores, tugger "taxis" and "buses" to the lines, forklifts). That is domain knowledge other teams will not have.

| Rank | Idea | What it does | Evidence | Weakness | Score | Build |
|---|---|---|---|---|---|---|
| 1 | **Fleet Watch: every robot becomes a scout** | When the fire alarm sounds in a plant or warehouse, the AMR and tugger fleet already inside (already mapped, already carrying LiDAR and cameras) switches to emergency mode. Robots move themselves out of fire exits and doorways, split the building between them, search racking aisles for workers who have not checked out, and report to the fire crew: "aisle 7 blocked, person seen at VNA aisle 3, 80%" | Atherstone-on-Stour warehouse fire, Warwickshire, Nov 2007: four firefighters died, many having gone in believing staff were trapped. UK lithium-ion battery fires: 1,760 in 2025, up 147% since 2022 (QBE). Idle or failed AGVs blocking emergency exits is a known hazard (safety vendor). BMW runs AMR fleets (idealworks) and named Hams Hall for further deployment (2023) | Normal AMRs are not heat-hardened, and LiDAR degrades in dense smoke, so this works in the early minutes. They are expendable and already inside | 94 | Medium |
| 2 | Smart taxi: tugger that re-plans in mixed traffic | Line-side tugger sees pedestrians, forklifts and dropped pallets, and re-routes instead of stopping dead | GB: 20 of 135 workplace deaths (2022/23) were people struck by moving vehicles; deadlock and stop-and-go in AMR aisles | Warehouse robots are crowded at hackathons | 84 | Medium |
| 3 | Humanoid line-side loader | Humanoid loads parts at the line and adapts when a part is misplaced | Figure 02 at BMW Spartanburg: 11 months, 1,250 hours, 90,000+ parts, 30,000+ X3s; 37-second load in an 84-second cycle, 99% per-shift target | Humanoid grasping in simulation in 22 hours is very risky; Figure already did it for real | 82 | Hard |
| 4 | Goods-in inspection rover | Scans pallet labels and damage at the drop zone instead of the 5-seconds-per-pallet manual check | From the user's sim process | Low drama, weak adaptation story | 78 | Easy |

### Why Fleet Watch fixes Smoke Search's biggest edge cases
| Smoke Search problem | Fleet Watch answer |
|---|---|
| Someone has to carry the robot up stairs | Robots are already in the building |
| No map, position drift in smoke | The plant map already exists; the fleet already localises against it |
| Doors | Plant doors are often automatic or open; robots move out of doorways themselves |
| Small flat: a crew with a thermal camera is as fast | Huge buildings with racking, where crews get lost: the robot's best case |
| One robot is slow | Coordinated fleet ("Coordinated fleets" is in the brief) splits the search |
| Who buys it? | Plants that already own the fleet: a software update, not new hardware |

### New edge cases for Fleet Watch
- The fleet's own lithium batteries are a fire source, so robots parked near the fire must move away from it first.
- Robots must never block evacuating people. Priority one is clearing exits and corridors.
- Workers' locations: the plant knows who is on shift (badge data) but not where they are, and badge data has privacy limits.
- Racking can collapse in fire, so robots avoid aisles under high heat.
- Normal AMR safety scanners stop the robot when anything is close. In emergency mode it still must not hit people.
- Wi-Fi may fail in a fire, so each robot keeps searching on its own and shares the map peer to peer when it can.

## Round 4: wide creative search (no boundaries)
Scores use the track weights (Perception 25, Adaptation 25, Autonomy 15, Relevance 20, Ambition 15).

| Rank | Idea | Who it helps | Key evidence | Existing work and gap | P | A | O | R | Q | Total | Build |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Smoke Search | Firefighters | 283 fire deaths in England (YE Dec 2025) | See above | 23 | 24 | 14 | 18 | 13 | 92 | Medium |
| 2 | **Never Clear: landmine-clearing rover** | Deminers, villagers returning home | Landmine Monitor 2025: 6,279 casualties in 2024, a four-year high; 90% civilians, almost half children. Ground radar is poor in wet or vegetated ground with high false alarms; metal detectors struggle with plastic mines | Drones survey surface mines; researchers call for uncertainty quantification. Gap: a ground robot that fuses metal detector, ground radar and camera into a probability map, and never drives on ground it has not scanned | 23 | 22 | 14 | 19 | 13 | 91 | **Easy** |
| 3 | Fell Swarm: mountain rescue drone fleet | Mountain rescue teams | Needs UK callout figures (verify) | Fits "Coordinated fleets". Gap: drones share the search area with no signal and re-split it when one drone's battery drops or it finds a clue | 21 | 23 | 14 | 17 | 14 | 89 | Medium |
| 4 | Dose Budget: nuclear decommissioning scout | Decommissioning engineers | Fukushima: radiation destroyed robots' wiring; Toshiba's scorpion robot aborted; Quince stranded after its cable disconnected | YC Salem Robotics does hazardous-site inspection. Gap: the robot treats its own lifetime radiation dose as a budget, maps hot spots and re-plans to come back alive | 22 | 22 | 14 | 17 | 13 | 88 | Easy |
| 5 | Bite: feeding arm for people with paralysis | People with spinal cord injury | Real need (Cornell "Feel the Bite", FEAST 2025) | Gap: adapts to head movement and spasms mid-bite and stops when unsure. Safety-critical, with food handled in simulation | 22 | 23 | 13 | 17 | 12 | 87 | Hard |
| 6 | Lava Tube Scout: Moon cave explorer | ESA and NASA mission planners | ESA cave-mission studies; NASA LUX (NIAC); LunarLeaper | Concepts rely on tethers or relays. Gap: fully autonomous mapping beyond relay range, re-planning after a rockfall, coming back to relay. Literally nothing to phone home to | 22 | 22 | 15 | 12 | 15 | 86 | Medium |
| 7 | Ghost Net Hunter: underwater drone | Ocean clean-up crews | WWF: 0.5–1 million tonnes of fishing gear lost each year (contested) | Most retrieval uses ROVs with pilots. Gap: autonomous sonar-plus-camera search as currents change. Underwater simulation is rough | 20 | 21 | 14 | 15 | 13 | 83 | Medium-hard |
| 8 | Lifeguard drone | Beach lifeguards | WHO: about 300,000 drowning deaths a year | Australian lifeguard drones exist (Little Ripper; unverified). Water simulation is weak | 20 | 21 | 12 | 18 | 12 | 83 | Medium-hard |
| 9 | Night ranger anti-poaching drone | Wildlife rangers | South Africa: 103 rhinos poached in Q1 2025 | Already widely used; it tracks people, which raises ethics questions | 20 | 20 | 14 | 15 | 11 | 80 | Medium |

### Never Clear (new number 2) in brief
- **Platform:** a light wheeled rover with a metal detector and ground radar on a front boom, plus a camera.
- **Golden rule, shown on screen:** it only drives on cells it has already scanned and marked safe. The rest stays grey.
- **Perceive:** metal detector (misses plastic mines), ground radar (fails in wet soil), camera (surface mines and tripwires). Each sensor's trust changes with conditions.
- **Unscripted change:** start rain mid-run, so the ground radar degrades. The robot slows, rescans and raises its "unsure" count. Or a teammate drops a surface object.
- **Unsure behaviour:** never says "clear", only "probability under X%". Anything above a threshold is flagged for a human deminer, and the robot never touches it.
- **Edge cases:** plastic mines with no metal, metal scrap causing false alarms, mines on slopes, the robot's own weight (keep it light and say so), GPS loss under trees.
- **Why it might beat Smoke Search:** easiest to simulate (sensor fields over a flat field, no smoke or fire rendering), an emotional and well-evidenced need, and a visual hook in the probability map.
- **Weakness:** less sci-fi, and judges may ask whether the sensor models are realistic.

## Smoke Search: how it works at a real fire

### Example: fire in a 7th-floor flat
1. Crews take the firefighting lift or stairs to a bridgehead. London Fire Brigade guidance (2011) says to leave the lift at least two floors below the fire floor, and breathing apparatus is required beyond the bridgehead. Check current LFB policy PN0633 before quoting distances.
2. A breathing-apparatus crew carries the rover (backpack-sized) up the stairs to the fire-floor lobby. The rover does not climb stairs.
3. The crew opens the flat's front door, which they must do anyway, and puts the rover down. The rover goes in while the crew are still getting the hose in.
4. The rover drives through the open space scanning continuously. It does not stop and scan each spot; it stops only for about 2 seconds at doorways to listen, and to confirm a possible person.
5. Its map streams to the crew and the entry-control board: "possible person, bedroom 2, 80%; kitchen hot; bedroom 1 door closed, not searched".

### Doors
- It cannot open doors. The crew opens the front door. Internal doors in flats are often open or ajar; the rover can push an ajar door but never forces one.
- A closed door is marked "not searched" and the crew goes there. That still helps: the crew searches 1 room instead of 4.
- Caution: opening doors lets air in. In UL tests the time from ventilation to flashover was 2 minutes in a modern home against more than 8 in a legacy one. Door control stays with the crew.

### Time: won't the fire spread first?
- Fires are fast. In FSRI burns, rooms with synthetic furnishings flashed over in about 3:20 to 4:50 after ignition; natural-furnished rooms mostly did not flash over within 30 minutes.
- So the rover must be fast and work in parallel with the crew, not instead of them. Target: a first pass of a flat in about 1 to 3 minutes (to be measured in simulation, not claimed).
- A room that has flashed over is not survivable and not the rover's job. The value is people in the other rooms, which are not yet involved.

### Why not just a firefighter with a thermal camera?
- Firefighters already carry thermal imaging cameras. They cannot see through walls; reflections confuse them; one hands-on test suggests a room search with a camera can take 2 to 3 times longer than expected; and disorientation was the leading cause of structural fireground deaths in 2007 (Firehouse).
- What the rover adds: it goes first without risking a crew member; it works in parallel; it stays low under the heat; it builds a map and a way out, which reduces disorientation; radar picks up breathing; and it keeps a record of what was searched.
- Honest answer: in a small flat, a good crew with a thermal camera may be as fast. The rover is more valuable in big or complex spaces where searches take long and crews get disoriented: care homes at night, hotels, offices, basements and car parks. Consider demoing a care-home corridor instead of a flat.

### Minimum version (is a thermal camera on wheels enough?)
Yes, for the core. MVP = thermal camera, radar rays and IMU on a rover, plus autonomous exploration, the map with confidence labels, and the stuck-and-heat safety rules. Hearing and Gemini voice are add-ons only if time allows.

## Smoke Search: engineering grill (what is it really, and does it survive?)

### Platform: small tracked ground robot (UGV)
| Option | For | Against | Verdict |
|---|---|---|---|
| Caged indoor drone (like Flyability Elios, used by Rotterdam Fire Brigade) | Fast, goes over debris | Hot gases collect at the ceiling, so a drone flies in the hottest layer; short battery life; rotor wash stirs up smoke | No. Also already sold, so not our gap |
| Quadruped (like Boston Dynamics Spot, used by FDNY) | Climbs stairs and rubble | One report says fire scenes can exceed 500 °C and no robot dog can withstand that; expensive | No |
| Humanoid | None here | Slow, fragile, falls over | No |
| **Small low tracked rover** | Stays near the floor, below the hot gas layer that forms under the ceiling before flashover; tracks handle debris; cheap enough to lose | Struggles with stairs and tall rubble | **Yes** |

### Sensors: radar over LiDAR (corrected)
- Evidence: LiDAR range errors appear once visibility falls below about 4 m; radar located targets within about 1% regardless of visibility (fire-service test). In SmokeBot's Dortmund test, LiDAR and cameras stopped giving usable signals when the building filled with smoke, and radar kept the robot running.
- Stack: mmWave radar (navigation through smoke), thermal camera (finds people and fire), IMU plus track odometry (dead reckoning), microphone (calls for help), and RGB and LiDAR used only while the air is clear.
- In simulation: MuJoCo rangefinder rays for both. LiDAR rays drop out in proportion to smoke density; radar rays are sparse and noisy but ignore smoke. Thermal comes from a noisy, low-resolution segmentation render.

### Navigation
- Localise: radar-inertial odometry (in sim, noisy pose from rays plus IMU).
- Map: occupancy grid plus a heat cost map (thermal radiation field from arXiv 2602.19108, recomputed as the fire spreads).
- Decide: frontier exploration ranked by "chance of a person" and "heat risk". Gemma 4 E2B runs locally for high-level choices; the planner (A*) runs every tick.
- No signal needed. When the radio drops, it keeps working and uploads the map when it is back in range. Possible extension: drop relay nodes like NIST's "breadcrumb" relays and DARPA LANdroid.

### Will it melt?
- Honest scope: a scout for smoke-logged rooms before flashover, not a robot for rooms on fire. NIST: flashover starts at gas temperatures of at least 400 °C and reaches about 1,000 °C. That is beyond our target, and beyond Spot's.
- Before flashover, the hot layer sits near the ceiling and the floor stays cooler. Dalmarnock tests define the layer boundary near the 100 °C line. A low rover lives under it.
- Safety rule: if body or air temperature at robot height rises past a set limit, it retreats along its known-safe path. The heat cost map also keeps it away from radiant sources. Ordinary electronics are typically rated to about 70–85 °C (general engineering figure, not from a cited source), so a real build needs an insulated shell. Say so under Limits.
- History: at the World Trade Center in 2001, heat could melt rescue robots' wheels and tethers got stuck (CNN transcript, responder account). Our design answers both: no tether, and heat-aware routing.

### Something falls on it or it gets stuck
- Detect: the tracks say "moving" but the IMU says "not moving", meaning it is stuck. An IMU spike plus tilt means it was hit or flipped.
- Respond: back out, mark the cell impassable, re-plan. If it is trapped for good, it becomes a fixed beacon: it keeps broadcasting its map and marks its own position as a hazard.
- In the video: a teammate drops debris on the robot or in its path in the viewer, which is the unscripted change.

### What it will not do (say this in the pitch)
Not firefighting (no water), not flashover rooms, not stairs in this version, and it never declares an area "clear", only "searched, no one found, confidence X%".

## Round 3: deep research, ideas built on published papers

### 1. Smoke Search v2, built on VULCAN and the Spot thermal-field paper (recommended)
- **VULCAN** (arXiv 2604.12831, Apr 2026): multi-robot fire search in Habitat. It fuses RGB-D, thermal and mmWave radar into a hazard map, a VLM picks frontier goals, and a hazard-aware planner drives. Its own gaps: "preliminary", only baselines evaluated, the VLM is not named, no hardware or latency figures, chairs and sofas stand in for humans, and no code released.
- **Understanding Fire Through Thermal Radiation Fields** (arXiv 2602.19108, Feb 2026): Spot with a thermal camera. It clusters points above 100 °C, estimates radiative power with Stefan-Boltzmann, spreads it over a 10 cm grid by inverse square with line-of-sight checks, and plans with A* over a thermal-plus-LiDAR cost map. Its gaps: one static fire, no smoke, no dynamic flames.
- **Our build closes those gaps:** a single rover running end to end; a named local model (Gemma 4 E2B) on a stated device with measured latency; the thermal radiation cost map recomputed live as the fire spreads, so the path re-plans visibly; smoke degrades RGB and the robot shifts trust to thermal and LiDAR; people are actual human models; a "confirmed" or "unconfirmed" label with a second-look behaviour.
- **Optional add-on, "it hears you":** a simulated microphone picks up calls for help (built on Sky-Ear, arXiv 2604.12455, and Fraunhofer LUCY). The robot stops its motors to listen, which cuts self-noise. This adds a fourth sensor and an emotional moment in the video.
- **Judge roast:** "VULCAN already did this." No: VULCAN is a preliminary paper with no working system or numbers. We ship the working version and report latency. "Smoke is fake." Every simulated sensor is a model; say so, and add noise.

### 2. Sniffer: gas leak hunter, built on SniffySquad
- **SniffySquad** (arXiv 2411.06121): multi-robot gas source localization. Its key finding: gas arrives in patches, and robots get trapped thinking a patch is the source. It used OpenFOAM and GADEN simulation, which are heavy ROS tools.
- **Our twist:** a single robot that doubts itself. A high reading triggers a check: the robot looks with its camera for a leaking valve and re-samples upwind before declaring a source. The unscripted change is a wind shift or a second leak opened live.
- **Need:** industrial gas leaks, sites where wireless is not allowed (ATEX zones), and early small leaks that satellites miss (World Oil, Mar 2026). Statistics need verifying.
- **Judge roast:** gas is invisible, so the video needs a strong heatmap overlay. A cheap puff-and-plume model in numpy stands in for GADEN, so say that under Limits. Novelty is high: no hackathon project like it surfaced in my searches.

### 3. Lost & Found v2, built on MIT DAAAM
- Gap: DAAAM remembers but never doubts. We add confidence that decays over time, re-search when wrong, a dementia user, and a stated edge budget. Safest build, lowest sci-fi factor.

### 4. Ears: acoustic survivor search in rubble (better as an add-on to idea 1)
- Built on Sky-Ear, SpaceNet (MDPI Sensors 2025), Fraunhofer LUCY and the hose-shaped rescue robot work on self-noise. Gap: drones cannot enter small voids, so a ground robot can. Strong idea, but visually weak on its own. Fold it into Smoke Search.

### 5. Limp Home: robot that adapts when damaged mid-mission
- Inspired by "Robots that can adapt like animals" (Cully et al., Nature 2015). A wheel motor fails mid-run; the robot detects the mismatch between its IMU and wheel odometry and compensates. Very sci-fi, and strong on adaptation, but its perception of the world is weak. Possible 20-minute stretch feature for Smoke Search ("debris damages a wheel").

### Ranking
1. Smoke Search v2 with acoustic add-on: best-evidenced gap, two 2026 papers to build on, all four judging criteria visible in one video.
2. Sniffer: most original, harder to make visual.
3. Lost & Found v2: safest backup.
4. Ears: merge it into 1.
5. Limp Home: stretch feature only.

## What already exists, and the gap we would build on
| Idea | Already built | Gap we fill |
|---|---|---|
| Smoke Search | Boston Dynamics Spot and tracked robots used by fire services; Örebro SmokeBot (radar, laser, thermal and gas mapping, but its developers say it may be too slow for urgent rescues); a UC San Diego stereo-plus-thermal 3D mapping robot (Feb 2026); YC company Salem Robotics (autonomous inspection of hazardous sites such as nuclear) | Fire robots today are mostly teleoperated, and the literature says remote control in smoke is the bottleneck: operators lose situational awareness, and novices struggle. We build the autonomy layer: the robot decides which sensor to trust as smoke rises, searches by itself, and hands crews a map that marks "confirmed" and "unconfirmed" victims. This is the "variable autonomy" direction researchers propose (Frontiers, 2024). |
| Lost & Found | MIT DAAAM (persistent language-and-3D object memory; "not ready for consumer products"); CMU/Meta SemExp (common-sense object search) | DAAAM remembers but does not doubt itself. We add memory that knows when it is stale (confidence decays, and the house knows people move things), active re-search when wrong, a dementia user, and a stated edge budget. |
| Night Watch | Rhem Labs (YC Fall 2026, home care robot reportedly using mmWave radar for falls); EBO Max FamilyBot; Vayyar, Kubo Care and Nobi fixed sensors; Guardia wristband | Crowded. The only gap: a robot that goes to verify a fixed sensor's alert, cutting false alarms. That is weak for a hackathon. |
| Guide | University of Glasgow RoboGuide, Binghamton robot guide dog, Guide-LLM paper | Little gap, and hard to simulate. |
| Blade Scout | Commercial inspection drones | Little gap within the time we have. |

Takeaway: every idea exists in some form, and hackathon winners (matcha bot, waste sorter) were not new categories either. Judges reward a sharp improvement on a real gap, shown working. Smoke Search has the best-evidenced gap: teleoperation fails in smoke, and autonomy is the fix researchers are asking for.

## Rank 1: Smoke Search. Fire-scene scout rover. Estimated score 90/100. Feasibility: medium-high.
**What it is:** a rover enters a smoke-filled building ahead of firefighters. It finds people, maps hazards, and reports where crews should go. Smoke blinds its camera, so it switches to its other sensors.
- Perceive: RGB camera (degraded by MuJoCo fog), simulated thermal camera (a segmentation render where people and fire are "hot"), and LiDAR rays.
- Decide: frontier exploration plus a victim-and-hazard map. A local Gemma 4 E2B model chooses priorities from the fused state ("possible person in room B, heat rising in the corridor, go via room C").
- Act: navigates, marks victims, and announces a route for crews.
- Unscripted change: during the take, a teammate drops a "collapsed" box across the corridor in the viewer, or the fire spreads. The rover re-plans.
- Unsure behaviour: when the camera and thermal readings disagree, it slows down, gets a second view, and labels the find "unconfirmed" instead of "victim".

**Judge roast**
- "A thermal camera made from a segmentation mask is cheating." Every simulated sensor is a model, so say that under Limits. Add noise, blur and a low resolution (for example 80x60) so it behaves like a cheap thermal sensor.
- "Who buys this? Fire services already use drones and Spot robots." Evidence needed (verify): firefighter injuries during search, and the time crews spend searching blind. Position it as a small throwaway scout sent in first, not as a replacement for crews.
- "Is the smoke realistic?" No. The point is sensor degradation and the switch to other sensors, which is the real problem.
- "Why does it need to run locally?" It is the strongest autonomy argument of the five: inside a burning building there is no network.
- Sensitivity: London judges will think of high-rise fires. Be respectful and never use a real tragedy as a hook.

**Improvements:** show a confidence bar per sensor on screen, and when smoke rolls in, show the camera weight dropping to zero. That makes "perception-driven" impossible to miss.

## Rank 2: Lost & Found. Home robot that remembers where things are. Estimated score 85/100. Feasibility: high.
**What it is:** a patrol robot for someone with early dementia. It keeps a memory of where it last saw everyday items. When asked "where are my glasses?", it goes there, checks, and if they have moved, searches likely spots chosen with common sense.
- Fits "Learning on the fly" and "Assistive perception" from the brief.
- Unscripted change: a teammate moves the glasses in the viewer mid-search. The robot arrives, says "not here any more", updates its memory and searches again. That is exactly the "corrects mid-run" behaviour.

**Judge roast**
- "This is object detection plus a dictionary." Ambition comes from the search reasoning: which rooms are likely and why, and how confidence decays with time since last seen. Show that reasoning as on-screen text.
- "AirTags exist." They need every item tagged, and people with dementia lose the tagged ones too. Evidence needed (verify): UK dementia prevalence and misplaced items as a reported daily burden.
- "Its only action is talking." The brief says "act" includes "communicate", but add pointing or leading the user to the item.
- Simulated household objects may be hard to recognise. Use Isaac Sim home assets if the A2000 runs it. Otherwise use distinct objects in MuJoCo plus Gemma vision.

**Improvements:** the memory map is the visual hook. Show stale memories fading on screen.

## Rank 3: Night Watch. Fall responder for older people living alone. Estimated score 83/100. Feasibility: high.
**What it is:** patrols at night, finds someone on the floor, checks whether they respond, then escalates.

**Judge roast**
- The killer question: "Why a robot and not a £30 fixed sensor or a pendant?" Without a strong answer, relevance drops. The best answer: it goes to the person, talks to them, brings the phone, and one device covers the whole home.
- Fall detection is a crowded idea, so novelty is low.
- A simulated human lying on the floor may not be detected. Test this in hour one.
- Evidence needed (verify): UK falls among people over 65, and how long people lie on the floor before help arrives.

**Improvements:** merge it with Lost & Found as one "home companion". This is risky because two behaviours take twice the work, so do it only if one is finished early.

## Rank 4: Guide. Robot guide for a blind traveller in a station. Estimated score 86 if built well. Feasibility: low.
**Judge roast**
- Strong need, and it matches "Assistive perception" in the brief.
- Simulating the human who follows the robot is awkward, and a fake follower will look fake on video.
- University of Glasgow's RoboGuide and Binghamton's robot guide dog already exist, so novelty is low.
- Hazards such as platform edges and crowds are hard to simulate well in 22 hours.

**Improvement if chosen:** model the person as a leashed body that the robot must keep safe, and make the platform edge the critical failure mode.

## Rank 5: Blade Scout. Offshore wind turbine inspection drone. Estimated score 80/100. Feasibility: low.
**Judge roast**
- Good autonomy argument: there is no network offshore. MuJoCo Menagerie includes the Skydio X2 drone model.
- Cracks painted on textures look like a toy, and judges will doubt the perception.
- Drone control plus wind gusts takes a large share of the time.

**Verdict:** drop it unless someone already has drone control experience.

## Rejected
Warehouse fleets (FleetMind, CarphaCom and AdaptiFleet already did them), quadruped search and rescue (the walking controller alone takes a day), and grasping tasks (manipulation is too risky in 22 hours).
