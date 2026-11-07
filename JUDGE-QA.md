# Smoke Search: Weaknesses and Judge Q&A

Prepared 7 November 2026. Each answer is honest about limits. Figures are as reported by the linked sources (see EVIDENCE.md, EDGE-CASES.md and IDEAS.md for the full lists).

## Biggest weaknesses, worst first
1. **There is no hard evidence that robots save lives in house fires.** The need (deaths, disorientation, slow searches) is evidenced; the robot's benefit is a hypothesis. Say this plainly and frame the demo as testing it.
2. **A crew with a thermal camera may be as fast in a small flat.** The robot's value grows with building size and complexity (care homes, hotels, offices, basements).
3. **Its own battery is a fire risk.** Lithium-ion cells can start self-heating at about 48 °C when fully charged, and runaway tends to occur around 130–150 °C (RSC 2026; Warwick). A robot that catches fire inside a fire is a disaster.
4. **It cannot open doors or climb stairs.** It depends on the crew for both.
5. **Trust can fail both ways.** In a mock-fire study, people kept following a robot even after it proved unreliable. In an Italian tunnel exercise, firefighters took manual control within a few metres. Both are real risks.
6. **The simulation is not physics.** MuJoCo fog is not smoke, and a segmentation render is not a thermal camera.
7. **VULCAN (arXiv, April 2026) is close.** Our claim is a working end-to-end system with measured numbers, not a new concept.
8. **Cost and adoption.** UK fire services have tight budgets, and procurement is slow.

## Edge cases from this round
| Edge case | Evidence | What the robot does |
|---|---|---|
| Children hide under beds and in wardrobes; bathrooms get skipped | Fire Engineering and CBS: crews expect children under beds and in closets; many victims are found in bathrooms that searches skip | Low enough to look under beds (a strength). Bathrooms and wardrobes are high-priority targets. A closed wardrobe is "not searched" |
| Most fire deaths happen in bedrooms at night | USFA: about half of residential fire deaths happen in bedrooms, between roughly 10–11 pm and 6–7 am. 2017–2019: 31% of victims were asleep and 37% were trying to escape | Search priority: bedrooms first, then the hallway and exit route, where people collapse while escaping |
| Care homes: residents cannot walk out, and few staff are on at night | A fire officer judged night staffing made safe evacuation of all residents impossible (Leeds case); Rosepark 2004, 14 deaths | Strong use case. The robot tells crews which rooms still have people in them |
| Battery overheats | Above | The battery temperature is a hard limit: retreat well before self-heating starts. Use LFP chemistry, which is the most thermally stable. Insulate the pack |
| A hiding child is scared of the robot | Children are told to hide | Speaker: "Firefighters are coming. Stay low. Call out." Soft light, not a strobe pointed at faces |
| Firefighters crawl and search by touch in smoke, so the robot is a trip hazard | Zero visibility for over 15 minutes is reported | Beeps when it is near people moving at crawl height, parks against walls, and appears on the crew's map |
| Pets look like people | A dog is warm and moves | Body-size and shape check; label "animal, possible"; still reported |
| Steam from hose spray blinds the thermal camera | Water on hot surfaces | Same rule as smoke: trust shifts to radar; slow down |
| Mirrors and glass reflect heat | Thermal-imaging guidance warns of reflections | Requires radar or audio agreement before "confirmed" |
| Slow start-up: the model takes time to load | A Gemma load can take tens of seconds | The robot boots and keeps the model loaded while being carried up. Target: moving within 10 seconds of being put down (to measure) |
| The compute is a laptop, not a robot | Our demo runs on the A2000 laptop | State the target edge device (Jetson Orin class) and the measured budget on our laptop. Do not claim Jetson numbers we have not measured |
| Crews over-trust the map | Mock-fire over-trust study | The map gives evidence ("heat blob plus breathing") and confidence, never orders. It never says "clear" |
| Crews ignore it | Tunnel exercise manual takeover | One-screen map, no chatter. Manual drive mode always available |
| A floor flashes over while it is inside | FSRI: modern rooms flash over in about 3–5 minutes | Retreats on heat rise; the robot is expendable, people are not |

## Judge Q&A (short answers)
- **"Why not a firefighter with a thermal camera?"** Cameras cannot see through walls, reflections confuse them, and searches with a camera can take 2–3 times longer than expected. Disorientation was the leading cause of fireground deaths in 2007. The robot goes first, works in parallel and maps the way out. In a small flat a good crew may be as fast, so we target care homes and larger buildings.
- **"Won't it melt?"** It works before flashover, low down under the hot layer, and retreats on a temperature limit. The first thing to fail is its battery, so the battery temperature is the limit. A real build needs an insulated shell and LFP cells.
- **"How does it open doors?"** It does not. The crew opens the front door. Closed inside doors are marked "not searched", so the crew goes straight to them.
- **"Won't the fire spread while it scans?"** It scans while driving and only pauses at doorways. It works alongside the crew, not before them. The target is a first pass of a flat in 1–3 minutes, measured in simulation.
- **"What if it is wrong?"** Every find has a confidence label, and it never declares a room clear. A crew member always makes the rescue decision.
- **"What runs where?"** On the robot: Gemma 4 E2B, the planner and the safety rules, all offline. In the cloud, only when there is signal: the commander's voice link and Gemini 4 Argon briefing, checked by our planner.
- **"What is the critical failure mode?"** Missing a person. Mitigations: three sensors (thermal, radar, audio), search priority by where victims are found, "not searched" labels, and never saying "clear". Shown on screen with a silent person behind furniture.
- **"Isn't this just VULCAN?"** VULCAN is a preliminary paper with no working system or latency figures. We built it end to end and measured it.
- **"How realistic is the simulation?"** Not physically. Smoke is modelled as sensor degradation, and thermal is a noisy render. We say so, and the decision logic is what we are demonstrating.
